//! Wraps `main` with a single `Blob<Field; N>` input to keep its parameter count independent of
//! input size. The wrapper writes parameters to the witness, reconstructs their typed values
//! (range-checking integers), initializes globals, calls `main`, binds its return, then deinitializes
//! globals. This pass also prepares unconstrained call arguments and witnesses their results.
//!
//! # Reference Arguments
//!
//! Argument lowering relies on Noir's monomorphizer rejecting mutable references with
//! `ConstrainedReferenceToUnconstrained`, and nested references and references inside tuples,
//! structs, arrays, or closure environments with `NestedOrContainerReferenceToUnconstrained`.
//! Only a direct immutable `&T` with reference-free `T` can cross this boundary. That guarantee
//! makes copying the pointee sound and checking only top-level `Ref` parameters complete:
//! HLSSA's `Ref<T>` itself does not retain mutability. Ordinary calls keep their original
//! signatures and reference identity.
//!
//! This runs before witness inference and points-to analysis, so neither needs to convert or
//! reason about references crossing an unconstrained boundary. Only static calls carry the
//! `unconstrained` flag; dynamic calls are not boundary calls in this IR.
//!
//! # Witness Layout
//!
//! Witness columns are allocated in execution order, so the wrapper's write order is the layout:
//!
//! ```text
//! [ 0: constant one | public parameters (ABI order, pinned) | private parameters (ABI order,
//!   unpinned) | ...main's internal witnesses... | computed return (pinned) ]
//! ```
//!
//! The constant one, public parameters and all return leaves are pinned. Each return column is
//! constrained equal to its computed value; an optional `return` in `Prover.toml` is compared
//! against it by the host after witness generation. A `call_data` parameter is written with the
//! private ones, and a `return_data` return is bound like any other.
//!
//! Private parameter writes feed a per-parameter array through a counted loop. Its dynamic index
//! prevents splitting, so DCE retains or elides the whole parameter; a one-field parameter is
//! written directly, since a one-element array cannot split. Integer range checks keep
//! otherwise unused parameters live. The driver rejects partially surviving parameters.
//! [`entry_param_origin`] and [`ENTRY_RETURN_ORIGIN`] tag writes so the R1CS generator can recover
//! their columns after DCE.

use std::{cell::RefCell, collections::BTreeSet, rc::Rc};

use crate::{
    collections::HashMap,
    compiler::{
        pass_manager::{AnalysisStore, Pass},
        passes::{
            shared::{limbs::widest_injective_int_bits, unsupported::unsupported_on_this_field},
            wide_witness_ints::multi_cell_int_bits,
        },
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

pub const ENTRY_ORIGIN_PREFIX: &str = "entry point: ";

pub fn entry_param_origin(index: usize) -> String {
    format!("{ENTRY_ORIGIN_PREFIX}param {index}")
}

pub const ENTRY_RETURN_ORIGIN: &str = "entry point: return";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlobSlot {
    pub offset: usize,
    pub width: usize,
    /// The ABI's `pub`, or every parameter of a hand-built SSA.
    pub public: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryBlobLayout {
    pub params: Vec<BlobSlot>,
    /// Present exactly when the ABI declares a return, so `Some(0)` is a zero-width return.
    pub return_width: Option<usize>,
    pub total: usize,
}

pub struct PrepareEntryPoint {
    main_is_unconstrained: bool,
    abi_has_return: Option<bool>,
    /// `None` treats every parameter as public, which keeps hand-built SSA on fixed columns.
    param_is_public: Option<Vec<bool>>,
    layout_sink: Option<Rc<RefCell<Option<EntryBlobLayout>>>>,
}

struct PrepareFnEntry {
    hlssa_type: Type,
    fn_id: FunctionId,
}

struct ReconstructFnEntry {
    hlssa_type: Type,
    fn_id: FunctionId,
}

struct ReferenceArgumentWrapper {
    function: FunctionId,
    ref_params: Vec<bool>,
}

impl ReferenceArgumentWrapper {
    fn new(ssa: &mut HLSSA, callee: FunctionId) -> Option<Self> {
        let function = ssa.get_function(callee);
        let params = function.get_param_types();
        let ref_params: Vec<_> = params
            .iter()
            .map(|ty| matches!(ty.expr, TypeExpr::Ref(_)))
            .collect();
        if !ref_params.iter().any(|is_ref| *is_ref) {
            return None;
        }
        let returns = function.get_returns().to_vec();
        let name = format!("{}_ref_wrapper", function.get_name());
        let (function, ()) = HLSSABuilder::new(ssa).add_function(name, |fb| {
            for ty in &returns {
                fb.function.add_return_type(ty.clone());
            }
            let entry = fb.function.get_entry_id();
            let mut e = fb
                .block(entry)
                .with_source_location(SourceLocation::synthetic("lower_reference_arguments"));
            let args = params
                .iter()
                .map(|ty| match &ty.expr {
                    TypeExpr::Ref(inner) => {
                        let value = e.add_parameter(inner.as_ref().clone());
                        e.alloc(value)
                    }
                    _ => e.add_parameter(ty.clone()),
                })
                .collect();
            let results = e.call(callee, args, returns.len());
            e.terminate_return(results);
        });
        Some(Self {
            function,
            ref_params,
        })
    }
}

impl Pass for PrepareEntryPoint {
    fn name(&self) -> &'static str {
        "prepare_entry_point"
    }

    fn run(&self, ssa: &mut HLSSA, _store: &AnalysisStore) {
        let layout = Self::wrap_main(
            ssa,
            self.main_is_unconstrained,
            self.abi_has_return,
            self.param_is_public.as_deref(),
        );
        if let Some(sink) = &self.layout_sink {
            *sink.borrow_mut() = Some(layout);
        }
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
            param_is_public: None,
            layout_sink: None,
        }
    }

    /// The Noir ABI distinguishes unit (no public return) from an empty struct even
    /// though both have an empty-tuple HLSSA result. Hand-built SSA defaults to its signature.
    pub fn with_abi_return(mut self, has_return: bool) -> Self {
        self.abi_has_return = Some(has_return);
        self
    }

    pub fn with_param_visibility(mut self, param_is_public: Vec<bool>) -> Self {
        self.param_is_public = Some(param_is_public);
        self
    }

    pub fn with_layout_sink(mut self, sink: Rc<RefCell<Option<EntryBlobLayout>>>) -> Self {
        self.layout_sink = Some(sink);
        self
    }

    fn wrap_main(
        ssa: &mut HLSSA,
        main_is_unconstrained: bool,
        abi_has_return: Option<bool>,
        param_is_public: Option<&[bool]>,
    ) -> EntryBlobLayout {
        let original_main_id = ssa.get_unique_entrypoint_id();
        let original_main = ssa.get_unique_entrypoint();
        let param_types = original_main.get_param_types();
        let return_types = original_main.get_returns().to_vec();

        let globals_init_fn = ssa.get_globals_init_fn();
        let globals_deinit_fn = ssa.get_globals_deinit_fn();

        ssa.get_unique_entrypoint_mut()
            .set_name("original_main".to_string());

        let has_return = abi_has_return.unwrap_or(!return_types.is_empty());
        if !has_return && !return_types.iter().all(Self::is_leafless_result) {
            ice!("ABI without a return cannot discard nonempty SSA results");
        }
        let param_is_public: Vec<bool> = match param_is_public {
            Some(flags) => {
                if flags.len() != param_types.len() {
                    ice!(
                        "{} parameter visibilities supplied for an entry point with {} parameters",
                        flags.len(),
                        param_types.len()
                    );
                }
                flags.to_vec()
            }
            None => vec![true; param_types.len()],
        };
        let mut reconstruct_fns = Vec::new();
        for typ in &param_types {
            Self::get_or_create_reconstruct_fn(typ, ssa, &mut reconstruct_fns);
        }

        let layout =
            Self::entry_blob_layout(&param_types, &return_types, has_return, &param_is_public);
        let total_fields = layout.total;

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

            // The per-parameter array keeps private writes live or dead together under DCE.
            let emit_param_writes = |e: &mut HLBlockEmitter<'_>, slot: BlobSlot, pinned: bool| {
                let BlobSlot { offset, width, .. } = slot;
                let initial_array = Self::emit_default_witness_array(e, &Type::field(), width);
                if width == 0 {
                    return initial_array;
                }
                let write = |e: &mut HLBlockEmitter<'_>, elem: ValueId| {
                    if pinned {
                        e.pinned_write_witness(elem)
                    } else {
                        e.write_witness(elem)
                    }
                };
                if width == 1 {
                    // A one-element array cannot be split, so no loop is needed to keep it whole.
                    let index = e.int_const(IntBits::from_u128(32, offset as u128));
                    let elem = e.array_get(blob_param, index);
                    let witness = write(e, elem);
                    let zero = e.int_const(IntBits::zero(32));
                    return e.array_set(initial_array, zero, witness);
                }
                let copied = e.build_counted_loop(
                    width,
                    vec![(initial_array, Type::field().array_of(width))],
                    |e, i, accumulators| {
                        let source = if offset == 0 {
                            i
                        } else {
                            let base = e.int_const(IntBits::from_u128(32, offset as u128));
                            e.uadd(base, i)
                        };
                        let elem = e.array_get(blob_param, source);
                        let witness = write(e, elem);
                        let updated = e.array_set(accumulators[0], i, witness);
                        vec![updated]
                    },
                );
                copied[0]
            };

            // Public parameters first, then private, each loop located at its parameter's
            // origin so the R1CS generator can recover the columns after DCE.
            let mut witness_arrays: Vec<Option<ValueId>> = vec![None; param_types.len()];
            for pinned in [true, false] {
                for (index, slot) in layout.params.iter().enumerate() {
                    if slot.public != pinned {
                        continue;
                    }
                    let origin = SourceLocation::synthetic(entry_param_origin(index));
                    witness_arrays[index] =
                        Some(e.emit_with_location(origin, |e| emit_param_writes(e, *slot, pinned)));
                }
            }

            let mut arg_values = Vec::new();
            for ((typ, slot), witness_array) in
                param_types.iter().zip(&layout.params).zip(&witness_arrays)
            {
                debug_assert_eq!(slot.width, Self::flattened_field_count(typ));
                let witness_array =
                    witness_array.expect("every parameter has its witness array emitted");
                let value = if slot.width == 0 {
                    Self::emit_default_witness_value(&mut e, typ)
                } else {
                    match &typ.expr {
                        TypeExpr::Field => {
                            let zero = e.int_const(IntBits::zero(32));
                            e.array_get(witness_array, zero)
                        }
                        TypeExpr::Int(_) | TypeExpr::Array(_, _) | TypeExpr::Tuple(_) => {
                            let fn_id = Self::find_reconstruct_fn(typ, &reconstruct_fns);
                            e.call(fn_id, vec![witness_array], 1)[0]
                        }
                        _ => todo!("Not implemented yet"),
                    }
                };
                arg_values.push(value);
            }

            if let Some(init_fn) = globals_init_fn {
                e.call(init_fn, vec![], 0);
            }

            let results = if main_is_unconstrained {
                e.call_unconstrained(original_main_id, arg_values, return_types.len())
            } else {
                e.call(original_main_id, arg_values, return_types.len())
            };
            if has_return {
                e.emit_with_location(SourceLocation::synthetic(ENTRY_RETURN_ORIGIN), |e| {
                    for (result, return_type) in results.iter().zip(return_types.iter()) {
                        Self::bind_return_deep(e, *result, return_type);
                    }
                });
            }

            if let Some(deinit_fn) = globals_deinit_fn {
                e.call(deinit_fn, vec![], 0);
            }

            e.terminate_return(vec![]);
        });
        ssa.set_unique_entrypoint(wrapper_id);
        layout
    }

    fn bind_return_deep(b: &mut HLBlockEmitter<'_>, result: ValueId, typ: &Type) {
        match &typ.expr {
            TypeExpr::Array(inner, size) => {
                for i in 0..*size {
                    let index = b.int_const(IntBits::from_u128(32, i as u128));
                    let result_elem = b.array_get(result, index);
                    Self::bind_return_deep(b, result_elem, inner);
                }
            }
            TypeExpr::Tuple(element_types) => {
                for (i, elem_type) in element_types.iter().enumerate() {
                    let result_elem = b.tuple_proj(result, i);
                    Self::bind_return_deep(b, result_elem, elem_type);
                }
            }
            _ => {
                let result_field = b.ensure_field(result, typ);
                // UntaintControlFlow strips witness taint from the hint; R1CS gets a fresh column.
                let written = b.pinned_write_witness(result_field);
                b.assert_eq(result_field, written);
            }
        }
    }

    /// Lower reference arguments and prepare witness results in one call-site sweep.
    fn process_unconstrained_calls(ssa: &mut HLSSA) {
        // Collect distinct callees once, including those without reference parameters.
        // Snapshot callers before adding wrappers and result preparation functions.
        let func_ids: Vec<FunctionId> = ssa.get_function_ids().collect();
        let callees: BTreeSet<_> = ssa
            .iter_functions()
            .flat_map(|(_, function)| function.get_blocks())
            .flat_map(|(_, block)| block.get_instructions())
            .filter_map(|instruction| match instruction {
                OpCode::Call {
                    function: CallTarget::Static(callee),
                    unconstrained: true,
                    ..
                } => Some(*callee),
                _ => None,
            })
            .collect();
        let mut callee_return_types: HashMap<FunctionId, Vec<Type>> = HashMap::default();
        let mut ref_wrappers = HashMap::default();
        for callee in callees {
            callee_return_types.insert(callee, ssa.get_function(callee).get_returns().to_vec());
            if let Some(wrapper) = ReferenceArgumentWrapper::new(ssa, callee) {
                ref_wrappers.insert(callee, wrapper);
            }
        }

        let mut prepare_fns = Vec::new();
        for return_types in callee_return_types.values() {
            for return_type in return_types {
                if !Self::is_leafless_result(return_type) {
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
                            args,
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

                            if let Some(wrapper) = ref_wrappers.get(callee_id) {
                                if args.len() != wrapper.ref_params.len() {
                                    ice!(
                                        "Unconstrained call to {callee_id:?} has {} arguments, \
                                         but its reference wrapper expects {}",
                                        args.len(),
                                        wrapper.ref_params.len(),
                                    );
                                }
                                for (arg, is_ref) in args.iter_mut().zip(&wrapper.ref_params) {
                                    if *is_ref {
                                        let result = fb.fresh_value();
                                        new_instructions.push(
                                            OpCode::Load { result, ptr: *arg }
                                                .locate(location.clone()),
                                        );
                                        *arg = result;
                                    }
                                }
                                *callee_id = wrapper.function;
                            }

                            let original_results = results.clone();
                            let fresh_results = original_results
                                .iter()
                                .zip(return_types)
                                .map(|(original, typ)| {
                                    if Self::is_leafless_result(typ) {
                                        *original
                                    } else {
                                        fb.fresh_value()
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
                                // Keep the original call result for leafless values:
                                // it is already defined, and has no witness fields to prepare.
                                if Self::is_leafless_result(return_type) {
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

    /// Whether tuple elision will erase this result entirely. Array length does
    /// not affect the number of SSA leaves: `[Field; 0]` still has one array leaf.
    /// Slices retain their length through `LowerZstSlices`, so they need preparation.
    fn is_leafless_result(typ: &Type) -> bool {
        match &typ.expr {
            TypeExpr::Tuple(fields) => fields.iter().all(Self::is_leafless_result),
            TypeExpr::Array(inner, _) => Self::is_leafless_result(inner),
            TypeExpr::WitnessOf(inner) => Self::is_leafless_result(inner),
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

    /// Witness columns are not derived from blob offsets: the wrapper writes public parameters
    /// before private ones, and DCE may elide an unused private parameter.
    pub(crate) fn entry_blob_layout(
        param_types: &[Type],
        return_types: &[Type],
        has_return: bool,
        param_is_public: &[bool],
    ) -> EntryBlobLayout {
        let mut offset = 0usize;
        let mut slot = |width: usize, public: bool| {
            let start = offset;
            offset += width;
            BlobSlot {
                offset: start,
                width,
                public,
            }
        };
        let params = param_types
            .iter()
            .zip(param_is_public)
            .map(|(typ, public)| slot(Self::flattened_field_count(typ), *public))
            .collect();
        let return_width =
            has_return.then(|| return_types.iter().map(Self::flattened_field_count).sum());
        EntryBlobLayout {
            params,
            return_width,
            total: offset,
        }
    }

    /// How many elements of the entry point's input blob a value of `typ` occupies.
    ///
    /// An integer takes one: a parameter is written to a single witness column and range-checked
    /// to its width there. A range check bounds a _field element_, so one slot carries any integer
    /// up to `widest_injective_int_bits`.
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
                    (!Self::is_leafless_result(elem_type))
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
            // `WideWitnessInts` splits a whole value written this wide into range-checked limbs.
            TypeExpr::Int(size) if *size > multi_cell_int_bits(e.field()) => {
                e.write_witness(value_id)
            }
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
                // This ABI carries each integer in one input element, so its width must fit
                // injectively. Wider integers can be assembled inside the program from ABI limbs.
                let widest = widest_injective_int_bits(e.field());
                if *size > widest {
                    unsupported_on_this_field(
                        format_args!(
                            "an int{size} crosses the entry point as one field element, which \
                             carries integers up to int{widest}"
                        ),
                        e.field(),
                    );
                }
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
    use crate::compiler::ssa::{DefaultSSAAnnotator, Terminator};

    fn add_sink(ssa: &mut HLSSA, params: &[Type]) -> FunctionId {
        HLSSABuilder::new(ssa)
            .add_function("sink".to_string(), |fb| {
                let entry = fb.function.get_entry_id();
                let mut e = fb.test_block(entry);
                for ty in params {
                    e.add_parameter(ty.clone());
                }
                e.terminate_return(vec![]);
            })
            .0
    }

    fn snapshot(ssa: &HLSSA) -> String {
        ssa.to_string(&DefaultSSAAnnotator)
    }

    #[test]
    fn reference_arguments_preserve_mixed_order_and_call_location() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();
        let params = vec![Type::field().ref_of(), Type::int(8), Type::int(32).ref_of()];
        let callee = add_sink(&mut ssa, &params);
        let location = SourceLocation::synthetic("reference boundary test");
        let args = HLSSABuilder::new(&mut ssa).modify_function(main, |fb| {
            let entry = fb.function.get_entry_id();
            let mut e = fb.block(entry).with_source_location(location.clone());
            let args: Vec<_> = params
                .iter()
                .map(|ty| e.add_parameter(ty.clone()))
                .collect();
            e.call_unconstrained(callee, args.clone(), 0);
            e.terminate_return(vec![]);
            args
        });

        PrepareEntryPoint::process_unconstrained_calls(&mut ssa);

        let entry = ssa.get_function(main).get_entry();
        let instructions: Vec<_> = entry.get_instructions().collect();
        let [
            OpCode::Load {
                result: first,
                ptr: first_ptr,
            },
            OpCode::Load {
                result: last,
                ptr: last_ptr,
            },
            OpCode::Call {
                function: CallTarget::Static(wrapper),
                args: lowered,
                unconstrained: true,
                ..
            },
        ] = instructions.as_slice()
        else {
            panic!("expected two loads followed by the boundary call: {instructions:?}")
        };
        assert_eq!((*first_ptr, *last_ptr), (args[0], args[2]));
        assert_eq!(*lowered, vec![*first, args[1], *last]);
        assert!(
            entry
                .get_instructions_with_source_locations()
                .all(|(_, source)| source == &location)
        );

        let wrapper = ssa.get_function(*wrapper);
        assert_eq!(
            wrapper.get_param_types(),
            vec![Type::field(), Type::int(8), Type::int(32)]
        );
        let values: Vec<_> = wrapper
            .get_entry()
            .get_parameter_values()
            .copied()
            .collect();
        let instructions: Vec<_> = wrapper.get_entry().get_instructions().collect();
        let [
            OpCode::Alloc {
                result: first_ref,
                value: first_value,
            },
            OpCode::Alloc {
                result: last_ref,
                value: last_value,
            },
            OpCode::Call {
                function: CallTarget::Static(target),
                args: reconstructed,
                unconstrained: false,
                ..
            },
        ] = instructions.as_slice()
        else {
            panic!("expected local refs and an ordinary call: {instructions:?}")
        };
        assert_eq!((*first_value, *last_value), (values[0], values[2]));
        assert_eq!(*target, callee);
        assert_eq!(*reconstructed, vec![*first_ref, values[1], *last_ref]);
        assert_eq!(ssa.get_function(callee).get_param_types(), params);
    }

    #[test]
    fn reference_wrapper_is_shared_across_callers_and_ordinary_calls_stay_intact() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();
        let callee = add_sink(&mut ssa, &[Type::field().ref_of()]);
        let other = ssa.add_function("other".to_string());
        for caller in [main, other] {
            HLSSABuilder::new(&mut ssa).modify_function(caller, |fb| {
                let entry = fb.function.get_entry_id();
                let mut e = fb.test_block(entry);
                let arg = e.add_parameter(Type::field().ref_of());
                e.call_unconstrained(callee, vec![arg], 0);
                e.call(callee, vec![arg], 0);
                e.call_unconstrained(callee, vec![arg], 0);
                e.terminate_return(vec![]);
            });
        }
        let functions_before = ssa.get_function_ids().count();
        PrepareEntryPoint::process_unconstrained_calls(&mut ssa);
        assert_eq!(ssa.get_function_ids().count(), functions_before + 1);

        let mut wrappers = Vec::new();
        for caller in [main, other] {
            let entry = ssa.get_function(caller).get_entry();
            let original_arg = *entry.get_parameter_values().next().unwrap();
            let mut ordinary_calls = 0;
            for instruction in entry.get_instructions() {
                if let OpCode::Call {
                    function: CallTarget::Static(target),
                    args,
                    unconstrained,
                    ..
                } = instruction
                {
                    if *unconstrained {
                        wrappers.push(*target);
                        assert_ne!(*target, callee);
                        assert_ne!(args[0], original_arg);
                    } else {
                        ordinary_calls += 1;
                        assert_eq!(*target, callee);
                        assert_eq!(*args, vec![original_arg]);
                    }
                }
            }
            assert_eq!(ordinary_calls, 1);
        }
        assert_eq!(wrappers.len(), 4);
        assert!(wrappers.iter().all(|wrapper| *wrapper == wrappers[0]));
    }

    #[test]
    fn ordinary_reference_calls_are_unchanged() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();
        let callee = add_sink(&mut ssa, &[Type::field().ref_of()]);
        HLSSABuilder::new(&mut ssa).modify_function(main, |fb| {
            let entry = fb.function.get_entry_id();
            let mut e = fb.test_block(entry);
            let arg = e.add_parameter(Type::field().ref_of());
            e.call(callee, vec![arg], 0);
            e.terminate_return(vec![]);
        });
        let before = snapshot(&ssa);
        PrepareEntryPoint::process_unconstrained_calls(&mut ssa);
        assert_eq!(snapshot(&ssa), before);
    }

    #[test]
    fn reference_free_boundary_calls_need_no_wrapper_or_loads() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();
        let callee = add_sink(&mut ssa, &[Type::field(), Type::int(8).array_of(3)]);
        HLSSABuilder::new(&mut ssa).modify_function(main, |fb| {
            let entry = fb.function.get_entry_id();
            let mut e = fb.test_block(entry);
            let field = e.add_parameter(Type::field());
            let array = e.add_parameter(Type::int(8).array_of(3));
            e.call_unconstrained(callee, vec![field, array], 0);
            e.call_unconstrained(callee, vec![field, array], 0);
            e.terminate_return(vec![]);
        });
        // Void returns isolate argument lowering from the independent result preparation.
        let before = snapshot(&ssa);
        PrepareEntryPoint::process_unconstrained_calls(&mut ssa);
        assert_eq!(snapshot(&ssa), before);
    }

    #[test]
    fn reference_wrapper_preserves_returns_and_witness_preparation() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();
        let callee = HLSSABuilder::new(&mut ssa)
            .add_function("read".to_string(), |fb| {
                fb.function.add_return_type(Type::field());
                let entry = fb.function.get_entry_id();
                let mut e = fb.test_block(entry);
                let arg = e.add_parameter(Type::field().ref_of());
                let value = e.load(arg);
                e.terminate_return(vec![value]);
            })
            .0;
        let result = HLSSABuilder::new(&mut ssa).modify_function(main, |fb| {
            fb.function.add_return_type(Type::field());
            let entry = fb.function.get_entry_id();
            let mut e = fb.test_block(entry);
            let arg = e.add_parameter(Type::field().ref_of());
            let result = e.call_unconstrained(callee, vec![arg], 1)[0];
            e.terminate_return(vec![result]);
            result
        });
        PrepareEntryPoint::process_unconstrained_calls(&mut ssa);
        let entry = ssa.get_function(main).get_entry();
        let instructions: Vec<_> = entry.get_instructions().collect();
        let [
            OpCode::Load { .. },
            OpCode::Call {
                function: CallTarget::Static(wrapper),
                results: raw_results,
                unconstrained: true,
                ..
            },
            OpCode::Call {
                results: prepared,
                args: prepare_args,
                unconstrained: false,
                ..
            },
        ] = instructions.as_slice()
        else {
            panic!("expected load, boundary call, and result preparation: {instructions:?}")
        };
        assert_eq!(*prepared, vec![result]);
        assert_eq!(prepare_args, raw_results);
        assert_ne!(raw_results[0], result);
        assert_eq!(ssa.get_function(*wrapper).get_returns(), &[Type::field()]);
        assert!(matches!(
            entry.get_terminator(),
            Some(Terminator::Return(values)) if *values == vec![result]
        ));
    }

    #[test]
    #[should_panic(expected = "reference wrapper expects 1")]
    fn malformed_reference_call_reports_arity_as_an_ice() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();
        let callee = add_sink(&mut ssa, &[Type::field().ref_of()]);
        HLSSABuilder::new(&mut ssa).modify_function(main, |fb| {
            let entry = fb.function.get_entry_id();
            let mut e = fb.test_block(entry);
            e.call_unconstrained(callee, vec![], 0);
            e.terminate_return(vec![]);
        });
        PrepareEntryPoint::process_unconstrained_calls(&mut ssa);
    }
    #[test]
    fn only_leafless_unconstrained_results_skip_preparation() {
        let unit = Type::tuple_of(vec![]);
        for (return_type, needs_preparation) in [
            (unit.clone(), false),
            (unit.clone().array_of(2), false),
            (Type::field().array_of(0), true),
            (unit.clone().slice_of(), true),
            (Type::field().slice_of(), true),
            (Type::tuple_of(vec![unit, Type::field()]), true),
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
            assert_eq!(ops.len(), if needs_preparation { 2 } else { 1 });
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
            if !needs_preparation {
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
                    assert!(!PrepareEntryPoint::is_leafless_result(
                        &f.get_param_types()[0]
                    ));
                }
            }
        }
    }

    #[test]
    fn absent_abi_return_has_no_slots_or_return_reconstruction() {
        let mut ssa = HLSSA::with_main("main".into());
        let main = ssa.get_unique_entrypoint_id();
        HLSSABuilder::new(&mut ssa).modify_function(main, |b| {
            b.function.add_return_type(Type::tuple_of(vec![]));
            let entry = b.function.get_entry_id();
            let mut e = b
                .block(entry)
                .with_source_location(SourceLocation::synthetic("test"));
            let value = e.mk_tuple(vec![], vec![]);
            e.terminate_return(vec![value]);
        });
        PrepareEntryPoint::wrap_main(&mut ssa, false, Some(false), None);
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

    /// Every witness write of the wrapper, as `(origin, pinned)`, in control-flow order: a loop
    /// body before the block after the loop.
    fn wrapper_writes(ssa: &HLSSA) -> Vec<(String, bool)> {
        let function = ssa.get_unique_entrypoint();
        let mut writes = Vec::new();
        let mut seen = BTreeSet::new();
        let mut pending = vec![function.get_entry_id()];
        while let Some(id) = pending.pop() {
            if !seen.insert(id) {
                continue;
            }
            let block = function.get_block(id);
            writes.extend(block.get_instructions_with_source_locations().filter_map(
                |(instruction, location)| match instruction {
                    OpCode::WriteWitness { pinned, .. } => {
                        Some((location.file.to_string(), *pinned))
                    }
                    _ => None,
                },
            ));
            match block.get_terminator() {
                Some(Terminator::Jmp(target, _)) => pending.push(*target),
                // The taken branch, a loop body, is visited before the fall-through.
                Some(Terminator::JmpIf(_, then_block, else_block)) => {
                    pending.push(*else_block);
                    pending.push(*then_block);
                }
                Some(Terminator::Return(_)) | None => {}
            }
        }
        writes
    }

    #[test]
    fn public_params_are_written_pinned_before_private_unpinned() {
        let mut ssa = HLSSA::with_main("main".into());
        let main = ssa.get_unique_entrypoint_id();
        HLSSABuilder::new(&mut ssa).modify_function(main, |b| {
            let entry = b.function.get_entry_id();
            let mut e = b.test_block(entry);
            for typ in [Type::field(), Type::int(8).array_of(2), Type::field()] {
                e.add_parameter(typ);
            }
            e.terminate_return(vec![]);
        });
        PrepareEntryPoint::wrap_main(&mut ssa, false, Some(false), Some(&[false, true, false]));

        let writes = wrapper_writes(&ssa);
        let origin =
            |index: usize| crate::compiler::located::synthetic_file(entry_param_origin(index));
        assert!(writes[0].1, "the constant one is pinned");
        let by_origin = |index: usize| -> Vec<bool> {
            writes
                .iter()
                .filter(|(file, _)| *file == origin(index))
                .map(|(_, pinned)| *pinned)
                .collect()
        };
        assert_eq!(by_origin(1), vec![true], "the public parameter is pinned");
        assert_eq!(by_origin(0), vec![false], "a private parameter is unpinned");
        assert_eq!(by_origin(2), vec![false]);
        let first_write_of = |index: usize| {
            writes
                .iter()
                .position(|(file, _)| *file == origin(index))
                .unwrap()
        };
        assert!(
            first_write_of(1) < first_write_of(0) && first_write_of(0) < first_write_of(2),
            "public writes come first, then private in signature order: {writes:?}"
        );
    }

    #[test]
    fn absent_abi_return_rejects_live_ssa_results() {
        for typ in [Type::int(8), Type::field().array_of(0)] {
            let failure = std::panic::catch_unwind(|| {
                let mut ssa = HLSSA::with_main("main".into());
                let main = ssa.get_unique_entrypoint_id();
                HLSSABuilder::new(&mut ssa).modify_function(main, |b| {
                    b.function.add_return_type(typ.clone());
                    let entry = b.function.get_entry_id();
                    let mut e = b.test_block(entry);
                    let value = e.add_parameter(typ);
                    e.terminate_return(vec![value]);
                });
                PrepareEntryPoint::wrap_main(&mut ssa, false, Some(false), None);
            })
            .expect_err("the wrapper must reject a nonempty result without an ABI return");
            let message = failure
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| failure.downcast_ref::<&str>().copied())
                .unwrap();
            assert!(message.contains("ABI without a return cannot discard nonempty SSA results"));
        }
    }
}
