//! Validation for Noir's `assert_constant` compiler builtin.
//!
//! Lowering preserves each assertion as an [`OpCode::AssertConstant`] marker through the initial
//! simplification pipeline. At that point tuples have been flattened and ClickCooper has computed
//! both unconditional and per-call-context constant facts. This module validates every marker and
//! erases it before the rest of the compiler runs.

use std::cell::RefCell;

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        analysis::{
            click_cooper::ClickCooper,
            flow_analysis::FlowAnalysis,
            shared::call_string::Context,
            types::{TypeInfo, Types},
            value_definitions::{FunctionValueDefinitions, ValueDefinition},
        },
        ssa::{
            BlockId, FunctionId, SourceLocation, Terminator, ValueId,
            hlssa::{CallTarget, CastTarget, HLSSA, OpCode, SequenceTargetType, TypeExpr},
        },
    },
};

/// Validate every reachable `AssertConstant` and erase all successfully validated markers.
///
/// On failure returns every failing assertion's source location, in program order, so a user fixing
/// several at once sees them all in one compile rather than one per rebuild.
pub(crate) fn validate_and_remove(ssa: &mut HLSSA) -> Result<(), Vec<SourceLocation>> {
    let assertions: Vec<_> = ssa
        .iter_functions()
        .flat_map(|(fid, function)| {
            function.get_blocks().flat_map(move |(bid, block)| {
                block
                    .get_instructions_with_source_locations()
                    .filter_map(move |(op, location)| match op {
                        OpCode::AssertConstant { value } => {
                            Some((*fid, *bid, *value, location.clone()))
                        }
                        _ => None,
                    })
            })
        })
        .collect();

    if assertions.is_empty() {
        return Ok(());
    }

    // Scoped so `compile_time` and `constants` release their borrow of `ssa` before the erasure
    // walk below takes it mutably.
    {
        let flow = FlowAnalysis::run(ssa);
        let types = Types::new().run(ssa, &flow);
        let context_depth = assertion_context_depth(ssa);
        let constants = ClickCooper::run_for_assert_constant(ssa, &flow, &types, context_depth);
        let compile_time = CompileTimeValues::new(ssa, &types, &constants, context_depth);

        let failures: Vec<SourceLocation> = assertions
            .iter()
            .filter(|(fid, bid, value, _)| !compile_time.assertion_holds(*fid, *bid, *value))
            .map(|(_, _, _, location)| location.clone())
            .collect();
        if !failures.is_empty() {
            return Err(failures);
        }
    }

    for (_, function) in ssa.iter_functions_mut() {
        for (_, block) in function.get_blocks_mut() {
            let instructions = block.take_instructions();
            block.put_instructions(
                instructions
                    .into_iter()
                    .filter(|op| !matches!(&**op, OpCode::AssertConstant { .. }))
                    .collect(),
            );
        }
    }
    Ok(())
}

/// Keep every call site on an acyclic path to an assertion in the context coordinate.
///
/// Noir forces non-recursive functions containing static assertions through its inliner before
/// validation. Mavros validates without cloning functions, so a 1-CFA truncation would instead
/// merge distinct outer call sites at a shared inner call. The number of functions that can reach
/// an assertion bounds every simple path through that relevant call-graph slice; recursion still
/// folds to a finite context, as required for termination.
///
/// The depth is deliberately uncapped: truncating it would reintroduce exactly the call-site
/// merging this function exists to prevent.
fn assertion_context_depth(ssa: &HLSSA) -> usize {
    // Reverse static call edges, built once. The transitive-caller closure below would otherwise
    // rescan every function body on each round, making the fixpoint quadratic in the program size.
    let mut callers: HashMap<FunctionId, Vec<FunctionId>> = HashMap::default();
    let mut relevant: HashSet<FunctionId> = HashSet::default();

    for (fid, function) in ssa.iter_functions() {
        for (_, block) in function.get_blocks() {
            for op in block.get_instructions() {
                match op {
                    OpCode::AssertConstant { .. } => {
                        relevant.insert(*fid);
                    }
                    OpCode::Call {
                        function: CallTarget::Static(callee),
                        ..
                    } => callers.entry(*callee).or_default().push(*fid),
                    _ => {}
                }
            }
        }
    }

    // Transitive callers of any asserting function, by worklist over the reverse edges.
    let mut worklist: Vec<FunctionId> = relevant.iter().copied().collect();
    while let Some(callee) = worklist.pop() {
        for caller in callers.get(&callee).into_iter().flatten() {
            if relevant.insert(*caller) {
                worklist.push(*caller);
            }
        }
    }

    relevant.len().saturating_sub(1).max(1)
}

/// An interned [`Context`], or the unconditional (context-free) view.
///
/// A `Context` is a call string of up to [`MAX_ASSERTION_CONTEXT_DEPTH`] sites, so using it
/// directly in a memo key would clone a `Vec` on every lookup along the hottest recursion path.
/// Interning makes the key `Copy` and the memo entries flat.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ContextId {
    /// The unconditional view, used for functions the specializer never reached.
    Unconditional,
    /// An index into [`ContextInterner::contexts`].
    Interned(usize),
}

/// Assigns a stable [`ContextId`] to each distinct [`Context`].
///
/// Contexts are append-only, so an id is an index into `contexts` and stays valid for the lifetime
/// of the interner.
#[derive(Default)]
struct ContextInterner {
    contexts: Vec<Context>,
    ids: HashMap<Context, usize>,
}

impl ContextInterner {
    fn intern(&mut self, context: Context) -> ContextId {
        if let Some(id) = self.ids.get(&context) {
            return ContextId::Interned(*id);
        }
        let id = self.contexts.len();
        self.contexts.push(context.clone());
        self.ids.insert(context, id);
        ContextId::Interned(id)
    }

    fn resolve(&self, id: ContextId) -> Option<&Context> {
        match id {
            ContextId::Unconditional => None,
            ContextId::Interned(index) => Some(&self.contexts[index]),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Query {
    function: FunctionId,
    context: ContextId,
    value: ValueId,
}

/// A memo table for a recursive per-[`Query`] fact, with cycle breaking.
///
/// Both queries below walk value definitions, which can be cyclic, so each needs the same
/// scaffolding: return a memoized answer, otherwise mark the query in-progress and recurse,
/// yielding a fixed answer if the recursion re-enters the same query. `T::default()` is that
/// cycle-breaking answer, chosen so it is the conservative one for each fact — `false` for "is a
/// compile-time value", `None` for "has a static length".
struct QueryCache<T> {
    done: RefCell<HashMap<Query, T>>,
    in_progress: RefCell<HashSet<Query>>,
}

// Hand-written rather than derived: `#[derive(Default)]` would add a spurious `T: Default` bound on
// the struct itself, which the tables do not need.
impl<T> Default for QueryCache<T> {
    fn default() -> Self {
        Self {
            done: RefCell::new(HashMap::default()),
            in_progress: RefCell::new(HashSet::default()),
        }
    }
}

impl<T: Copy + Default> QueryCache<T> {
    /// The cached value of `query`, else `compute()` memoized under it.
    ///
    /// Returns `T::default()` without calling `compute` when `query` is already being computed
    /// further up the stack.
    fn get_or_compute(&self, query: Query, compute: impl FnOnce() -> T) -> T {
        if let Some(result) = self.done.borrow().get(&query) {
            return *result;
        }
        if !self.in_progress.borrow_mut().insert(query) {
            return T::default();
        }
        let result = compute();
        self.in_progress.borrow_mut().remove(&query);
        self.done.borrow_mut().insert(query, result);
        result
    }
}

/// A non-materializing compile-time-value query layered over ClickCooper.
///
/// ClickCooper deliberately caps aggregate materialization to bound analysis memory. The
/// structural cases below preserve the language-level meaning of `assert_constant` beyond that
/// implementation limit. They also track sequence shape separately from element constness, so the
/// length of an array-to-vector cast remains compile-time known even when its elements are not.
struct CompileTimeValues<'a> {
    ssa: &'a HLSSA,
    types: &'a TypeInfo,
    constants: &'a ClickCooper,
    context_depth: usize,
    definitions: HashMap<FunctionId, FunctionValueDefinitions>,
    global_initializers: HashMap<usize, (FunctionId, ValueId)>,
    contexts: RefCell<ContextInterner>,
    /// Memoizes [`Self::is_compile_time_value`]; a cycle answers `false`.
    is_constant: QueryCache<bool>,
    /// Memoizes [`Self::static_sequence_length`]; a cycle answers `None`.
    lengths: QueryCache<Option<usize>>,
}

impl<'a> CompileTimeValues<'a> {
    fn new(
        ssa: &'a HLSSA,
        types: &'a TypeInfo,
        constants: &'a ClickCooper,
        context_depth: usize,
    ) -> Self {
        let definitions = ssa
            .iter_functions()
            .map(|(fid, function)| (*fid, FunctionValueDefinitions::from_function(function)))
            .collect();
        let global_initializers = Self::index_global_initializers(ssa);
        Self {
            ssa,
            types,
            constants,
            context_depth,
            definitions,
            global_initializers,
            contexts: RefCell::new(ContextInterner::default()),
            is_constant: QueryCache::default(),
            lengths: QueryCache::default(),
        }
    }

    /// Whether the `AssertConstant` on `value` at `fid`/`bid` holds in every context reaching it.
    fn assertion_holds(&self, fid: FunctionId, bid: BlockId, value: ValueId) -> bool {
        self.in_every_context(fid, |context| {
            !self.is_reachable(fid, context, bid) || self.is_compile_time_value(fid, context, value)
        })
    }

    /// Intern `context` for use as a memo key.
    fn intern(&self, context: Context) -> ContextId {
        self.contexts.borrow_mut().intern(context)
    }

    /// The [`Context`] behind `id`, cloned out of the interner.
    ///
    /// Cloned rather than borrowed because the interner sits behind a [`RefCell`] that the
    /// recursive callers below re-enter to intern callee contexts.
    fn context_of(&self, id: ContextId) -> Option<Context> {
        self.contexts.borrow().resolve(id).cloned()
    }

    /// Whether `bid` is reachable in `fid` under `context`.
    ///
    /// Under [`ContextId::Unconditional`] there is no specialized reachability to consult, so every
    /// block counts as reachable — the conservative answer, since it only ever forces more
    /// assertions to be proven.
    fn is_reachable(&self, fid: FunctionId, context: ContextId, bid: BlockId) -> bool {
        match self.context_of(context) {
            Some(context) => self.constants.is_reachable_in(fid, &context, bid),
            None => true,
        }
    }

    fn index_global_initializers(ssa: &HLSSA) -> HashMap<usize, (FunctionId, ValueId)> {
        let mut initializers = HashMap::default();
        for (fid, function) in ssa.iter_functions() {
            for (_, block) in function.get_blocks() {
                for op in block.get_instructions() {
                    let OpCode::InitGlobal { global, value } = op else {
                        continue;
                    };
                    assert_eq!(
                        Some(*fid),
                        ssa.get_globals_init_fn(),
                        "ICE: InitGlobal outside the dedicated globals_init function"
                    );
                    assert!(
                        initializers.insert(*global, (*fid, *value)).is_none(),
                        "ICE: global slot {global} initialized more than once"
                    );
                }
            }
        }
        initializers
    }

    fn definition(&self, fid: FunctionId, value: ValueId) -> Option<&OpCode> {
        match self.definitions.get(&fid)?.get_definition(value)? {
            ValueDefinition::Instruction(_, _, op) => Some(op),
            ValueDefinition::Param(..) => None,
        }
    }

    fn is_compile_time_value(&self, fid: FunctionId, context: ContextId, value: ValueId) -> bool {
        let query = Query {
            function: fid,
            context,
            value,
        };
        self.is_constant.get_or_compute(query, || {
            let known_by_analysis = match self.context_of(context) {
                Some(context) => self.constants.is_constant_in(fid, &context, value),
                None => self.constants.is_constant(fid, value),
            };
            known_by_analysis
                || match self.definition(fid, value) {
                    Some(OpCode::MkSeq { elems, .. }) => elems
                        .iter()
                        .all(|value| self.is_compile_time_value(fid, context, *value)),
                    Some(OpCode::MkRepeated { element, .. }) => {
                        self.is_compile_time_value(fid, context, *element)
                    }
                    Some(OpCode::MkSeqOfBlob { blob, .. }) => {
                        self.is_compile_time_value(fid, context, *blob)
                    }
                    Some(OpCode::Cast {
                        value,
                        target: CastTarget::ArrayToSlice | CastTarget::Nop,
                        ..
                    }) => self.is_compile_time_value(fid, context, *value),
                    Some(OpCode::ArraySet {
                        array,
                        index,
                        value,
                        ..
                    }) => {
                        self.is_compile_time_value(fid, context, *array)
                            && self.is_compile_time_value(fid, context, *index)
                            && self.is_compile_time_value(fid, context, *value)
                    }
                    Some(OpCode::SlicePush { slice, values, .. }) => {
                        self.is_compile_time_value(fid, context, *slice)
                            && values
                                .iter()
                                .all(|value| self.is_compile_time_value(fid, context, *value))
                    }
                    Some(OpCode::SliceLen { slice, .. }) => {
                        self.static_sequence_length(fid, context, *slice).is_some()
                    }
                    Some(OpCode::ReadGlobal { offset, .. }) => self
                        .global_initializer(*offset)
                        .is_some_and(|(init_fid, initializer)| {
                            self.in_every_context(init_fid, |context| {
                                self.is_compile_time_value(init_fid, context, initializer)
                            })
                        }),
                    Some(OpCode::Call {
                        results,
                        function: CallTarget::Static(callee),
                        args,
                        unconstrained: false,
                    }) => self
                        .static_call_result(fid, context, value, results, *callee, args)
                        .is_some_and(|(callee, callee_context, result_index)| {
                            let returns =
                                self.reachable_return_values(callee, &callee_context, result_index);
                            let Some(first) = returns.first() else {
                                return false;
                            };
                            let callee_id = self.intern(callee_context.clone());
                            self.is_compile_time_value(callee, callee_id, *first)
                                && returns.iter().skip(1).all(|value| {
                                    self.is_compile_time_value(callee, callee_id, *value)
                                        && self.constants.known_equal_in(
                                            callee,
                                            &callee_context,
                                            *first,
                                            *value,
                                        )
                                })
                        }),
                    _ => false,
                }
        })
    }

    fn static_sequence_length(
        &self,
        fid: FunctionId,
        context: ContextId,
        value: ValueId,
    ) -> Option<usize> {
        let query = Query {
            function: fid,
            context,
            value,
        };
        self.lengths.get_or_compute(query, || {
            let types = self.types.get_function(fid);
            if let TypeExpr::Array(_, len) = &types.get_value_type(value).expr {
                return Some(*len);
            }
            match self.definition(fid, value) {
                Some(OpCode::Cast {
                    value,
                    target: CastTarget::ArrayToSlice | CastTarget::Nop,
                    ..
                }) => self.static_sequence_length(fid, context, *value),
                Some(OpCode::MkSeq {
                    elems,
                    seq_type: SequenceTargetType::Slice,
                    ..
                }) => Some(elems.len()),
                Some(OpCode::MkRepeated {
                    seq_type: SequenceTargetType::Slice,
                    count,
                    ..
                }) => Some(*count),
                Some(OpCode::SlicePush { slice, values, .. }) => self
                    .static_sequence_length(fid, context, *slice)
                    .and_then(|len| len.checked_add(values.len())),
                // No `Select` arm: its result type is the *arithmetic* join of its operands
                // (`Type::get_arithmetic_result_type`), which is scalar-only, so a `Select` never
                // carries a sequence whose length could be asked for here.
                Some(OpCode::ReadGlobal { offset, .. }) => self
                    .global_initializer(*offset)
                    .and_then(|(init_fid, initializer)| {
                        let mut length = None;
                        self.in_every_context(init_fid, |context| {
                            let next = self.static_sequence_length(init_fid, context, initializer);
                            match (length, next) {
                                (None, Some(next)) => {
                                    length = Some(next);
                                    true
                                }
                                (Some(previous), Some(next)) => previous == next,
                                (_, None) => false,
                            }
                        })
                        .then_some(length)
                        .flatten()
                    }),
                Some(OpCode::Call {
                    results,
                    function: CallTarget::Static(callee),
                    args,
                    unconstrained: false,
                }) => self
                    .static_call_result(fid, context, value, results, *callee, args)
                    .and_then(|(callee, callee_context, result_index)| {
                        let returns =
                            self.reachable_return_values(callee, &callee_context, result_index);
                        let callee_id = self.intern(callee_context);
                        let mut lengths = returns
                            .iter()
                            .map(|value| self.static_sequence_length(callee, callee_id, *value));
                        let first = lengths.next()??;
                        lengths.all(|length| length == Some(first)).then_some(first)
                    }),
                _ => None,
            }
        })
    }

    fn global_initializer(&self, offset: u64) -> Option<(FunctionId, ValueId)> {
        usize::try_from(offset)
            .ok()
            .and_then(|offset| self.global_initializers.get(&offset).copied())
    }

    /// Apply `predicate` to every known context of `fid`, falling back to the unconditional view
    /// for synthetic/unreachable functions that have no context.
    ///
    /// The fallback is load-bearing, not a convenience: a bare `all` over an empty context set
    /// returns `true`, which would accept an assertion the analysis never examined. Every
    /// context-quantified query must route through here so that case is decided rather than
    /// vacuously passed.
    fn in_every_context(
        &self,
        fid: FunctionId,
        mut predicate: impl FnMut(ContextId) -> bool,
    ) -> bool {
        let contexts = self.constants.contexts_of(fid);
        if contexts.is_empty() {
            predicate(ContextId::Unconditional)
        } else {
            contexts
                .into_iter()
                .all(|context| predicate(self.intern(context)))
        }
    }

    /// Resolve a static call result to the callee, its context, and the result's position.
    ///
    /// Returns `None` under [`ContextId::Unconditional`]: extending a call string requires a caller
    /// context to extend, and the context-free view has none. That treats every call result as
    /// non-constant, which is deliberately fail-closed — it only ever rejects an assertion the
    /// context-sensitive view would have had to prove anyway.
    fn static_call_result(
        &self,
        caller: FunctionId,
        caller_context: ContextId,
        queried_result: ValueId,
        results: &[ValueId],
        callee: FunctionId,
        args: &[ValueId],
    ) -> Option<(FunctionId, Context, usize)> {
        let result_index = results
            .iter()
            .position(|result| *result == queried_result)?;
        let caller_context = self.context_of(caller_context)?;
        let site = results.first().or_else(|| args.first()).copied()?;
        Some((
            callee,
            caller_context.push((caller, site), self.context_depth),
            result_index,
        ))
    }

    fn reachable_return_values(
        &self,
        fid: FunctionId,
        context: &Context,
        result_index: usize,
    ) -> Vec<ValueId> {
        self.ssa
            .get_function(fid)
            .get_blocks()
            .filter(|(bid, _)| self.constants.is_reachable_in(fid, context, **bid))
            .filter_map(|(_, block)| match block.get_terminator() {
                Some(Terminator::Return(values)) => values.get(result_index).copied(),
                _ => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::validate_and_remove;
    use crate::compiler::{
        Field,
        ssa::{
            FunctionId, SourceLocation, Terminator, ValueId,
            hlssa::{
                BinaryArithOpKind, CallTarget, CastTarget, Constant, HLSSA, OpCode,
                SequenceTargetType, Type,
            },
        },
    };

    fn assert_constant(value: ValueId) -> crate::compiler::ssa::Located<OpCode> {
        OpCode::AssertConstant { value }.locate(SourceLocation::test())
    }

    fn add_asserting_helper(ssa: &mut HLSSA) -> FunctionId {
        let helper = ssa.add_function("asserting_helper".to_string());
        let parameter = ssa.fresh_value();
        let entry = ssa.get_function_mut(helper).get_entry_mut();
        entry.push_parameter(parameter, Type::field());
        entry.push_instruction(assert_constant(parameter));
        entry.set_terminator(Terminator::Return(vec![]));
        helper
    }

    fn call(ssa: &mut HLSSA, callee: FunctionId, argument: ValueId, unconstrained: bool) {
        let main = ssa.get_unique_entrypoint_id();
        ssa.get_function_mut(main).get_entry_mut().push_instruction(
            OpCode::Call {
                results: vec![],
                function: CallTarget::Static(callee),
                args: vec![argument],
                unconstrained,
            }
            .locate(SourceLocation::test()),
        );
    }

    #[test]
    fn accepts_constants_in_every_call_context_and_removes_markers() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let helper = add_asserting_helper(&mut ssa);
        let five = ssa.add_const(Constant::Field(Field::from(5u64)));
        let six = ssa.add_const(Constant::Field(Field::from(6u64)));
        call(&mut ssa, helper, five, false);
        call(&mut ssa, helper, six, false);
        ssa.get_unique_entrypoint_mut()
            .get_entry_mut()
            .set_terminator(Terminator::Return(vec![]));

        validate_and_remove(&mut ssa).unwrap();
        assert!(ssa.iter_functions().all(|(_, function)| {
            function.get_blocks().all(|(_, block)| {
                block
                    .get_instructions()
                    .all(|op| !matches!(op, OpCode::AssertConstant { .. }))
            })
        }));
    }

    #[test]
    fn rejects_when_any_call_context_is_dynamic() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let helper = add_asserting_helper(&mut ssa);
        let dynamic = ssa.fresh_value();
        let five = ssa.add_const(Constant::Field(Field::from(5u64)));
        ssa.get_unique_entrypoint_mut()
            .get_entry_mut()
            .push_parameter(dynamic, Type::field());
        call(&mut ssa, helper, five, false);
        call(&mut ssa, helper, dynamic, false);
        ssa.get_unique_entrypoint_mut()
            .get_entry_mut()
            .set_terminator(Terminator::Return(vec![]));

        assert!(validate_and_remove(&mut ssa).is_err());
    }

    #[test]
    fn keeps_distinct_outer_contexts_through_a_shared_inner_call_site() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let leaf = add_asserting_helper(&mut ssa);
        let middle = ssa.add_function("middle".to_string());
        let parameter = ssa.fresh_value();
        {
            let entry = ssa.get_function_mut(middle).get_entry_mut();
            entry.push_parameter(parameter, Type::field());
            entry.push_instruction(
                OpCode::Call {
                    results: vec![],
                    function: CallTarget::Static(leaf),
                    args: vec![parameter],
                    unconstrained: false,
                }
                .locate(SourceLocation::test()),
            );
            entry.set_terminator(Terminator::Return(vec![]));
        }

        let five = ssa.add_const(Constant::Field(Field::from(5u64)));
        let six = ssa.add_const(Constant::Field(Field::from(6u64)));
        call(&mut ssa, middle, five, false);
        call(&mut ssa, middle, six, false);
        ssa.get_unique_entrypoint_mut()
            .get_entry_mut()
            .set_terminator(Terminator::Return(vec![]));

        validate_and_remove(&mut ssa).unwrap();
    }

    #[test]
    fn accepts_pure_call_result_with_constant_arguments() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();
        let plus_one = ssa.add_function("plus_one".to_string());
        let parameter = ssa.fresh_value();
        let sum = ssa.fresh_value();
        let result = ssa.fresh_value();
        let one = ssa.add_const(Constant::Field(Field::from(1u64)));
        let eleven = ssa.add_const(Constant::Field(Field::from(11u64)));
        {
            let function = ssa.get_function_mut(plus_one);
            function.add_return_type(Type::field());
            let entry = function.get_entry_mut();
            entry.push_parameter(parameter, Type::field());
            entry.push_instruction(
                OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::Add,
                    result: sum,
                    lhs: parameter,
                    rhs: one,
                }
                .locate(SourceLocation::test()),
            );
            entry.set_terminator(Terminator::Return(vec![sum]));
        }
        {
            let entry = ssa.get_function_mut(main).get_entry_mut();
            entry.push_instruction(
                OpCode::Call {
                    results: vec![result],
                    function: CallTarget::Static(plus_one),
                    args: vec![eleven],
                    unconstrained: false,
                }
                .locate(SourceLocation::test()),
            );
            entry.push_instruction(assert_constant(result));
            entry.set_terminator(Terminator::Return(vec![]));
        }

        validate_and_remove(&mut ssa).unwrap();
    }

    #[test]
    fn rejects_call_result_selected_by_dynamic_control_flow() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();
        let choose = ssa.add_function("choose".to_string());
        let dynamic = ssa.fresh_value();
        let condition = ssa.fresh_value();
        let result = ssa.fresh_value();
        let one = ssa.add_const(Constant::Field(Field::from(1u64)));
        let two = ssa.add_const(Constant::Field(Field::from(2u64)));
        {
            let function = ssa.get_function_mut(choose);
            function.add_return_type(Type::field());
            let then_block = function.add_block();
            let else_block = function.add_block();
            let entry = function.get_entry_mut();
            entry.push_parameter(condition, Type::bool());
            entry.set_terminator(Terminator::JmpIf(condition, then_block, else_block));
            function
                .get_block_mut(then_block)
                .set_terminator(Terminator::Return(vec![one]));
            function
                .get_block_mut(else_block)
                .set_terminator(Terminator::Return(vec![two]));
        }
        {
            let entry = ssa.get_function_mut(main).get_entry_mut();
            entry.push_parameter(dynamic, Type::bool());
            entry.push_instruction(
                OpCode::Call {
                    results: vec![result],
                    function: CallTarget::Static(choose),
                    args: vec![dynamic],
                    unconstrained: false,
                }
                .locate(SourceLocation::test()),
            );
            entry.push_instruction(assert_constant(result));
            entry.set_terminator(Terminator::Return(vec![]));
        }

        assert!(validate_and_remove(&mut ssa).is_err());
    }

    #[test]
    fn rejects_dynamic_unconstrained_call_context() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let helper = add_asserting_helper(&mut ssa);
        let dynamic = ssa.fresh_value();
        ssa.get_unique_entrypoint_mut()
            .get_entry_mut()
            .push_parameter(dynamic, Type::field());
        call(&mut ssa, helper, dynamic, true);
        ssa.get_unique_entrypoint_mut()
            .get_entry_mut()
            .set_terminator(Terminator::Return(vec![]));

        assert!(validate_and_remove(&mut ssa).is_err());
    }

    #[test]
    fn distinguishes_constant_and_dynamic_aggregates() {
        for dynamic_elements in [false, true] {
            let mut ssa = HLSSA::with_main("main".to_string());
            let dynamic = ssa.fresh_value();
            let constant = ssa.add_const(Constant::Field(Field::from(5u64)));
            let array = ssa.fresh_value();

            let entry = ssa.get_unique_entrypoint_mut().get_entry_mut();
            entry.push_parameter(dynamic, Type::field());
            entry.push_instruction(
                OpCode::MkSeq {
                    result: array,
                    elems: vec![if dynamic_elements { dynamic } else { constant }],
                    seq_type: SequenceTargetType::Array(1),
                    elem_type: Type::field(),
                }
                .locate(SourceLocation::test()),
            );
            entry.push_instruction(assert_constant(array));
            entry.set_terminator(Terminator::Return(vec![]));

            assert_eq!(validate_and_remove(&mut ssa).is_ok(), !dynamic_elements);
        }
    }

    #[test]
    fn accepts_large_constant_aggregate_without_materializing_it() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let constant = ssa.add_const(Constant::Field(Field::from(5u64)));
        let array = ssa.fresh_value();
        let entry = ssa.get_unique_entrypoint_mut().get_entry_mut();
        entry.push_instruction(
            OpCode::MkRepeated {
                result: array,
                element: constant,
                seq_type: SequenceTargetType::Array(5_000),
                count: 5_000,
                elem_type: Type::field(),
            }
            .locate(SourceLocation::test()),
        );
        entry.push_instruction(assert_constant(array));
        entry.set_terminator(Terminator::Return(vec![]));

        validate_and_remove(&mut ssa).unwrap();
    }

    #[test]
    fn accepts_large_constant_aggregate_read_from_global() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();
        let globals_init = ssa.add_function("globals_init".to_string());
        ssa.set_globals_init_fn(globals_init);
        ssa.set_global_types(vec![Type::field().array_of(5_000)]);

        let constant = ssa.add_const(Constant::Field(Field::from(5u64)));
        let initializer = ssa.fresh_value();
        let read = ssa.fresh_value();
        {
            let entry = ssa.get_function_mut(globals_init).get_entry_mut();
            entry.push_instruction(
                OpCode::MkRepeated {
                    result: initializer,
                    element: constant,
                    seq_type: SequenceTargetType::Array(5_000),
                    count: 5_000,
                    elem_type: Type::field(),
                }
                .locate(SourceLocation::test()),
            );
            entry.push_instruction(
                OpCode::InitGlobal {
                    global: 0,
                    value: initializer,
                }
                .locate(SourceLocation::test()),
            );
            entry.set_terminator(Terminator::Return(vec![]));
        }
        {
            let entry = ssa.get_function_mut(main).get_entry_mut();
            entry.push_instruction(
                OpCode::ReadGlobal {
                    result: read,
                    offset: 0,
                    result_type: Type::field().array_of(5_000),
                }
                .locate(SourceLocation::test()),
            );
            entry.push_instruction(assert_constant(read));
            entry.set_terminator(Terminator::Return(vec![]));
        }

        validate_and_remove(&mut ssa).unwrap();
    }

    #[test]
    fn array_to_slice_length_is_constant_even_with_dynamic_elements() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let dynamic = ssa.fresh_value();
        let array = ssa.fresh_value();
        let slice = ssa.fresh_value();
        let len = ssa.fresh_value();
        let entry = ssa.get_unique_entrypoint_mut().get_entry_mut();
        entry.push_parameter(dynamic, Type::field());
        entry.push_instruction(
            OpCode::MkRepeated {
                result: array,
                element: dynamic,
                seq_type: SequenceTargetType::Array(17),
                count: 17,
                elem_type: Type::field(),
            }
            .locate(SourceLocation::test()),
        );
        entry.push_instruction(
            OpCode::Cast {
                result: slice,
                value: array,
                target: CastTarget::ArrayToSlice,
            }
            .locate(SourceLocation::test()),
        );
        entry.push_instruction(
            OpCode::SliceLen { result: len, slice }.locate(SourceLocation::test()),
        );
        entry.push_instruction(assert_constant(len));
        entry.set_terminator(Terminator::Return(vec![]));

        validate_and_remove(&mut ssa).unwrap();
    }

    /// A function no call reaches gets no specialized context. Deciding it by `all` over that empty
    /// context set would vacuously accept and erase the marker unexamined.
    #[test]
    fn rejects_dynamic_assertion_in_a_function_with_no_contexts() {
        let mut ssa = HLSSA::with_main("main".to_string());
        // `add_asserting_helper` asserts its parameter, which is dynamic in the context-free view.
        // Deliberately never called, so `specialize` never reaches it.
        add_asserting_helper(&mut ssa);
        ssa.get_unique_entrypoint_mut()
            .get_entry_mut()
            .set_terminator(Terminator::Return(vec![]));

        assert!(validate_and_remove(&mut ssa).is_err());
    }

    /// The counterpart: an uncalled function whose assertion is constant regardless of context is
    /// still accepted, so the fix above rejects only what it must.
    #[test]
    fn accepts_constant_assertion_in_a_function_with_no_contexts() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let helper = ssa.add_function("uncalled".to_string());
        let five = ssa.add_const(Constant::Field(Field::from(5u64)));
        {
            let entry = ssa.get_function_mut(helper).get_entry_mut();
            entry.push_instruction(assert_constant(five));
            entry.set_terminator(Terminator::Return(vec![]));
        }
        ssa.get_unique_entrypoint_mut()
            .get_entry_mut()
            .set_terminator(Terminator::Return(vec![]));

        validate_and_remove(&mut ssa).unwrap();
    }
}
