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
            FunctionId, SourceLocation, Terminator, ValueId,
            hlssa::{CallTarget, CastTarget, HLSSA, OpCode, SequenceTargetType, TypeExpr},
        },
    },
};

/// Validate every reachable `AssertConstant` and erase all successfully validated markers.
pub(crate) fn validate_and_remove(ssa: &mut HLSSA) -> Result<(), SourceLocation> {
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

    let flow = FlowAnalysis::run(ssa);
    let types = Types::new().run(ssa, &flow);
    let context_depth = assertion_context_depth(ssa);
    let constants = ClickCooper::run_with_context_depth(ssa, &flow, &types, context_depth);
    let compile_time = CompileTimeValues::new(ssa, &types, &constants, context_depth);

    for (fid, bid, value, location) in assertions {
        let valid = constants.contexts_of(fid).iter().all(|context| {
            !constants.is_reachable_in(fid, context, bid)
                || compile_time.is_compile_time_value(fid, Some(context), value)
        });
        if !valid {
            return Err(location);
        }
    }

    drop(compile_time);
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
fn assertion_context_depth(ssa: &HLSSA) -> usize {
    let mut relevant: HashSet<FunctionId> = ssa
        .iter_functions()
        .filter(|(_, function)| {
            function.get_blocks().any(|(_, block)| {
                block
                    .get_instructions()
                    .any(|op| matches!(op, OpCode::AssertConstant { .. }))
            })
        })
        .map(|(fid, _)| *fid)
        .collect();

    loop {
        let mut changed = false;
        for (fid, function) in ssa.iter_functions() {
            if relevant.contains(fid) {
                continue;
            }
            let calls_relevant = function.get_blocks().any(|(_, block)| {
                block.get_instructions().any(|op| {
                    matches!(
                        op,
                        OpCode::Call {
                            function: crate::compiler::ssa::hlssa::CallTarget::Static(callee),
                            ..
                        } if relevant.contains(callee)
                    )
                })
            });
            if calls_relevant {
                relevant.insert(*fid);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    relevant.len().saturating_sub(1).max(1)
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Query {
    function: FunctionId,
    context: Option<Context>,
    value: ValueId,
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
    memo: RefCell<HashMap<Query, bool>>,
    active: RefCell<HashSet<Query>>,
    length_memo: RefCell<HashMap<Query, Option<usize>>>,
    length_active: RefCell<HashSet<Query>>,
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
            memo: RefCell::new(HashMap::default()),
            active: RefCell::new(HashSet::default()),
            length_memo: RefCell::new(HashMap::default()),
            length_active: RefCell::new(HashSet::default()),
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

    fn is_compile_time_value(
        &self,
        fid: FunctionId,
        context: Option<&Context>,
        value: ValueId,
    ) -> bool {
        let query = Query {
            function: fid,
            context: context.cloned(),
            value,
        };
        if let Some(result) = self.memo.borrow().get(&query) {
            return *result;
        }
        if !self.active.borrow_mut().insert(query.clone()) {
            return false;
        }

        let known_by_analysis = match context {
            Some(context) => self.constants.is_constant_in(fid, context, value),
            None => self.constants.is_constant(fid, value),
        };
        let result = known_by_analysis
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
                        self.is_compile_time_value(callee, Some(&callee_context), *first)
                            && returns.iter().skip(1).all(|value| {
                                self.is_compile_time_value(callee, Some(&callee_context), *value)
                                    && self.constants.known_equal_in(
                                        callee,
                                        &callee_context,
                                        *first,
                                        *value,
                                    )
                            })
                    }),
                _ => false,
            };

        self.active.borrow_mut().remove(&query);
        self.memo.borrow_mut().insert(query, result);
        result
    }

    fn static_sequence_length(
        &self,
        fid: FunctionId,
        context: Option<&Context>,
        value: ValueId,
    ) -> Option<usize> {
        let key = Query {
            function: fid,
            context: context.cloned(),
            value,
        };
        if let Some(result) = self.length_memo.borrow().get(&key) {
            return *result;
        }
        if !self.length_active.borrow_mut().insert(key.clone()) {
            return None;
        }

        let types = self.types.get_function(fid);
        let result = if let TypeExpr::Array(_, len) = &types.get_value_type(value).expr {
            Some(*len)
        } else {
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
                Some(OpCode::Select { if_t, if_f, .. }) => {
                    let then_len = self.static_sequence_length(fid, context, *if_t)?;
                    let else_len = self.static_sequence_length(fid, context, *if_f)?;
                    (then_len == else_len).then_some(then_len)
                }
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
                        let mut lengths = returns.iter().map(|value| {
                            self.static_sequence_length(callee, Some(&callee_context), *value)
                        });
                        let first = lengths.next()??;
                        lengths.all(|length| length == Some(first)).then_some(first)
                    }),
                _ => None,
            }
        };

        self.length_active.borrow_mut().remove(&key);
        self.length_memo.borrow_mut().insert(key, result);
        result
    }

    fn global_initializer(&self, offset: u64) -> Option<(FunctionId, ValueId)> {
        usize::try_from(offset)
            .ok()
            .and_then(|offset| self.global_initializers.get(&offset).copied())
    }

    /// Apply `predicate` to every known context of `fid`, falling back to the unconditional view
    /// for synthetic/unreachable functions that have no context.
    fn in_every_context(
        &self,
        fid: FunctionId,
        mut predicate: impl FnMut(Option<&Context>) -> bool,
    ) -> bool {
        let contexts = self.constants.contexts_of(fid);
        if contexts.is_empty() {
            predicate(None)
        } else {
            contexts.iter().all(|context| predicate(Some(context)))
        }
    }

    fn static_call_result(
        &self,
        caller: FunctionId,
        caller_context: Option<&Context>,
        queried_result: ValueId,
        results: &[ValueId],
        callee: FunctionId,
        args: &[ValueId],
    ) -> Option<(FunctionId, Context, usize)> {
        let result_index = results
            .iter()
            .position(|result| *result == queried_result)?;
        let caller_context = caller_context?;
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
}
