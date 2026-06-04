//! Concrete length inference for slice values that the current LLSSA backend
//! must unroll as fixed-size array storage.

use std::collections::HashMap;

use crate::compiler::{
    analysis::types::{FunctionTypeInfo, TypeInfo},
    ssa::{
        FunctionId, Terminator, ValueId,
        hlssa::{CallTarget, CastTarget, HLFunction, HLSSA, OpCode, SequenceTargetType, TypeExpr},
    },
};

#[derive(Clone, Debug, PartialEq, Eq)]
enum LengthFact {
    Known(usize),
    Unknown(String),
}

type LengthFacts = HashMap<ValueId, LengthFact>;

#[derive(Clone, Default)]
pub struct FunctionSliceLengths {
    facts: HashMap<ValueId, LengthFact>,
}

impl FunctionSliceLengths {
    pub fn get(&self, value: &ValueId) -> Option<&usize> {
        match self.facts.get(value) {
            Some(LengthFact::Known(len)) => Some(len),
            _ => None,
        }
    }

    pub fn require(&self, value: ValueId, context: &str) -> usize {
        match self.facts.get(&value) {
            Some(LengthFact::Known(len)) => *len,
            Some(LengthFact::Unknown(reason)) => {
                panic!(
                    "{context}: slice value {value:?} does not have a single concrete length. The current LLSSA backend requires every slice value to have one statically inferred concrete length: {reason}"
                )
            }
            None => {
                panic!(
                    "{context}: concrete length for slice value {value:?} is not available. The current LLSSA backend requires every slice value to have one statically inferred concrete length"
                )
            }
        }
    }
}

pub struct SliceLengthAnalysis {
    values: HashMap<FunctionId, FunctionSliceLengths>,
}

impl SliceLengthAnalysis {
    pub fn run(ssa: &HLSSA, type_info: &TypeInfo) -> Self {
        let mut values: HashMap<FunctionId, LengthFacts> = HashMap::new();
        let mut returns: HashMap<FunctionId, HashMap<usize, LengthFact>> = HashMap::new();
        let function_ids: Vec<_> = ssa.get_function_ids().collect();
        let mut changed = true;

        while changed {
            changed = false;

            for function_id in &function_ids {
                if !type_info.has_function(*function_id) {
                    continue;
                }

                let function = ssa.get_function(*function_id);
                let fn_type_info = type_info.get_function(*function_id);
                changed |= infer_function(
                    *function_id,
                    function,
                    fn_type_info,
                    ssa,
                    &mut values,
                    &mut returns,
                );
            }
        }

        let values = values
            .into_iter()
            .map(|(function_id, facts)| (function_id, FunctionSliceLengths { facts }))
            .collect();

        Self { values }
    }

    pub fn function_values(&self, function_id: FunctionId) -> Option<&FunctionSliceLengths> {
        self.values.get(&function_id)
    }
}

fn record_slice_length(facts: &mut LengthFacts, value: ValueId, len: usize, reason: &str) -> bool {
    match facts.get(&value) {
        Some(LengthFact::Known(old)) if *old == len => false,
        Some(LengthFact::Known(old)) => {
            facts.insert(
                value,
                LengthFact::Unknown(format!("{reason}; observed lengths {old} and {len}")),
            );
            true
        }
        Some(LengthFact::Unknown(_)) => false,
        None => {
            facts.insert(value, LengthFact::Known(len));
            true
        }
    }
}

fn record_unknown_slice_length(facts: &mut LengthFacts, value: ValueId, reason: String) -> bool {
    match facts.get(&value) {
        Some(LengthFact::Unknown(_)) => false,
        Some(_) => {
            facts.insert(value, LengthFact::Unknown(reason));
            true
        }
        None => {
            facts.insert(value, LengthFact::Unknown(reason));
            true
        }
    }
}

fn copy_slice_length_fact(
    source_facts: &LengthFacts,
    target_facts: &mut LengthFacts,
    source: ValueId,
    target: ValueId,
    reason: &str,
) -> bool {
    match source_facts.get(&source) {
        Some(LengthFact::Known(len)) => record_slice_length(target_facts, target, *len, reason),
        Some(LengthFact::Unknown(source_reason)) => {
            record_unknown_slice_length(target_facts, target, source_reason.clone())
        }
        None => false,
    }
}

fn record_return_length(
    returns: &mut HashMap<FunctionId, HashMap<usize, LengthFact>>,
    function_id: FunctionId,
    index: usize,
    len: usize,
    reason: &str,
) -> bool {
    let function_returns = returns.entry(function_id).or_default();
    match function_returns.get(&index) {
        Some(LengthFact::Known(old)) if *old == len => false,
        Some(LengthFact::Known(old)) => {
            function_returns.insert(
                index,
                LengthFact::Unknown(format!("{reason}; observed lengths {old} and {len}")),
            );
            true
        }
        Some(LengthFact::Unknown(_)) => false,
        None => {
            function_returns.insert(index, LengthFact::Known(len));
            true
        }
    }
}

fn record_unknown_return_length(
    returns: &mut HashMap<FunctionId, HashMap<usize, LengthFact>>,
    function_id: FunctionId,
    index: usize,
    reason: String,
) -> bool {
    let function_returns = returns.entry(function_id).or_default();
    match function_returns.get(&index) {
        Some(LengthFact::Unknown(_)) => false,
        Some(_) => {
            function_returns.insert(index, LengthFact::Unknown(reason));
            true
        }
        None => {
            function_returns.insert(index, LengthFact::Unknown(reason));
            true
        }
    }
}

fn return_length_fact(
    returns: &HashMap<FunctionId, HashMap<usize, LengthFact>>,
    function_id: FunctionId,
    index: usize,
) -> Option<&LengthFact> {
    returns.get(&function_id)?.get(&index)
}

fn known_value_length(
    facts: &LengthFacts,
    type_info: &FunctionTypeInfo,
    value: ValueId,
) -> Option<usize> {
    match &type_info.get_value_type(value).expr {
        TypeExpr::Array(_, count) => Some(*count),
        TypeExpr::Slice(_) => match facts.get(&value) {
            Some(LengthFact::Known(len)) => Some(*len),
            _ => None,
        },
        _ => None,
    }
}

fn is_slice_type(typ: &crate::compiler::ssa::hlssa::Type) -> bool {
    matches!(typ.expr, TypeExpr::Slice(_))
}

#[allow(clippy::too_many_arguments)]
fn infer_function(
    function_id: FunctionId,
    function: &HLFunction,
    fn_type_info: &FunctionTypeInfo,
    ssa: &HLSSA,
    values: &mut HashMap<FunctionId, LengthFacts>,
    returns: &mut HashMap<FunctionId, HashMap<usize, LengthFact>>,
) -> bool {
    let mut changed = false;

    for (_, block) in function.get_blocks() {
        for instruction in block.get_instructions() {
            match instruction {
                OpCode::MkSeq {
                    result,
                    elems,
                    seq_type: SequenceTargetType::Slice,
                    ..
                } => {
                    let facts = values.entry(function_id).or_default();
                    changed |= record_slice_length(
                        facts,
                        *result,
                        elems.len(),
                        "slice literal has conflicting concrete lengths",
                    );
                }
                OpCode::MkRepeated {
                    result,
                    seq_type: SequenceTargetType::Slice,
                    count,
                    ..
                } => {
                    let facts = values.entry(function_id).or_default();
                    changed |= record_slice_length(
                        facts,
                        *result,
                        *count,
                        "repeated slice has conflicting concrete lengths",
                    );
                }
                OpCode::Cast {
                    result,
                    value,
                    target,
                } if matches!(
                    fn_type_info.get_value_type(*result).expr,
                    TypeExpr::Slice(_)
                ) =>
                {
                    let facts = values.entry(function_id).or_default();
                    let len = match target {
                        CastTarget::ArrayToSlice | CastTarget::Nop => {
                            known_value_length(facts, fn_type_info, *value)
                        }
                        _ => None,
                    };
                    if let Some(len) = len {
                        changed |= record_slice_length(
                            facts,
                            *result,
                            len,
                            "slice cast has conflicting concrete lengths",
                        );
                    } else {
                        changed |= copy_slice_length_fact(
                            &facts.clone(),
                            facts,
                            *value,
                            *result,
                            "slice cast has conflicting concrete lengths",
                        );
                    }
                }
                OpCode::ArraySet { result, array, .. }
                    if matches!(
                        fn_type_info.get_value_type(*result).expr,
                        TypeExpr::Slice(_)
                    ) =>
                {
                    let facts = values.entry(function_id).or_default();
                    if let Some(len) = known_value_length(facts, fn_type_info, *array) {
                        changed |= record_slice_length(
                            facts,
                            *result,
                            len,
                            "slice update has conflicting concrete lengths",
                        );
                    } else {
                        changed |= copy_slice_length_fact(
                            &facts.clone(),
                            facts,
                            *array,
                            *result,
                            "slice update has conflicting concrete lengths",
                        );
                    }
                }
                OpCode::SlicePush {
                    result,
                    slice,
                    values: pushed,
                    ..
                } => {
                    let facts = values.entry(function_id).or_default();
                    if let Some(len) = known_value_length(facts, fn_type_info, *slice) {
                        changed |= record_slice_length(
                            facts,
                            *result,
                            len + pushed.len(),
                            "slice push has conflicting concrete lengths",
                        );
                    } else {
                        changed |= copy_slice_length_fact(
                            &facts.clone(),
                            facts,
                            *slice,
                            *result,
                            "slice push has conflicting concrete lengths",
                        );
                    }
                }
                OpCode::Select {
                    result, if_t, if_f, ..
                } if matches!(
                    fn_type_info.get_value_type(*result).expr,
                    TypeExpr::Slice(_)
                ) =>
                {
                    let facts = values.entry(function_id).or_default();
                    let t_len = known_value_length(facts, fn_type_info, *if_t);
                    let f_len = known_value_length(facts, fn_type_info, *if_f);
                    if let (Some(t_len), Some(f_len)) = (t_len, f_len) {
                        if t_len == f_len {
                            changed |= record_slice_length(
                                facts,
                                *result,
                                t_len,
                                "select result has conflicting concrete slice lengths",
                            );
                        } else {
                            changed |= record_unknown_slice_length(
                                facts,
                                *result,
                                format!(
                                    "select branches produce different concrete slice lengths {t_len} and {f_len}"
                                ),
                            );
                        }
                    } else {
                        let reason = match (facts.get(if_t), facts.get(if_f)) {
                            (Some(LengthFact::Unknown(reason)), _) => Some(reason.clone()),
                            (_, Some(LengthFact::Unknown(reason))) => Some(reason.clone()),
                            _ => None,
                        };
                        if let Some(reason) = reason {
                            changed |= record_unknown_slice_length(facts, *result, reason);
                        }
                    }
                }
                OpCode::Call {
                    results,
                    function: CallTarget::Static(callee_id),
                    args,
                    ..
                } => {
                    let caller_facts = values.entry(function_id).or_default().clone();
                    let callee = ssa.get_function(*callee_id);

                    for ((param, param_type), arg) in
                        callee.get_entry().get_parameters().zip(args.iter())
                    {
                        if is_slice_type(param_type) {
                            if let Some(len) = known_value_length(&caller_facts, fn_type_info, *arg)
                            {
                                let callee_facts = values.entry(*callee_id).or_default();
                                changed |= record_slice_length(
                                    callee_facts,
                                    *param,
                                    len,
                                    &format!(
                                        "function '{}' slice parameter {param:?} is reached with multiple concrete lengths from call sites",
                                        callee.get_name()
                                    ),
                                );
                            } else if let Some(LengthFact::Unknown(reason)) = caller_facts.get(arg)
                            {
                                let callee_facts = values.entry(*callee_id).or_default();
                                changed |= record_unknown_slice_length(
                                    callee_facts,
                                    *param,
                                    reason.clone(),
                                );
                            }
                        }
                    }

                    for (index, result) in results.iter().enumerate() {
                        if matches!(
                            fn_type_info.get_value_type(*result).expr,
                            TypeExpr::Slice(_)
                        ) {
                            if let Some(fact) = return_length_fact(returns, *callee_id, index) {
                                let facts = values.entry(function_id).or_default();
                                match fact {
                                    LengthFact::Known(len) => {
                                        changed |= record_slice_length(
                                            facts,
                                            *result,
                                            *len,
                                            "call result has conflicting concrete slice lengths",
                                        );
                                    }
                                    LengthFact::Unknown(reason) => {
                                        changed |= record_unknown_slice_length(
                                            facts,
                                            *result,
                                            reason.clone(),
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        match block.get_terminator() {
            Some(Terminator::Jmp(target, args)) => {
                let target_block = function.get_block(*target);
                let facts_snapshot = values.entry(function_id).or_default().clone();
                for ((param, param_type), arg) in target_block.get_parameters().zip(args) {
                    if is_slice_type(param_type) {
                        if let Some(len) = known_value_length(&facts_snapshot, fn_type_info, *arg) {
                            let facts = values.entry(function_id).or_default();
                            changed |= record_slice_length(
                                facts,
                                *param,
                                len,
                                "block parameter has conflicting concrete slice lengths",
                            );
                        } else if let Some(LengthFact::Unknown(reason)) = facts_snapshot.get(arg) {
                            let facts = values.entry(function_id).or_default();
                            changed |= record_unknown_slice_length(facts, *param, reason.clone());
                        }
                    }
                }
            }
            Some(Terminator::Return(return_values)) => {
                let facts = values.entry(function_id).or_default();
                for (index, (value, return_type)) in
                    return_values.iter().zip(function.get_returns()).enumerate()
                {
                    if is_slice_type(return_type) {
                        if let Some(len) = known_value_length(facts, fn_type_info, *value) {
                            changed |= record_return_length(
                                returns,
                                function_id,
                                index,
                                len,
                                &format!(
                                    "function '{}' return slot {index} has multiple concrete slice lengths",
                                    function.get_name()
                                ),
                            );
                        } else if let Some(LengthFact::Unknown(reason)) = facts.get(value) {
                            changed |= record_unknown_return_length(
                                returns,
                                function_id,
                                index,
                                reason.clone(),
                            );
                        }
                    }
                }
            }
            _ => {}
        }
    }

    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{
        analysis::{flow_analysis::FlowAnalysis, types::Types},
        ssa::hlssa::{SliceOpDir, Type},
    };

    fn analyze(ssa: &HLSSA) -> SliceLengthAnalysis {
        let flow_analysis = FlowAnalysis::run(ssa);
        let type_info = Types::new().run(ssa, &flow_analysis);
        SliceLengthAnalysis::run(ssa, &type_info)
    }

    #[test]
    fn unknown_downgrades_report_changes() {
        let mut facts = LengthFacts::new();
        let value = ValueId(1);
        assert!(record_slice_length(&mut facts, value, 3, "first length"));
        assert!(record_unknown_slice_length(
            &mut facts,
            value,
            "ambiguous".to_string()
        ));
        assert!(!record_unknown_slice_length(
            &mut facts,
            value,
            "still ambiguous".to_string()
        ));

        let mut returns = HashMap::new();
        let function = FunctionId(0);
        assert!(record_return_length(
            &mut returns,
            function,
            0,
            3,
            "first return"
        ));
        assert!(record_unknown_return_length(
            &mut returns,
            function,
            0,
            "ambiguous return".to_string()
        ));
        assert!(!record_unknown_return_length(
            &mut returns,
            function,
            0,
            "still ambiguous".to_string()
        ));
    }

    #[test]
    fn infers_slice_lengths_across_call_params_and_returns() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_main_id();
        let set_id = ssa.add_function("set".to_string());
        let field = Type::field();
        let slice = field.clone().slice_of();

        let set_slice = ssa.fresh_value();
        let set_index = ssa.fresh_value();
        let set_value = ssa.fresh_value();
        let set_result = ssa.fresh_value();
        {
            let set = ssa.get_function_mut(set_id);
            set.add_return_type(slice.clone());
            let entry = set.get_entry_id();
            set.get_block_mut(entry)
                .push_parameter(set_slice, slice.clone());
            set.get_block_mut(entry)
                .push_parameter(set_index, Type::u(32));
            set.get_block_mut(entry)
                .push_parameter(set_value, field.clone());
            set.get_block_mut(entry).push_instruction(OpCode::ArraySet {
                result: set_result,
                array: set_slice,
                index: set_index,
                value: set_value,
            });
            set.terminate_block_with_return(entry, vec![set_result]);
        }

        let main_a = ssa.fresh_value();
        let main_b = ssa.fresh_value();
        let main_c = ssa.fresh_value();
        let main_index = ssa.fresh_value();
        let main_value = ssa.fresh_value();
        let main_slice = ssa.fresh_value();
        let call_result = ssa.fresh_value();
        {
            let main = ssa.get_function_mut(main_id);
            let entry = main.get_entry_id();
            main.get_block_mut(entry)
                .push_parameter(main_a, field.clone());
            main.get_block_mut(entry)
                .push_parameter(main_b, field.clone());
            main.get_block_mut(entry)
                .push_parameter(main_c, field.clone());
            main.get_block_mut(entry)
                .push_parameter(main_index, Type::u(32));
            main.get_block_mut(entry)
                .push_parameter(main_value, field.clone());
            main.get_block_mut(entry).push_instruction(OpCode::MkSeq {
                result: main_slice,
                elems: vec![main_a, main_b, main_c],
                seq_type: SequenceTargetType::Slice,
                elem_type: field.clone(),
            });
            main.get_block_mut(entry).push_instruction(OpCode::Call {
                results: vec![call_result],
                function: CallTarget::Static(set_id),
                args: vec![main_slice, main_index, main_value],
                unconstrained: false,
            });
            main.terminate_block_with_return(entry, vec![]);
        }

        let analysis = analyze(&ssa);

        assert_eq!(
            analysis.function_values(set_id).unwrap().get(&set_slice),
            Some(&3)
        );
        assert_eq!(
            analysis.function_values(main_id).unwrap().get(&call_result),
            Some(&3)
        );
    }

    #[test]
    fn infers_slice_lengths_from_call_return_producers() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_main_id();
        let make_id = ssa.add_function("make".to_string());
        let field = Type::field();
        let slice = field.clone().slice_of();

        let make_a = ssa.fresh_value();
        let make_b = ssa.fresh_value();
        let make_c = ssa.fresh_value();
        let make_slice = ssa.fresh_value();
        {
            let make = ssa.get_function_mut(make_id);
            make.add_return_type(slice);
            let entry = make.get_entry_id();
            make.get_block_mut(entry)
                .push_parameter(make_a, field.clone());
            make.get_block_mut(entry)
                .push_parameter(make_b, field.clone());
            make.get_block_mut(entry)
                .push_parameter(make_c, field.clone());
            make.get_block_mut(entry).push_instruction(OpCode::MkSeq {
                result: make_slice,
                elems: vec![make_a, make_b, make_c],
                seq_type: SequenceTargetType::Slice,
                elem_type: field.clone(),
            });
            make.terminate_block_with_return(entry, vec![make_slice]);
        }

        let main_a = ssa.fresh_value();
        let main_b = ssa.fresh_value();
        let main_c = ssa.fresh_value();
        let call_result = ssa.fresh_value();
        {
            let main = ssa.get_function_mut(main_id);
            let entry = main.get_entry_id();
            main.get_block_mut(entry)
                .push_parameter(main_a, field.clone());
            main.get_block_mut(entry)
                .push_parameter(main_b, field.clone());
            main.get_block_mut(entry)
                .push_parameter(main_c, field.clone());
            main.get_block_mut(entry).push_instruction(OpCode::Call {
                results: vec![call_result],
                function: CallTarget::Static(make_id),
                args: vec![main_a, main_b, main_c],
                unconstrained: false,
            });
            main.terminate_block_with_return(entry, vec![]);
        }

        let analysis = analyze(&ssa);

        assert_eq!(
            analysis.function_values(main_id).unwrap().get(&call_result),
            Some(&3)
        );
    }

    #[test]
    fn infers_slice_push_result_lengths() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_main_id();
        let field = Type::field();

        let main_a = ssa.fresh_value();
        let main_b = ssa.fresh_value();
        let pushed = ssa.fresh_value();
        let main_slice = ssa.fresh_value();
        let push_result = ssa.fresh_value();
        {
            let main = ssa.get_function_mut(main_id);
            let entry = main.get_entry_id();
            main.get_block_mut(entry)
                .push_parameter(main_a, field.clone());
            main.get_block_mut(entry)
                .push_parameter(main_b, field.clone());
            main.get_block_mut(entry)
                .push_parameter(pushed, field.clone());
            main.get_block_mut(entry).push_instruction(OpCode::MkSeq {
                result: main_slice,
                elems: vec![main_a, main_b],
                seq_type: SequenceTargetType::Slice,
                elem_type: field.clone(),
            });
            main.get_block_mut(entry)
                .push_instruction(OpCode::SlicePush {
                    dir: SliceOpDir::Back,
                    result: push_result,
                    slice: main_slice,
                    values: vec![pushed],
                });
            main.terminate_block_with_return(entry, vec![]);
        }

        let analysis = analyze(&ssa);

        assert_eq!(
            analysis.function_values(main_id).unwrap().get(&push_result),
            Some(&3)
        );
    }

    #[test]
    #[should_panic(expected = "does not have a single concrete length")]
    fn conflicting_call_site_lengths_are_reported_as_ambiguous() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_main_id();
        let helper_id = ssa.add_function("helper".to_string());
        let field = Type::field();
        let slice = field.clone().slice_of();

        let helper_slice = ssa.fresh_value();
        {
            let helper = ssa.get_function_mut(helper_id);
            let entry = helper.get_entry_id();
            helper
                .get_block_mut(entry)
                .push_parameter(helper_slice, slice);
            helper.terminate_block_with_return(entry, vec![]);
        }

        let a = ssa.fresh_value();
        let b = ssa.fresh_value();
        let c = ssa.fresh_value();
        let first_slice = ssa.fresh_value();
        let second_slice = ssa.fresh_value();
        {
            let main = ssa.get_function_mut(main_id);
            let entry = main.get_entry_id();
            main.get_block_mut(entry).push_parameter(a, field.clone());
            main.get_block_mut(entry).push_parameter(b, field.clone());
            main.get_block_mut(entry).push_parameter(c, field.clone());
            main.get_block_mut(entry).push_instruction(OpCode::MkSeq {
                result: first_slice,
                elems: vec![a, b, c],
                seq_type: SequenceTargetType::Slice,
                elem_type: field.clone(),
            });
            main.get_block_mut(entry).push_instruction(OpCode::MkSeq {
                result: second_slice,
                elems: vec![a, b],
                seq_type: SequenceTargetType::Slice,
                elem_type: field,
            });
            main.get_block_mut(entry).push_instruction(OpCode::Call {
                results: vec![],
                function: CallTarget::Static(helper_id),
                args: vec![first_slice],
                unconstrained: false,
            });
            main.get_block_mut(entry).push_instruction(OpCode::Call {
                results: vec![],
                function: CallTarget::Static(helper_id),
                args: vec![second_slice],
                unconstrained: false,
            });
            main.terminate_block_with_return(entry, vec![]);
        }

        let analysis = analyze(&ssa);
        assert_eq!(
            analysis
                .function_values(helper_id)
                .unwrap()
                .get(&helper_slice),
            None
        );
        analysis
            .function_values(helper_id)
            .unwrap()
            .require(helper_slice, "test");
    }
}
