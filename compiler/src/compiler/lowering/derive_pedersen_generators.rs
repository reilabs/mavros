//! Compile-time lowering for Noir's `derive_pedersen_generators` builtin.
//!
//! Noir deliberately requires both builtin arguments to be compile-time constants. The
//! monomorphized AST still keeps the builtin behind its `#[inline_always]` stdlib wrappers, while
//! Mavros lowers that AST directly instead of running Noir's SSA inliner. This module performs the
//! narrow specialization those wrappers need: clone only call paths which reach a derivation,
//! substitute constants from each call site, then replace the builtin placeholder with ordinary
//! tuple/array construction.

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        Field,
        analysis::click_cooper::lattice,
        ssa::hlssa::{
            Blob, CallTarget, Constant, HLFunction, HLSSA, OpCode, SequenceTargetType, Type,
            TypeExpr,
        },
        ssa::{BlockId, FunctionId, Instruction, SourceLocation, Terminator, ValueId},
    },
};

const PLACEHOLDER: &str = "compile-time derive_pedersen_generators";

#[derive(Clone, Debug)]
pub(super) struct PendingDerivation {
    pub result: ValueId,
    pub domain_separator: ValueId,
    pub starting_index: ValueId,
    pub num_generators: u32,
}

#[derive(Clone)]
struct CallSite {
    block: BlockId,
    index: usize,
    callee: FunctionId,
    args: Vec<ValueId>,
}

/// Resolve all pending derivations before the newly lowered SSA is exposed to compiler passes.
pub(super) fn lower(ssa: &mut HLSSA, mut derivations: HashMap<FunctionId, Vec<PendingDerivation>>) {
    if derivations.is_empty() {
        return;
    }

    let relevant = functions_reaching_derivation(ssa, derivations.keys().copied().collect());
    let mut roots: Vec<_> = ssa.get_entry_points().to_vec();
    roots.extend(ssa.get_globals_init_fn());
    roots.sort_by_key(|id| id.0);
    roots.dedup();

    let mut clone_number = 0usize;
    for root in roots {
        let mut lineage = vec![root];
        specialize_function(
            ssa,
            root,
            &relevant,
            &mut derivations,
            &mut lineage,
            &mut clone_number,
        );
    }

    // Every reachable placeholder must have been materialized. Leaving one behind would turn a
    // missing constant-propagation case into a much later and less useful codegen error.
    let mut reachable_roots = ssa.get_entry_points().to_vec();
    reachable_roots.extend(ssa.get_globals_init_fn());
    let reachable = reachable_functions(ssa, &reachable_roots);
    for function_id in reachable {
        assert!(
            !contains_placeholder(ssa.get_function(function_id)),
            "derive_pedersen_generators arguments must be compile-time constants"
        );
    }
}

fn functions_reaching_derivation(
    ssa: &HLSSA,
    mut relevant: HashSet<FunctionId>,
) -> HashSet<FunctionId> {
    loop {
        let mut changed = false;
        for (function_id, function) in ssa.iter_functions() {
            if relevant.contains(function_id) {
                continue;
            }
            if static_callees(function).any(|callee| relevant.contains(&callee)) {
                changed |= relevant.insert(*function_id);
            }
        }
        if !changed {
            return relevant;
        }
    }
}

fn specialize_function(
    ssa: &mut HLSSA,
    function_id: FunctionId,
    relevant: &HashSet<FunctionId>,
    derivations: &mut HashMap<FunctionId, Vec<PendingDerivation>>,
    lineage: &mut Vec<FunctionId>,
    clone_number: &mut usize,
) {
    materialize_derivations(ssa, function_id, derivations);

    let constants = known_constants(ssa, ssa.get_function(function_id));
    let mut calls = collect_relevant_calls(ssa.get_function(function_id), relevant);
    calls.sort_by_key(|site| (site.block.0, site.index));

    for call in calls {
        assert!(
            !lineage.contains(&call.callee),
            "recursive call path reaches derive_pedersen_generators"
        );

        let argument_constants: Vec<_> = call
            .args
            .iter()
            .map(|argument| constant_for(ssa, &constants, *argument))
            .collect();
        let (clone, remap) = ssa.duplicate_function_with_remap(call.callee);
        *clone_number += 1;
        let original_name = ssa.get_function(call.callee).get_name().to_string();
        ssa.get_function_mut(clone)
            .set_name(format!("{original_name}#derive_pedersen_{}", *clone_number));

        if let Some(original_sites) = derivations.get(&call.callee).cloned() {
            let cloned_sites = original_sites
                .into_iter()
                .map(|site| remap_derivation(site, &remap))
                .collect();
            derivations.insert(clone, cloned_sites);
        }

        substitute_constant_parameters(ssa, clone, &argument_constants, derivations);
        rewrite_call_target(ssa, function_id, &call, clone);
        lineage.push(call.callee);
        specialize_function(ssa, clone, relevant, derivations, lineage, clone_number);
        lineage.pop();
    }
}

fn remap_derivation(
    mut site: PendingDerivation,
    remap: &HashMap<ValueId, ValueId>,
) -> PendingDerivation {
    for value in [
        &mut site.result,
        &mut site.domain_separator,
        &mut site.starting_index,
    ] {
        if let Some(remapped) = remap.get(value) {
            *value = *remapped;
        }
    }
    site
}

fn substitute_constant_parameters(
    ssa: &mut HLSSA,
    function_id: FunctionId,
    argument_constants: &[Option<Constant>],
    derivations: &mut HashMap<FunctionId, Vec<PendingDerivation>>,
) {
    let parameters: Vec<_> = ssa
        .get_function(function_id)
        .get_entry()
        .get_parameters()
        .map(|(value, typ)| (*value, typ.clone()))
        .collect();
    let mut replacements = HashMap::default();
    let mut aggregate_initializers = Vec::new();
    for ((parameter, parameter_type), constant) in parameters.into_iter().zip(argument_constants) {
        let Some(constant) = constant else {
            continue;
        };
        let constant_id = ssa.add_const(constant.clone());
        let replacement = if let (Constant::Blob(blob), TypeExpr::Array(..)) =
            (&constant, &parameter_type.expr)
        {
            // Blobs are constant-pool storage, not runtime arrays. Re-view one as an array before
            // substituting it for an array parameter, exactly as literal lowering does.
            let array = ssa.fresh_value();
            aggregate_initializers.push(
                OpCode::MkSeqOfBlob {
                    result: array,
                    element_type: blob.elem_type.clone(),
                    blob: constant_id,
                }
                .locate(SourceLocation::synthetic("derive_pedersen_generators")),
            );
            array
        } else if constant.is_scalar() {
            constant_id
        } else {
            // A blob can only be re-viewed directly as a fixed array. Do not silently substitute
            // it for a slice (whose runtime representation includes a length).
            continue;
        };
        replacements.insert(parameter, replacement);
    }

    let function = ssa.get_function_mut(function_id);
    if !aggregate_initializers.is_empty() {
        let entry = function.get_entry_mut();
        aggregate_initializers.extend(entry.take_instructions());
        entry.put_instructions(aggregate_initializers);
    }
    for (_, block) in function.get_blocks_mut() {
        for instruction in block.get_instructions_mut() {
            for input in instruction.get_inputs_mut() {
                if let Some(replacement) = replacements.get(input) {
                    *input = *replacement;
                }
            }
        }
        match block.get_terminator_mut() {
            Terminator::Jmp(_, args) | Terminator::Return(args) => {
                for argument in args {
                    if let Some(replacement) = replacements.get(argument) {
                        *argument = *replacement;
                    }
                }
            }
            Terminator::JmpIf(condition, _, _) => {
                if let Some(replacement) = replacements.get(condition) {
                    *condition = *replacement;
                }
            }
        }
    }

    if let Some(sites) = derivations.get_mut(&function_id) {
        for site in sites {
            if let Some(replacement) = replacements.get(&site.domain_separator) {
                site.domain_separator = *replacement;
            }
            if let Some(replacement) = replacements.get(&site.starting_index) {
                site.starting_index = *replacement;
            }
        }
    }
}

fn materialize_derivations(
    ssa: &mut HLSSA,
    function_id: FunctionId,
    derivations: &HashMap<FunctionId, Vec<PendingDerivation>>,
) {
    let Some(sites) = derivations.get(&function_id) else {
        return;
    };
    let constants = known_constants(ssa, ssa.get_function(function_id));

    for site in sites {
        let domain = constant_for(ssa, &constants, site.domain_separator)
            .and_then(domain_separator_bytes)
            .unwrap_or_else(|| {
                panic!("derive_pedersen_generators domain separator must be a constant [u8]")
            });
        let starting_index = constant_for(ssa, &constants, site.starting_index)
            .and_then(constant_u32)
            .unwrap_or_else(|| {
                panic!("derive_pedersen_generators starting index must be a constant u32")
            });
        let generators =
            bn254_blackbox_solver::derive_generators(&domain, site.num_generators, starting_index)
                .into_iter()
                .map(|generator| (Field::from(generator.x), Field::from(generator.y)))
                .collect();

        replace_placeholder(ssa, function_id, site, generators);
    }
}

fn replace_placeholder(
    ssa: &mut HLSSA,
    function_id: FunctionId,
    site: &PendingDerivation,
    generators: Vec<(Field, Field)>,
) {
    let point_type = Type::tuple_of(vec![Type::field(), Type::field()]);
    let mut replacement = Vec::with_capacity(generators.len() + 1);
    let mut points = Vec::with_capacity(generators.len());

    for (x, y) in generators {
        let x = ssa.add_const(Constant::Field(x));
        let y = ssa.add_const(Constant::Field(y));
        let point = ssa.fresh_value();
        points.push(point);
        replacement.push(OpCode::MkTuple {
            result: point,
            elems: vec![x, y],
            element_types: vec![Type::field(), Type::field()],
        });
    }
    replacement.push(OpCode::MkSeq {
        result: site.result,
        elems: points,
        seq_type: SequenceTargetType::Array(site.num_generators as usize),
        elem_type: point_type,
    });

    let function = ssa.get_function_mut(function_id);
    let mut replaced = false;
    for (_, block) in function.get_blocks_mut() {
        let instructions = block.take_instructions();
        let mut lowered = Vec::with_capacity(instructions.len() + replacement.len());
        for instruction in instructions {
            let is_site = matches!(
                &*instruction,
                OpCode::Todo { payload, results, .. }
                    if payload == PLACEHOLDER && results.as_slice() == [site.result]
            );
            if is_site {
                assert!(
                    !replaced,
                    "duplicate derive_pedersen_generators placeholder"
                );
                let location = instruction.location().clone();
                lowered.extend(
                    replacement
                        .iter()
                        .cloned()
                        .map(|instruction| instruction.locate(location.clone())),
                );
                replaced = true;
            } else {
                lowered.push(instruction);
            }
        }
        block.put_instructions(lowered);
    }
    assert!(replaced, "missing derive_pedersen_generators placeholder");
}

fn collect_relevant_calls(function: &HLFunction, relevant: &HashSet<FunctionId>) -> Vec<CallSite> {
    let mut calls = Vec::new();
    for (block_id, block) in function.get_blocks() {
        for (index, instruction) in block.get_instructions().enumerate() {
            if let OpCode::Call {
                function: CallTarget::Static(callee),
                args,
                ..
            } = instruction
                && relevant.contains(callee)
            {
                calls.push(CallSite {
                    block: *block_id,
                    index,
                    callee: *callee,
                    args: args.clone(),
                });
            }
        }
    }
    calls
}

fn rewrite_call_target(ssa: &mut HLSSA, caller: FunctionId, site: &CallSite, clone: FunctionId) {
    let instruction = ssa
        .get_function_mut(caller)
        .get_block_mut(site.block)
        .get_instructions_mut()
        .nth(site.index)
        .expect("derive_pedersen_generators call site disappeared");
    let OpCode::Call {
        function: CallTarget::Static(callee),
        ..
    } = instruction
    else {
        panic!("derive_pedersen_generators call site changed shape")
    };
    assert_eq!(*callee, site.callee);
    *callee = clone;
}

fn known_constants(ssa: &HLSSA, function: &HLFunction) -> HashMap<ValueId, Constant> {
    let mut known = HashMap::default();
    loop {
        let mut changed = false;
        for (_, block) in function.get_blocks() {
            for instruction in block.get_instructions() {
                let folded = match instruction {
                    OpCode::BinaryArithOp {
                        kind,
                        result,
                        lhs,
                        rhs,
                    } => binary_constants(ssa, &known, *lhs, *rhs)
                        .and_then(|(lhs, rhs)| lattice::eval_binary(*kind, &lhs, &rhs, ssa.field()))
                        .map(|constant| (*result, constant)),
                    OpCode::Cmp {
                        kind,
                        result,
                        lhs,
                        rhs,
                    } => binary_constants(ssa, &known, *lhs, *rhs)
                        .and_then(|(lhs, rhs)| lattice::eval_cmp(*kind, &lhs, &rhs))
                        .map(|constant| (*result, constant)),
                    OpCode::MkSeq {
                        result,
                        elems,
                        elem_type,
                        ..
                    } => all_constants(ssa, &known, elems).map(|elements| {
                        (
                            *result,
                            Constant::Blob(Blob::new(elem_type.clone(), elements)),
                        )
                    }),
                    OpCode::MkSeqOfBlob { result, blob, .. } => {
                        constant_for(ssa, &known, *blob).map(|constant| (*result, constant))
                    }
                    OpCode::MkRepeated {
                        result,
                        element,
                        count,
                        elem_type,
                        ..
                    } => constant_for(ssa, &known, *element).map(|element| {
                        (
                            *result,
                            Constant::Blob(Blob::new(elem_type.clone(), vec![element; *count])),
                        )
                    }),
                    OpCode::Cast {
                        result,
                        value,
                        target,
                    } => constant_for(ssa, &known, *value)
                        .and_then(|constant| lattice::eval_cast(target, &constant, ssa.field()))
                        .map(|constant| (*result, constant)),
                    OpCode::SExt {
                        result,
                        value,
                        from_bits,
                        to_bits,
                    } => constant_for(ssa, &known, *value)
                        .and_then(|constant| lattice::eval_sext(&constant, *from_bits, *to_bits))
                        .map(|constant| (*result, constant)),
                    OpCode::BitRange {
                        result,
                        value,
                        offset,
                        width,
                    } => constant_for(ssa, &known, *value)
                        .and_then(|constant| lattice::eval_bit_range(&constant, *offset, *width))
                        .map(|constant| (*result, constant)),
                    OpCode::Not { result, value } => constant_for(ssa, &known, *value)
                        .and_then(|constant| lattice::eval_not(&constant))
                        .map(|constant| (*result, constant)),
                    OpCode::ArrayGet {
                        result,
                        array,
                        index,
                    } => binary_constants(ssa, &known, *array, *index)
                        .and_then(|(array, index)| lattice::eval_array_get(&array, &index))
                        .map(|constant| (*result, constant)),
                    OpCode::ArraySet {
                        result,
                        array,
                        index,
                        value,
                    } => constant_for(ssa, &known, *array)
                        .zip(constant_for(ssa, &known, *index))
                        .zip(constant_for(ssa, &known, *value))
                        .and_then(|((array, index), value)| {
                            lattice::eval_array_set(array, &index, value)
                        })
                        .map(|constant| (*result, constant)),
                    OpCode::Select {
                        result,
                        cond,
                        if_t,
                        if_f,
                    } => constant_for(ssa, &known, *cond)
                        .and_then(|condition| match condition {
                            Constant::U(1, 0) => constant_for(ssa, &known, *if_f),
                            Constant::U(1, 1) => constant_for(ssa, &known, *if_t),
                            _ => None,
                        })
                        .map(|constant| (*result, constant)),
                    _ => None,
                };
                if let Some((result, constant)) = folded
                    && known.get(&result) != Some(&constant)
                {
                    known.insert(result, constant);
                    changed = true;
                }
            }
        }
        if !changed {
            return known;
        }
    }
}

fn binary_constants(
    ssa: &HLSSA,
    known: &HashMap<ValueId, Constant>,
    lhs: ValueId,
    rhs: ValueId,
) -> Option<(Constant, Constant)> {
    Some((
        constant_for(ssa, known, lhs)?,
        constant_for(ssa, known, rhs)?,
    ))
}

fn all_constants(
    ssa: &HLSSA,
    known: &HashMap<ValueId, Constant>,
    values: &[ValueId],
) -> Option<Vec<Constant>> {
    values
        .iter()
        .map(|value| constant_for(ssa, known, *value))
        .collect()
}

fn constant_for(
    ssa: &HLSSA,
    known: &HashMap<ValueId, Constant>,
    value: ValueId,
) -> Option<Constant> {
    ssa.get_const(value)
        .map(|constant| constant.as_ref().clone())
        .or_else(|| known.get(&value).cloned())
}

fn domain_separator_bytes(constant: Constant) -> Option<Vec<u8>> {
    let Constant::Blob(blob) = constant else {
        return None;
    };
    blob.elements
        .into_iter()
        .map(|element| match element {
            Constant::U(8, value) => u8::try_from(value).ok(),
            _ => None,
        })
        .collect()
}

fn constant_u32(constant: Constant) -> Option<u32> {
    match constant {
        Constant::U(bits, value) if bits <= 32 => u32::try_from(value).ok(),
        _ => None,
    }
}

fn static_callees(function: &HLFunction) -> impl Iterator<Item = FunctionId> + '_ {
    function
        .get_blocks()
        .flat_map(|(_, block)| block.get_instructions())
        .flat_map(Instruction::get_static_call_targets)
}

fn contains_placeholder(function: &HLFunction) -> bool {
    function.get_blocks().any(|(_, block)| {
        block.get_instructions().any(
            |instruction| matches!(instruction, OpCode::Todo { payload, .. } if payload == PLACEHOLDER),
        )
    })
}

fn reachable_functions(ssa: &HLSSA, roots: &[FunctionId]) -> HashSet<FunctionId> {
    let mut reachable = HashSet::default();
    let mut pending = roots.to_vec();
    while let Some(function_id) = pending.pop() {
        if reachable.insert(function_id) {
            pending.extend(static_callees(ssa.get_function(function_id)));
        }
    }
    reachable
}
