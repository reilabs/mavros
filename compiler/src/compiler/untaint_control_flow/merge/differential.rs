//! Execute the real sparse emitter and counted-loop fallback, independently checking
//! both against a flat-vector model. No test interpreter or production mode switch:
//! the existing guard lowering, DCE, and symbolic executor execute the generated SSA.

use super::*;
use crate::compiler::{
    analysis::{flow_analysis::FlowAnalysis, symbolic_executor::SymbolicExecutor, types::Types},
    codegen::hlssa_to_r1cs::{R1CGen, Value},
    pass_manager::{AnalysisStore, Pass},
    passes::{
        dead_code_elimination::{Config as DceConfig, DCE},
        instruction_lowering::InstructionLowering,
        strip_witness_of::StripWitnessOf,
    },
    ssa::{
        Terminator,
        hlssa::{HLSSA, OpCode, builder::HLFunctionBuilder},
    },
};
use mavros_artifacts::FieldConfig;
use proptest::prelude::*;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

#[derive(Clone, Debug)]
struct Write {
    keys: Vec<usize>,
    overwrite: bool,
    delta: u64,
}

#[derive(Clone, Debug)]
struct Case {
    shape: Vec<usize>,
    writes: Vec<Write>,
    indices: [u32; 3],
    bits: usize,
    integer_leaves: bool,
    seed: u64,
}

fn cases() -> impl Strategy<Value = Case> {
    (
        prop::collection::vec(1usize..5, 1..4),
        prop::collection::vec(
            (
                prop::collection::vec(0usize..3, 1..4),
                any::<bool>(),
                1u64..8,
            ),
            1..9,
        ),
        prop::array::uniform3(prop_oneof![0u32..6, Just(u32::MAX)]),
        prop::sample::select(vec![1usize, 8, 32]),
        any::<bool>(),
        1u64..50,
    )
        .prop_map(
            |(shape, writes, indices, bits, integer_leaves, seed)| Case {
                writes: writes
                    .into_iter()
                    .map(|(mut keys, overwrite, delta)| {
                        keys.truncate(shape.len());
                        Write {
                            keys,
                            overwrite,
                            delta,
                        }
                    })
                    .collect(),
                shape,
                indices: indices.map(|i| if bits == 32 { i } else { i & ((1 << bits) - 1) }),
                bits,
                integer_leaves,
                seed,
            },
        )
}

fn array_type(shape: &[usize], leaf: &Type) -> Type {
    shape
        .iter()
        .rev()
        .fold(leaf.clone(), |t, &len| t.array_of(len))
}

fn repeated(
    b: &mut HLBlockEmitter<'_>,
    shape: &[usize],
    leaf: ValueId,
    leaf_type: &Type,
) -> ValueId {
    if shape.is_empty() {
        return leaf;
    }
    let value = repeated(b, &shape[1..], leaf, leaf_type);
    b.mk_repeated(
        value,
        crate::compiler::ssa::hlssa::SequenceTargetType::Array(shape[0]),
        shape[0],
        array_type(&shape[1..], leaf_type),
    )
}

fn update(
    b: &mut HLBlockEmitter<'_>,
    array: ValueId,
    shape: &[usize],
    keys: &[usize],
    indices: &[ValueId],
    write: &Write,
    leaf_type: &Type,
) -> ValueId {
    let index = indices[keys[0]];
    let value = if keys.len() > 1 {
        let row = b.array_get(array, index);
        update(b, row, &shape[1..], &keys[1..], indices, write, leaf_type)
    } else {
        let delta = if leaf_type.strip_witness().is_field() {
            b.field_const(b.field().constant(write.delta as u128))
        } else {
            b.int_const(64, write.delta as u128)
        };
        if shape.len() == 1 && !write.overwrite {
            let old = b.array_get(array, index);
            b.uadd(old, delta)
        } else {
            let delta = b.cast_to_witness_of(delta);
            repeated(b, &shape[1..], delta, leaf_type)
        }
    };
    b.array_set(array, index, value)
}

fn assert_leaves(
    b: &mut HLBlockEmitter<'_>,
    array: ValueId,
    shape: &[usize],
    expected: &[ValueId],
    cursor: &mut usize,
    check_values: ValueId,
) {
    if shape.is_empty() {
        b.emit_guarded(
            Some(check_values),
            OpCode::AssertCmp {
                kind: crate::compiler::ssa::hlssa::CmpKind::Eq,
                lhs: array,
                rhs: expected[*cursor],
            },
        );
        *cursor += 1;
    } else {
        for i in 0..shape[0] {
            let index = b.int_const(32, i as u128);
            let value = b.array_get(array, index);
            assert_leaves(b, value, &shape[1..], expected, cursor, check_values);
        }
    }
}

/// Build both lanes from the same unguarded update chain. Snapshot provenance,
/// then guard the original operations, just as control-flow linearization does.
struct Program {
    ssa: HLSSA,
    types: crate::compiler::analysis::types::TypeInfo,
    parameters: Vec<ValueId>,
}

fn program(case: &Case, changed_then: bool, sparse: bool) -> Program {
    let mut ssa = HLSSA::with_main("differential_merge".into());
    let fid = ssa.get_unique_entrypoint_id();
    let mut f = ssa.take_function(fid);
    let entry = f.get_entry_id();
    let body = f.add_block();
    let merge = f.add_block();
    let leaf = Type::witness_of(if case.integer_leaves {
        Type::int(64)
    } else {
        Type::field()
    });
    let typ = array_type(&case.shape, &leaf);
    let mut param = |ty| {
        let id = ssa.fresh_value();
        f.get_block_mut(entry).push_parameter(id, ty);
        id
    };
    let base = param(typ.clone());
    let condition = param(Type::witness_of(Type::int(1)));
    let outer = param(Type::witness_of(Type::int(1)));
    let indices: Vec<_> = (0..3).map(|_| param(Type::int(case.bits))).collect();
    let check_values = param(Type::int(1));
    let expected: Vec<_> = (0..case.shape.iter().product())
        .map(|_| param(leaf.clone()))
        .collect();
    let (then_active, else_active, changed);
    {
        let mut fb = HLFunctionBuilder::new(&mut f, &mut ssa);
        let mut b = fb.test_block(entry);
        then_active = b.and(outer, condition);
        let negated = b.not(condition);
        else_active = b.and(outer, negated);
        b.set_terminator(Terminator::Jmp(body, vec![]));
        drop(b);
        let mut b = fb.test_block(body);
        let mut array = base;
        for write in &case.writes {
            array = update(
                &mut b,
                array,
                &case.shape,
                &write.keys,
                &indices,
                write,
                &leaf,
            );
        }
        changed = array;
        b.set_terminator(Terminator::Jmp(merge, vec![]));
        drop(b);
        fb.test_block(merge)
            .set_terminator(Terminator::Return(vec![]));
    }
    ssa.put_function(fid, f);
    let types = Types::new().run(&ssa, &FlowAnalysis::run(&ssa));
    let mut f = ssa.take_function(fid);
    let mut lowering = MergeLowering::new(&f, Some(types.get_function(fid)));
    if !sparse {
        lowering.sparse = None;
    }
    let active = if changed_then {
        then_active
    } else {
        else_active
    };
    let instructions = f.get_block_mut(body).take_instructions();
    f.get_block_mut(body).put_instructions(
        instructions
            .into_iter()
            .map(|op| {
                let (op, location) = op.take();
                OpCode::Guard {
                    condition: active,
                    inner: Box::new(op),
                }
                .locate(location)
            })
            .collect(),
    );
    {
        let mut fb = HLFunctionBuilder::new(&mut f, &mut ssa);
        let mut b = fb.test_block(merge);
        let (lhs, rhs) = if changed_then {
            (changed, base)
        } else {
            (base, changed)
        };
        // Require a real sparse match; silently testing the fallback twice proves nothing.
        let result = if sparse {
            lowering
                .sparse
                .as_ref()
                .unwrap()
                .try_emit(&mut b, condition, then_active, else_active, lhs, rhs, &typ)
                .expect("generated chain must use sparse merging")
        } else {
            lowering.emit(&mut b, condition, then_active, else_active, lhs, rhs, &typ)
        };
        // A merge inside an inactive outer branch is unobservable. Its enclosing
        // merge returns the original array, but all intermediate accesses must be safe.
        let result = emit_merge_select(&mut b, outer, result, base, &typ, &typ, &typ);
        assert_leaves(&mut b, result, &case.shape, &expected, &mut 0, check_values);
        b.set_terminator(Terminator::Return(vec![]));
    }
    ssa.put_function(fid, f);
    // Exercise removal of obsolete original writes as well as execution. No
    // test-inserted bounds asserts can accidentally mask a missing sparse check.
    let parameters = ssa
        .get_function(fid)
        .get_block(entry)
        .get_parameter_values()
        .copied()
        .collect();
    let flow = FlowAnalysis::run(&ssa);
    DCE::new(DceConfig::pre_r1c()).do_run(&mut ssa, &flow);
    InstructionLowering::guards().run(&mut ssa, &AnalysisStore::new());
    StripWitnessOf::new().do_run(&mut ssa);
    let types = Types::new().run(&ssa, &FlowAnalysis::run(&ssa));
    Program {
        ssa,
        types,
        parameters,
    }
}

fn input_array(shape: &[usize], values: &[u64], cursor: &mut usize) -> Value {
    if shape.is_empty() {
        let value = Value::Const(ark_bn254::Fr::from(values[*cursor]));
        *cursor += 1;
        value
    } else {
        Value::mk_array(
            (0..shape[0])
                .map(|_| input_array(&shape[1..], values, cursor))
                .collect(),
        )
    }
}

/// A flat-vector oracle, independent of SSA matching, guards, and merge emission.
fn expected(case: &Case, base: &[u64], indices: &[u32; 3], active: bool) -> Option<Vec<u64>> {
    let mut result = base.to_vec();
    if active {
        for write in &case.writes {
            let mut offset = 0;
            for (depth, &key) in write.keys.iter().enumerate() {
                let i = indices[key] as usize;
                if i >= case.shape[depth] {
                    return None;
                }
                offset += i * case.shape[depth + 1..].iter().product::<usize>();
            }
            let size = case.shape[write.keys.len()..].iter().product::<usize>();
            if write.keys.len() == case.shape.len() && !write.overwrite {
                result[offset] += write.delta;
            } else {
                result[offset..offset + size].fill(write.delta);
            }
        }
    }
    Some(result)
}

fn succeeds(
    program: &Program,
    case: &Case,
    base: &[u64],
    indices: &[u32; 3],
    condition: bool,
    outer: bool,
    expected: Option<&[u64]>,
) -> bool {
    let ssa = &program.ssa;
    let scalar = |n: u64| Value::Const(ark_bn254::Fr::from(n));
    let mut params = vec![
        input_array(&case.shape, base, &mut 0),
        scalar(condition as u64),
        scalar(outer as u64),
    ];
    params.extend(indices.iter().map(|&i| scalar(i as u64)));
    // Invalid inputs must fail on their own bounds checks. Disable the oracle's
    // output assertions there, so an incorrect output cannot masquerade as the
    // bounds failure we expected (and conceal a missing production assertion).
    params.push(scalar(expected.is_some() as u64));
    params.extend(expected.unwrap_or(base).iter().map(|&i| scalar(i)));
    // DCE can remove unused entry parameters. Keep arguments associated with their
    // original SSA ids instead of relying on the optimized parameter positions.
    let supplied: crate::collections::HashMap<_, _> =
        program.parameters.iter().copied().zip(params).collect();
    let function = ssa.get_function(ssa.get_unique_entrypoint_id());
    let params = function
        .get_block(function.get_entry_id())
        .get_parameters()
        .map(|(id, _)| supplied[id].clone())
        .collect();
    // The existing evaluator reports array bounds failures as indexing panics,
    // whereas explicit assertions return Err. Normalize only bounds panics;
    // all other panics are compiler/test bugs and must fail the property.
    match catch_unwind(AssertUnwindSafe(|| {
        SymbolicExecutor::new().run(
            ssa,
            &program.types,
            ssa.get_unique_entrypoint_id(),
            params,
            &mut R1CGen::new(FieldConfig::bn254()),
        )
    })) {
        Ok(result) => result.is_ok(),
        Err(payload) => {
            let text = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied());
            if text.is_some_and(|s| s.starts_with("index out of bounds:")) {
                false
            } else {
                resume_unwind(payload)
            }
        }
    }
}

fn check_case(case: &Case) -> proptest::test_runner::TestCaseResult {
    let base: Vec<_> = (0..case.shape.iter().product::<usize>())
        .map(|i| case.seed + i as u64)
        .collect();
    let valid = case
        .indices
        .map(|i| i % *case.shape.iter().min().unwrap() as u32);
    for changed_then in [false, true] {
        let sparse = program(case, changed_then, true);
        let fallback = program(case, changed_then, false);
        for indices in [valid, case.indices] {
            for condition in [false, true] {
                for outer in [false, true] {
                    let expected =
                        expected(case, &base, &indices, outer && condition == changed_then);
                    let a = succeeds(
                        &sparse,
                        case,
                        &base,
                        &indices,
                        condition,
                        outer,
                        expected.as_deref(),
                    );
                    let b = succeeds(
                        &fallback,
                        case,
                        &base,
                        &indices,
                        condition,
                        outer,
                        expected.as_deref(),
                    );
                    prop_assert_eq!(
                        a,
                        expected.is_some(),
                        "sparse: changed_then={}, condition={}, outer={}, indices={:?}",
                        changed_then,
                        condition,
                        outer,
                        indices
                    );
                    prop_assert_eq!(
                        b,
                        expected.is_some(),
                        "fallback: changed_then={}, condition={}, outer={}, indices={:?}",
                        changed_then,
                        condition,
                        outer,
                        indices
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn bounds_failure_survives_dead_write_elimination() {
    check_case(&Case {
        shape: vec![2],
        writes: vec![Write {
            keys: vec![0],
            overwrite: true,
            delta: 7,
        }],
        indices: [2, 0, 0],
        bits: 32,
        integer_leaves: false,
        seed: 1,
    })
    .unwrap();
}

#[test]
fn runtime_aliases_preserve_nested_write_order() {
    check_case(&Case {
        shape: vec![2, 3],
        writes: vec![
            Write {
                keys: vec![0, 1],
                overwrite: true,
                delta: 3,
            },
            Write {
                keys: vec![0, 2],
                overwrite: true,
                delta: 7,
            },
            Write {
                keys: vec![0, 1],
                overwrite: false,
                delta: 2,
            },
        ],
        // Different SSA index parameters identify the same cell at runtime.
        indices: [1, 2, 2],
        bits: 8,
        integer_leaves: true,
        seed: 11,
    })
    .unwrap();
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]
    #[test]
    fn sparse_and_fallback_match_array_semantics(case in cases()) {
        check_case(&case)?;
    }
}
