use crate::{
    collections::HashMap,
    compiler::{
        passes::{
            shared::limbs::{
                single_cell_product_fits, widest_injective_int_bits, witness_limb_bits,
            },
            wide_witness_ints::{
                limb_config::is_evaluable,
                rewriter::product::{
                    full_bounds,
                    plan::{Step, schoolbook_fits},
                    plan_product,
                },
                test_utils::{
                    bn254, emitted, fields, guard_every_instruction, limbed_widths,
                    operation_with_in, run_pass,
                },
            },
        },
        ssa::{
            Instruction, Located, ValueId,
            hlssa::{BinaryArithOpKind, CastTarget, Constant, HLSSA, OpCode, Type},
        },
    },
};

use super::{super::super::types::limb_widths, ProductPlan};
use crate::compiler::Field;
use mavros_artifacts::FieldConfig;
use mavros_int_semantics::{IntBits, int_bits::HOST_LIMB_BITS};
use num_bigint::BigUint;
use num_traits::{One, Zero};

/// Re-derive what `plan` claims from its steps alone and hold it to the soundness argument
/// [`plan_product`] states without reading how the plan was built.
fn assert_plan_is_sound(
    plan: &ProductPlan,
    lhs: &[BigUint],
    rhs: &[BigUint],
    addend: &[BigUint],
    widths: &[usize],
    injective: usize,
) {
    let limit = BigUint::one() << injective;
    let count = widths.len();
    let mut added = crate::collections::HashSet::default();
    let mut added_addends = crate::collections::HashSet::default();
    let mut incoming: Vec<BigUint> = Vec::new();

    for (column, (steps, &width)) in plan.columns.iter().zip(widths).enumerate() {
        let top = column + 1 == count;
        let mut accumulated = BigUint::zero();
        let mut carries = incoming.iter();
        let mut outgoing = Vec::new();
        for step in steps {
            match *step {
                Step::Product {
                    lhs: left,
                    rhs: right,
                } => {
                    assert_eq!(
                        left + right,
                        column,
                        "a partial product in the wrong column"
                    );
                    assert!(added.insert((left, right)), "a partial product added twice");
                    accumulated += &lhs[left] * &rhs[right];
                }
                Step::Addend(index) => {
                    assert_eq!(index, column, "an addend limb in the wrong column");
                    assert!(added_addends.insert(index), "an addend limb added twice");
                    accumulated += &addend[index];
                }
                Step::Carry => {
                    accumulated += carries.next().expect("a carry the column below made");
                }
                Step::Reduce { carry_bits: None } => {
                    assert!(top, "only the top column reduces without a carry");
                    accumulated = (BigUint::one() << width) - 1u8;
                }
                Step::Reduce {
                    carry_bits: Some(carry_bits),
                } => {
                    assert!(!top, "the top column carries nothing out");
                    assert!(
                        accumulated < BigUint::one() << (width + carry_bits),
                        "column {column}'s honest carry does not fit its range check"
                    );
                    assert!(
                        width + carry_bits <= injective,
                        "column {column}'s split does not stay below the modulus"
                    );
                    outgoing.push((BigUint::one() << carry_bits) - 1u8);
                    accumulated = (BigUint::one() << width) - 1u8;
                }
            }
            assert!(accumulated < limit, "column {column} passes the modulus");
            assert!(
                accumulated.bits() as usize <= plan.hint_bits,
                "column {column} passes the hint width"
            );
        }
        assert!(carries.next().is_none(), "column {column} drops a carry");
        assert!(
            accumulated < BigUint::one() << width,
            "column {column} ends outside its limb"
        );
        incoming = outgoing;
    }
    assert!(incoming.is_empty(), "the top column carries out");

    // An evaluated product holds every column, the ones past the answer included, to its
    // integer sum, which has to stay below the modulus for that to mean anything.
    if plan.evaluated.is_some() {
        assert!(
            plan.overflow.is_empty(),
            "an evaluated product with overflow terms"
        );
        for column in 0..2 * count - 1 {
            let sum: BigUint = (0..count)
                .filter(|left| *left <= column && column - left < count)
                .map(|left| &lhs[left] * &rhs[column - left])
                .sum();
            assert!(
                sum < limit,
                "column {column} of an evaluated product passes the modulus"
            );
        }
        return;
    }

    let mut covered = crate::collections::HashSet::default();
    for (left, group) in &plan.overflow {
        let sum: BigUint = group.iter().map(|right| &rhs[*right]).sum();
        assert!(
            &lhs[*left] * sum < limit,
            "an overflow term passes the modulus"
        );
        for right in group {
            assert!(left + right >= count, "an overflow term inside the answer");
            assert!(
                covered.insert((*left, *right)),
                "an overflow term checked twice"
            );
        }
    }

    for (index, bound) in addend.iter().enumerate() {
        assert!(
            bound.is_zero() || added_addends.contains(&index),
            "addend limb {index} is never added"
        );
    }

    for left in 0..count {
        for right in 0..count {
            if (&lhs[left] * &rhs[right]).is_zero() {
                continue;
            }
            let (set, what) = if left + right < count {
                (&added, "added to its column")
            } else {
                (&covered, "held to zero")
            };
            assert!(
                set.contains(&(left, right)),
                "the partial product ({left}, {right}) is never {what}"
            );
        }
    }
}

/// The plan is sound on any field that can hold it. bn254 can at every width and limb tried,
/// and goldilocks at every width tried on the limb `witness_limb_bits` gives it.
///
/// The `2h + 2`, `2h + 1` and `2h` injective widths stand for fields this compiler cannot be
/// configured for. They exercise reductions part-way along a column, which neither configured
/// field needs, and the refusal of a field too narrow for one product plus its headroom.
#[test]
fn the_schoolbook_plan_is_sound_on_any_field_that_holds_it() {
    let bn254 = widest_injective_int_bits(bn254());
    let configured: Vec<(usize, usize)> = fields()
        .into_iter()
        .map(|field| (widest_injective_int_bits(field), witness_limb_bits(field)))
        .collect();
    let goldilocks = widest_injective_int_bits(FieldConfig::goldilocks());
    let sparse = |widths: &[usize]| -> Vec<BigUint> {
        full_bounds(widths)
            .into_iter()
            .enumerate()
            .map(|(index, bound)| match index % 3 {
                0 => bound,
                1 => BigUint::zero(),
                _ => BigUint::from(5u8),
            })
            .collect()
    };
    for limb_bits in [16usize, 32, 64] {
        for bits in [64usize, 100, 127, 200, 253, 254, 320, 1000] {
            let widths = limb_widths(bits, limb_bits);
            let full = full_bounds(&widths);
            for injective in [
                bn254,
                goldilocks,
                2 * limb_bits + 2,
                2 * limb_bits + 1,
                2 * limb_bits,
            ] {
                for rhs in [full.clone(), sparse(&widths)] {
                    for addend in [Vec::new(), full.clone(), sparse(&widths)] {
                        for evaluate in [false, true] {
                            if evaluate && !is_evaluable(bits, limb_bits, injective) {
                                continue;
                            }
                            let planned =
                                plan_product(&full, &rhs, &addend, &widths, injective, evaluate);
                            match planned {
                                Some(plan) => assert_plan_is_sound(
                                    &plan, &full, &rhs, &addend, &widths, injective,
                                ),
                                None => {
                                    assert_ne!(injective, bn254, "bn254 refuses int{bits}");
                                    assert!(
                                        !configured.contains(&(injective, limb_bits)),
                                        "a configured field refuses int{bits} on its limb"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// On bn254 a whole column fits an element, so each reduces once, at its end, and each left
/// limb's overflow terms are one constraint. Evaluated, every column fits too, so the product
/// is evaluated and has no overflow terms at all.
#[test]
fn bn254_reduces_each_column_once() {
    let widths = limb_widths(320, witness_limb_bits(bn254()));
    let full = full_bounds(&widths);
    let injective = widest_injective_int_bits(bn254());

    for bits in [127usize, 254, 320, 1000, 16384] {
        let widths = limb_widths(bits, witness_limb_bits(bn254()));
        let full = full_bounds(&widths);
        let plan = plan_product(&full, &full, &[], &widths, injective, true)
            .unwrap_or_else(|| panic!("bn254 holds an int{bits} product"));
        assert!(plan.evaluated.is_some(), "int{bits} is evaluated on bn254");
        assert!(plan.overflow.is_empty(), "int{bits} has no overflow terms");
    }

    let plan = plan_product(&full, &full, &[], &widths, injective, false)
        .expect("bn254 holds an int320 product");

    for (column, steps) in plan.columns.iter().enumerate() {
        let reductions = steps
            .iter()
            .filter(|step| matches!(step, Step::Reduce { .. }))
            .count();
        assert_eq!(reductions, 1, "column {column}");
        assert!(matches!(steps.last(), Some(Step::Reduce { .. })));
    }
    assert_eq!(
        plan.overflow.len(),
        widths.len() - 1,
        "one overflow constraint per left limb above the lowest"
    );
}

/// A field one partial product fills refuses; a field that holds one with room to spare but not
/// a whole column reduces part-way along it.
#[test]
fn a_narrow_field_reduces_more_often_or_refuses() {
    let widths = limb_widths(320, 64);
    let full = full_bounds(&widths);

    // A 32-bit limb on goldilocks: `(2^32 - 1)^2` alone has 64 bits against an injective 63.
    let goldilocks = limb_widths(320, 32);
    assert!(
        plan_product(
            &full_bounds(&goldilocks),
            &full_bounds(&goldilocks),
            &[],
            &goldilocks,
            63,
            false
        )
        .is_none()
    );

    // Five products to a column do not fit 130 bits, so summing reduces mid-column.
    let injective = 2 * 64 + 2;
    let plan = plan_product(&full, &full, &[], &widths, injective, false)
        .expect("two products and a carry fit 130 bits");
    assert_plan_is_sound(&plan, &full, &full, &[], &widths, injective);
    assert!(
        plan.columns.iter().any(|steps| {
            steps
                .iter()
                .filter(|step| matches!(step, Step::Reduce { .. }))
                .count()
                > 1
        }),
        "a column of five products reduces before its end"
    );
}

/// bn254 plans a schoolbook product, and the `q·d + r` of a division, at every width class it
/// has, so neither ever reaches [`unsupported_on_this_field`].
#[test]
fn bn254_holds_a_product_at_every_width() {
    for bits in [1usize, 127, 128, 129, 253, 254, 1000, 16383, 16384] {
        for with_addend in [false, true] {
            assert!(
                schoolbook_fits(bn254(), bits, with_addend),
                "int{bits}, addend: {with_addend}"
            );
        }
    }
}

/// `main(lhs: WitnessOf<int(bits)>, rhs: rhs_type) -> WitnessOf<int(bits)> { lhs * rhs }` over
/// `field`.
fn product_with(field: FieldConfig, bits: usize, rhs_type: Type) -> HLSSA {
    operation_with_in(field, BinaryArithOpKind::UMul, bits, rhs_type)
}

/// `main(lhs, rhs: WitnessOf<int(bits)>) -> WitnessOf<int(bits)> { lhs * rhs }`.
fn product(bits: usize) -> HLSSA {
    product_in(bn254(), bits)
}

/// [`product`] over `field`.
fn product_in(field: FieldConfig, bits: usize) -> HLSSA {
    product_with(field, bits, Type::witness_of(Type::int(bits)))
}

/// The range check on each witness a schoolbook writes after its operands, in the order they are
/// written, [`None`] for one nothing checks.
fn planned_written_checks(
    plan: &ProductPlan,
    widths: &[usize],
    pinned: bool,
) -> Vec<Option<usize>> {
    if let Some(blocks) = &plan.evaluated {
        let count = widths.len();
        let mut expected = Vec::new();
        for (column, &width) in widths.iter().enumerate() {
            if !pinned {
                expected.push(Some(width));
            }
            if column + 1 < count {
                expected.push(
                    blocks
                        .iter()
                        .find(|block| block.last == column)
                        .and_then(|block| block.carry_bits),
                );
            }
        }
        expected
    } else {
        // Summed, the only witnesses are the carries.
        plan.columns
            .iter()
            .flatten()
            .filter_map(|step| match step {
                Step::Reduce {
                    carry_bits: Some(bits),
                } => Some(*bits),
                _ => None,
            })
            .map(Some)
            .collect()
    }
}

/// Every carry a product witnesses is range-checked at the width its plan gives, and every limb
/// of the answer at its width, however the columns are formed.
///
/// Evaluated (both operands witnessed), each limb and each carry is a separate witness, the
/// carries inside a block pinned only by the `2k - 1` evaluations, and nothing is held to zero.
/// Summed (a pure right operand), the columns are linear and every left limb above the lowest
/// is held to a zero overflow.
///
/// Structural for the reason [`every_carry_is_a_bit_and_every_limb_of_the_chain_is_bounded`]
/// gives: perturbing a limb or a boundary carry trips its check, an interior carry an evaluation.
#[test]
fn every_carry_and_every_limb_of_a_product_is_bounded() {
    for field in fields() {
        let id = field.id();
        let h = witness_limb_bits(field);
        for bits in limbed_widths(field) {
            let widths = limb_widths(bits, h);
            let k = widths.len();
            let full = full_bounds(&widths);
            for (what, rhs_type, evaluated) in [
                ("evaluated", Type::witness_of(Type::int(bits)), true),
                ("summed", Type::int(bits), false),
            ] {
                let plan = plan_product(
                    &full,
                    &full,
                    &[],
                    &widths,
                    widest_injective_int_bits(field),
                    evaluated,
                )
                .expect("the field holds the product");
                assert_eq!(plan.evaluated.is_some(), evaluated);
                let expected = planned_written_checks(&plan, &widths, false);

                let mut ssa = product_with(field, bits, rhs_type);
                run_pass(&mut ssa);
                let ops = emitted(&ssa);

                let checked = |value: ValueId| {
                    ops.iter().find_map(|op| match op {
                        OpCode::Rangecheck {
                            value: checked,
                            max_bits,
                        } if *checked == value => Some(*max_bits),
                        _ => None,
                    })
                };
                let written: Vec<ValueId> = ops
                    .iter()
                    .filter_map(|op| match op {
                        OpCode::WriteWitness {
                            result: Some(result),
                            ..
                        } => Some(*result),
                        _ => None,
                    })
                    .collect();
                assert_eq!(
                    written
                        .iter()
                        .map(|value| checked(*value))
                        .collect::<Vec<_>>(),
                    expected,
                    "{id:?} int{bits} {what}: each witness checked as planned"
                );

                let limbs: Vec<usize> = ops
                    .iter()
                    .filter_map(|op| match op {
                        OpCode::Rangecheck { value, max_bits } if !written.contains(value) => {
                            Some(*max_bits)
                        }
                        _ => None,
                    })
                    .collect();
                assert_eq!(
                    limbs,
                    if evaluated {
                        Vec::new()
                    } else {
                        widths.clone()
                    },
                    "{id:?} int{bits} {what}: a summed answer's limbs are bounded at their widths"
                );

                let constraints = ops
                    .iter()
                    .filter(|op| matches!(op, OpCode::Constrain { .. }))
                    .count();
                assert_eq!(
                    constraints,
                    if evaluated { 2 * k - 1 } else { k - 1 },
                    "{id:?} int{bits} {what}: the evaluations, or one overflow constraint per left limb"
                );
            }
        }
    }
}

/// The evaluations are at `2k - 1` distinct points, enough to make the degree-`2k - 2` identity
/// one of polynomials, and each reads every limb of both operands, the answer and the carries.
///
/// Structural, because an honest witness satisfies the identity at any set of points. With one
/// point short, or two the same, a prover could move value into the columns past the answer,
/// where nothing else would see it.
#[test]
fn an_evaluated_product_is_pinned_at_distinct_points() {
    for field in fields() {
        let id = field.id();
        let bits = limbed_widths(field)[1];
        let k = limb_widths(bits, witness_limb_bits(field)).len();
        let mut ssa = product_in(field, bits);
        run_pass(&mut ssa);
        let ops = emitted(&ssa);

        let definitions: HashMap<ValueId, OpCode> = ops
            .iter()
            .flat_map(|op| op.get_results().map(move |result| (*result, op.clone())))
            .collect();
        // The constant each term of a linear combination is scaled by, `1` where it is not.
        let terms = |value: ValueId| -> Vec<(ValueId, Field)> {
            let mut out = Vec::new();
            let mut stack = vec![(value, field.one())];
            while let Some((value, scale)) = stack.pop() {
                match definitions.get(&value) {
                    Some(OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::UAdd,
                        lhs,
                        rhs,
                        ..
                    }) => stack.extend([(*lhs, scale), (*rhs, scale)]),
                    Some(OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::UMul,
                        lhs,
                        rhs,
                        ..
                    }) if matches!(ssa.get_const(*rhs).as_deref(), Some(Constant::Field(_))) => {
                        let Some(Constant::Field(constant)) =
                            ssa.get_const(*rhs).as_deref().cloned()
                        else {
                            unreachable!()
                        };
                        stack.push((*lhs, scale * constant));
                    }
                    _ => out.push((value, scale)),
                }
            }
            out
        };

        let evaluations: Vec<(Vec<(ValueId, Field)>, Vec<(ValueId, Field)>)> = ops
            .iter()
            .filter_map(|op| match op {
                OpCode::Constrain { a, b, c } => {
                    Some((terms(*a), terms(*b).into_iter().chain(terms(*c)).collect()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            evaluations.len(),
            2 * k - 1,
            "{id:?}: one evaluation per point"
        );

        // The point is the scale of the second coefficient, and `0` where only the first is
        // read.
        let mut points: Vec<Field> = evaluations
            .iter()
            .map(|(a, _)| {
                if a.len() == 1 {
                    field.zero()
                } else {
                    a.iter()
                        .map(|(_, scale)| *scale)
                        .find(|scale| *scale != field.one())
                        .unwrap_or(field.one())
                }
            })
            .collect();
        points.sort_by_key(|point| format!("{point:?}"));
        points.dedup();
        assert_eq!(points.len(), 2 * k - 1, "{id:?}: the points are distinct");
        for (index, (a, rest)) in evaluations.iter().enumerate() {
            if index == 0 {
                continue;
            }
            assert_eq!(
                a.len(),
                k,
                "{id:?}: evaluation {index} reads every left limb"
            );
            assert_eq!(
                rest.len(),
                2 * k + (k - 1),
                "{id:?}: evaluation {index} reads every right limb, answer limb and carry"
            );
        }
    }
}

/// A guarded product checks nothing where the guard is off, and its answer is zero there.
///
/// Evaluated, the product itself is made zero there instead, by scaling the left limbs: the
/// identity then holds of the operands the branch not taken left behind, and its range checks
/// are guarded all the same.
#[test]
fn a_guarded_product_is_checked_only_under_its_guard() {
    for field in fields() {
        let id = field.id();
        let bits = limbed_widths(field)[1];
        let mut ssa = product_in(field, bits);
        let condition = guard_every_instruction(&mut ssa);
        run_pass(&mut ssa);
        let ops = emitted(&ssa);

        assert!(
            !ops.iter().any(|op| matches!(op, OpCode::Rangecheck { .. })),
            "{id:?}: no range check is unguarded"
        );
        let widths = limb_widths(bits, witness_limb_bits(field));
        let k = widths.len();
        let full = full_bounds(&widths);
        let plan = plan_product(
            &full,
            &full,
            &[],
            &widths,
            widest_injective_int_bits(field),
            true,
        )
        .expect("the field holds the product");
        let planned = planned_written_checks(&plan, &widths, false)
            .into_iter()
            .flatten()
            .count();
        let guarded = ops
            .iter()
            .filter(|op| {
                matches!(op, OpCode::Guard { condition: c, inner }
                if *c == condition && matches!(inner.as_ref(), OpCode::Rangecheck { .. }))
            })
            .count();
        assert_eq!(
            guarded, planned,
            "{id:?}: every limb and every boundary carry, under the guard"
        );

        let flag: Vec<ValueId> = ops
            .iter()
            .filter_map(|op| match op {
                OpCode::Cast {
                    result,
                    value,
                    target: CastTarget::Field,
                } if *value == condition => Some(*result),
                _ => None,
            })
            .collect();
        let scaled = ops
            .iter()
            .filter(|op| {
                matches!(op, OpCode::BinaryArithOp { kind: BinaryArithOpKind::UMul, lhs, .. }
                if flag.contains(lhs))
            })
            .count();
        assert_eq!(
            scaled, k,
            "{id:?}: every left limb is scaled by the guard, so the product is zero where it is off"
        );
        let evaluations = ops
            .iter()
            .filter(|op| matches!(op, OpCode::Constrain { .. }))
            .count();
        assert_eq!(
            evaluations,
            2 * k - 1,
            "{id:?}: the evaluations need no guard of their own"
        );

        let selected = ops
            .iter()
            .filter(|op| matches!(op, OpCode::Select { cond, .. } if *cond == condition))
            .count();
        assert_eq!(
            selected, k,
            "{id:?}: each limb of the answer is zero where the guard is off"
        );
    }
}

/// A product the single cell cannot hold goes through the schoolbook, on a decomposition of
/// each operand while they still have an element; one it can hold, and the double lane's width,
/// are left alone.
#[test]
fn a_product_the_single_cell_cannot_hold_takes_the_schoolbook() {
    let field = bn254();
    let widest = widest_injective_int_bits(field) / 2;
    assert!(!single_cell_product_fits(field, widest + 1));

    let mut ssa = product(widest + 1);
    let main = ssa.get_unique_entrypoint_id();
    let operands: Vec<ValueId> = ssa
        .get_function(main)
        .get_entry()
        .get_parameters()
        .map(|(value, _)| *value)
        .collect();
    run_pass(&mut ssa);
    let ops = emitted(&ssa);
    assert!(
        !ops.iter().any(|op| matches!(
            op,
            OpCode::BinaryArithOp { lhs, rhs, .. }
                if operands.contains(lhs) && operands.contains(rhs)
        )),
        "the single-cell product is gone"
    );
    let tied = ops
        .iter()
        .filter(|op| matches!(op, OpCode::Constrain { .. }))
        .count();
    assert_eq!(tied, 2 + 3, "two decompositions and three evaluations");

    for bits in [widest, 2 * HOST_LIMB_BITS] {
        let mut ssa = product(bits);
        let before = format!("{:?}", emitted(&ssa));
        run_pass(&mut ssa);
        assert_eq!(format!("{:?}", emitted(&ssa)), before, "int{bits}");
    }
}

/// A constant factor is cut at compile time, so the limbs it does not reach cost nothing: a
/// product by a one-limb constant has no partial product above the diagonal and nothing to hold
/// to zero.
#[test]
fn a_constant_factor_drops_the_limbs_it_does_not_reach() {
    let bits = 320usize;
    let mut ssa = product(bits);
    let main = ssa.get_unique_entrypoint_id();
    let five = ssa.add_const(Constant::Int(IntBits::from_u128(bits, 5)));
    {
        let entry = ssa.get_function_mut(main).get_entry_mut();
        let instructions: Vec<_> = entry
            .take_instructions()
            .into_iter()
            .map(|op| {
                let OpCode::BinaryArithOp {
                    kind, result, lhs, ..
                } = op.as_ref().clone()
                else {
                    panic!("the program is one product")
                };
                let rewritten = OpCode::BinaryArithOp {
                    kind,
                    result,
                    lhs,
                    rhs: five,
                };
                Located::new(rewritten, op.location().clone())
            })
            .collect();
        entry.put_instructions(instructions);
    }
    run_pass(&mut ssa);
    let ops = emitted(&ssa);

    let k = limb_widths(bits, witness_limb_bits(bn254())).len();
    let products = ops
        .iter()
        .filter(|op| {
            matches!(
                op,
                OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::UMul,
                    ..
                }
            )
        })
        .count();
    assert_eq!(
        ops.iter()
            .filter(|op| matches!(op, OpCode::Constrain { .. }))
            .count(),
        0,
        "no partial product reaches past the answer"
    );
    // Per column: the partial product and its hint, then the carry's place value.
    assert_eq!(products, 2 * k + (k - 1), "one partial product per column");
}

/// The range check on each witness the division's schoolbook writes, in order, at full operand ranges.
pub(crate) fn division_carries(
    field: FieldConfig,
    widths: &[usize],
    evaluate: bool,
) -> Vec<Option<usize>> {
    let full = full_bounds(widths);
    let plan = plan_product(
        &full,
        &full,
        &full,
        widths,
        widest_injective_int_bits(field),
        evaluate,
    )
    .expect("the field holds the division");
    assert_eq!(plan.evaluated.is_some(), evaluate);
    planned_written_checks(&plan, widths, true)
}
