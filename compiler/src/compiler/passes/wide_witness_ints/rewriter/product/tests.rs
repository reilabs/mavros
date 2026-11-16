use crate::{
    collections::HashMap,
    compiler::{
        passes::{
            shared::limbs::{
                single_cell_product_fits, widest_injective_int_bits, witness_limb_bits,
            },
            wide_witness_ints::{
                rewriter::product::{
                    Polarity, Term, full_bounds,
                    plan::{Answer, Factor, Limb, build_f_polynomial},
                },
                test_utils::{
                    bn254, emitted, fields, guard_every_instruction, limbed_widths,
                    operation_with_in, run_pass,
                },
            },
        },
        ssa::{
            Instruction, ValueId,
            hlssa::{BinaryArithOpKind, CastTarget, Constant, HLSSA, OpCode, Type},
        },
    },
};

use super::super::super::types::limb_widths;
use crate::compiler::Field;
use mavros_artifacts::FieldConfig;
use mavros_int_semantics::int_bits::HOST_LIMB_BITS;

/// A checked schoolbook's terms: `factor · factor`, plus `factor` where `addend`.
fn schoolbook_terms(factor: &Factor, addend: bool) -> Vec<Term<'_>> {
    let mut terms = vec![Term {
        polarity: Polarity::Positive,
        lhs: factor,
        rhs: Some(factor),
        gated: true,
    }];
    if addend {
        terms.push(Term {
            polarity: Polarity::Positive,
            lhs: factor,
            rhs: None,
            gated: false,
        });
    }
    terms
}

/// bn254 plans a schoolbook product, and the `q·d + r` of a division, at every width class it
/// has, so neither ever reaches [`unsupported_on_this_field`].
#[test]
fn bn254_holds_a_product_at_every_width() {
    let injective = widest_injective_int_bits(bn254());
    for bits in [127usize, 128, 129, 253, 254, 320, 1000, 16383, 16384] {
        let widths = limb_widths(bits, witness_limb_bits(bn254()));
        assert!(
            planned_written_checks(&widths, injective, false, false).is_some(),
            "int{bits}"
        );
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

/// The range check on each witness a checked schoolbook at limb `widths` writes after its
/// operands, in the order they are written, [`None`] for one nothing checks; or [`None`] if a
/// field whose widest injective width is `injective` cannot hold it. Its operands are full-range
/// witnesses, the left one also an addend where `addend`.
fn planned_written_checks(
    widths: &[usize],
    injective: usize,
    addend: bool,
    pinned: bool,
) -> Option<Vec<Option<usize>>> {
    // The planner reads only the factor's shape.
    let factor = Factor {
        limbs: widths.iter().map(|_| Limb::Value(ValueId(0))).collect(),
        is_witnessed: true,
        bounds: full_bounds(widths),
        L: None,
    };
    let identity = build_f_polynomial(
        widths,
        injective,
        schoolbook_terms(&factor, addend),
        Some(Answer::Unknown(widths)),
    )?;
    let count = widths.len();
    let mut expected = Vec::new();
    for (column, &width) in widths.iter().enumerate() {
        if !pinned {
            expected.push(Some(width));
        }
        if column + 1 < count {
            expected.push(
                identity
                    .blocks
                    .iter()
                    .find(|block| block.last == column)
                    .and_then(|block| block.carry_bits),
            );
        }
    }
    Some(expected)
}

/// Every carry a product witnesses is range-checked at the width its identity gives, and every
/// limb of the answer at its width, whether the right operand is witnessed or pure.
///
/// Each limb and each carry is a separate witness, the carries inside a block pinned only by the
/// `2k - 1` evaluations, and nothing is held to zero.
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
            for (what, rhs_type) in [
                ("witnessed", Type::witness_of(Type::int(bits))),
                ("pure", Type::int(bits)),
            ] {
                let expected =
                    planned_written_checks(&widths, widest_injective_int_bits(field), false, false)
                        .expect("the field holds the identity");

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
                assert!(
                    limbs.is_empty(),
                    "{id:?} int{bits} {what}: every check is on a written witness"
                );

                let constraints = ops
                    .iter()
                    .filter(|op| matches!(op, OpCode::Constrain { .. }))
                    .count();
                assert_eq!(
                    constraints,
                    2 * k - 1,
                    "{id:?} int{bits} {what}: the evaluations"
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
        let planned =
            planned_written_checks(&widths, widest_injective_int_bits(field), false, false)
                .expect("the field holds the product")
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

/// The range check on each witness the division's schoolbook writes, in order, at full operand ranges.
pub(crate) fn division_carries(field: FieldConfig, widths: &[usize]) -> Vec<Option<usize>> {
    planned_written_checks(widths, widest_injective_int_bits(field), true, true)
        .expect("the field holds the division's identity")
}
