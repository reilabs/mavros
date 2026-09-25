//! The oracle's test suite: what the pipeline must agree with the model about.

mod harness;

use harness::{BinaryOpOracle, Compiled, Verdict, bytecode_listing, input_block, main_program};
use mavros_artifacts::{Field, FieldConfig, InputValueOrdered};
use mavros_compiler::{
    compiler::{
        passes::shared::limbs::witness_limb_bits,
        ssa::{
            SourceLocation, SourcePosition, Terminator, ValueId,
            hlssa::{
                BinaryArithOpKind, Blob, CastTarget, CmpKind, Constant, HLSSA, OpCode,
                SequenceTargetType, SliceOpDir, Type,
                builder::{HLBlockEmitter, HLEmitter as _, HLSSABuilder},
            },
        },
    },
    driver::{Driver, Error as DriverError},
};
use mavros_int_semantics::{IntBits, IntOp, corners, int_bits::HOST_LIMB_BITS};
use num_bigint::BigUint;

// UTILITIES
// ================================================================================================

/// Every operand pair worth trying for one operation at one width: the corner values against each
/// other, with a shift's right operand drawn from the amounts instead.
fn corner_pairs(op: IntOp, bits: usize) -> Vec<(IntBits, IntBits)> {
    let rhs = if op.is_shift() {
        corners::shift_amounts(bits, bits)
    } else {
        corners::values(bits)
    };
    corners::values(bits)
        .into_iter()
        .flat_map(|a| {
            rhs.iter()
                .map(move |&b| (IntBits::from_u128(bits, a), IntBits::from_u128(bits, b)))
        })
        .collect()
}

/// [`corner_pairs`] from the wide corner set, which adds the limb boundaries.
fn wide_corner_pairs(op: IntOp, bits: usize) -> Vec<(IntBits, IntBits)> {
    let (lhs, rhs) = corners::wide_operands(op, bits);
    lhs.iter()
        .flat_map(|a| rhs.iter().map(move |b| (a.clone(), b.clone())))
        .collect()
}

/// Sweeps one operation at one width, reporting the first disagreement.
fn sweep(kind: BinaryArithOpKind, bits: usize) {
    sweep_over(kind, bits, corner_pairs(IntOp::from(kind), bits));
}

/// [`sweep`] over the given operand pairs.
fn sweep_over(kind: BinaryArithOpKind, bits: usize, pairs: Vec<(IntBits, IntBits)>) {
    let op = IntOp::from(kind);
    let oracle = BinaryOpOracle::new(kind, bits, bits)
        .unwrap_or_else(|e| panic!("{kind:?} at {bits} bits failed to compile: {e}"));
    for (lhs, rhs) in &pairs {
        if let Err(disagreement) = oracle.check(lhs, rhs) {
            panic!("{disagreement}");
        }
    }
    let accepted = pairs
        .iter()
        .find(|(lhs, rhs)| mavros_int_semantics::eval(op, lhs, rhs).value().is_some())
        .unwrap_or_else(|| panic!("{kind:?} at {bits} bits: the model accepts no corner pair"));
    if let Err(unconstrained) = oracle.check_a_wrong_answer_is_refused(&accepted.0, &accepted.1) {
        panic!("{unconstrained}");
    }
}

/// The operations that reach the pipeline at any width.
fn non_shift_operations() -> impl Iterator<Item = BinaryArithOpKind> {
    BinaryArithOpKind::ALL
        .into_iter()
        .filter(|kind| !IntOp::from(*kind).is_shift())
}

/// Adds one to each column of an accepted witness in turn, requiring each to break a constraint.
///
/// `inputs` is the honest input block: the program's parameters followed by its declared returns.
/// A column no constraint mentions survives the perturbation untouched, and that is the reading.
fn assert_every_column_is_pinned(what: &str, compiled: &Compiled, inputs: &[&IntBits]) {
    let verdict = compiled.run(&input_block(inputs));
    let witness = verdict
        .witness()
        .unwrap_or_else(|| panic!("{what}: {verdict:?}"))
        .to_vec();

    for column in 0..witness.len() {
        let mut perturbed = witness.clone();
        perturbed[column] += Field::from(1u64);
        assert!(
            compiled.first_unsatisfied_constraint(&perturbed).is_some(),
            "{what}: column {column} of {} is unconstrained — \
             adding one to it left every constraint satisfied",
            witness.len()
        );
    }
}

// FUNCTIONAL TESTS
// ================================================================================================

/// The headline claim, at the width every backend agrees is ordinary: for every operation, over the
/// whole corner matrix, the model, the constraint system and the VM say the same thing.
#[test]
fn every_operation_agrees_with_the_model_at_a_byte() {
    for kind in BinaryArithOpKind::ALL {
        sweep(kind, 8);
    }
}

/// The same claim at widths **no Noir program can name**.
#[test]
fn every_operation_agrees_with_the_model_at_a_width_noir_cannot_express() {
    for kind in non_shift_operations() {
        sweep(kind, 5);
    }
}

/// A width above the host half-word and below the host word, still not a power of two.
#[test]
fn arithmetic_agrees_at_thirty_seven_bits() {
    for kind in [
        BinaryArithOpKind::UAdd,
        BinaryArithOpKind::USub,
        BinaryArithOpKind::UMul,
        BinaryArithOpKind::UDiv,
    ] {
        sweep(kind, 37);
    }
}

/// The top of the VM's two-cell lane, and the one width past a host word where a witnessed product
/// has a lowering: `lower_unsigned_mul` splits an `int128` into two limbs and multiplies them
/// schoolbook, where every narrower product is one field multiplication.
#[test]
fn arithmetic_agrees_at_the_top_of_the_two_cell_lane() {
    for kind in [BinaryArithOpKind::UAdd, BinaryArithOpKind::UMul] {
        sweep(kind, 128);
    }
}

/// The ablation, ensuring that the tests above are actually catching things.
///
/// This ensures that the test can actually go red, by driving the same program to each case that we
/// care about:
///
/// - The model's value is accepted.
/// - A value the model did not produce is refused at the return check.
/// - Operands the model rejects are refused somewhere else.
#[test]
fn the_oracle_distinguishes_a_wrong_answer_from_a_refusal() {
    let byte = |v| IntBits::from_u128(8, v);
    let oracle = BinaryOpOracle::new(BinaryArithOpKind::UAdd, 8, 8).unwrap();

    let honest = oracle
        .compiled()
        .run(&input_block(&[&byte(1), &byte(2), &byte(3)]));
    assert!(
        honest.is_accepted(),
        "1 + 2 == 3 should be accepted: {honest:?}"
    );

    let wrong = oracle
        .compiled()
        .run(&input_block(&[&byte(1), &byte(2), &byte(4)]));
    assert!(
        matches!(
            wrong,
            Verdict::Trapped {
                in_return_check: true,
                ..
            }
        ),
        "1 + 2 == 4 should fail the return check: {wrong:?}"
    );

    // 200 + 100 overflows a byte, which Noir rejects. The residue is declared as the return, so an
    // acceptance here would mean the overflow guard never fired.
    let overflow = oracle
        .compiled()
        .run(&input_block(&[&byte(200), &byte(100), &byte(44)]));
    assert!(
        overflow.is_refusal(),
        "200 + 100 should be refused by a guard, not by the return check: {overflow:?}"
    );
}

/// The mutation test: **every** column of an accepted witness is pinned by some constraint.
///
/// This is the only check here that an under-constrained circuit can fail, as it is the only one
/// that judges a witness the VM did not produce.
#[test]
fn every_witness_column_is_pinned_by_a_constraint() {
    for (kind, bits) in [
        (BinaryArithOpKind::UAdd, 8usize),
        (BinaryArithOpKind::UAdd, 200),
        (BinaryArithOpKind::USub, 200),
        (BinaryArithOpKind::UAdd, 253),
        (BinaryArithOpKind::USub, 253),
        (BinaryArithOpKind::UMul, 32),
        (BinaryArithOpKind::UMul, 127),
        (BinaryArithOpKind::UMul, 128),
        (BinaryArithOpKind::UMul, 200),
        (BinaryArithOpKind::UMul, 253),
        (BinaryArithOpKind::And, 40),
        (BinaryArithOpKind::And, 128),
        (BinaryArithOpKind::Xor, 200),
        (BinaryArithOpKind::UDiv, 16),
        (BinaryArithOpKind::SShr, 8),
    ] {
        let oracle = BinaryOpOracle::new(kind, bits, bits)
            .unwrap_or_else(|error| panic!("{kind:?} at {bits} bits does not compile: {error}"));
        let (lhs, rhs) = (IntBits::from_u128(bits, 7), IntBits::from_u128(bits, 3));
        let expected = mavros_int_semantics::eval(IntOp::from(kind), &lhs, &rhs)
            .value()
            .expect("7 op 3 is accepted for each operation swept here");
        assert_every_column_is_pinned(
            &format!("{kind:?} at {bits} bits"),
            oracle.compiled(),
            &[&lhs, &rhs, &expected],
        );
    }
}

/// The same mutation test for the ordering the wide subtraction shares its borrow chain with.
///
/// `ULt` is a [`CmpKind`] and not a [`BinaryArithOpKind`], so the oracle does not build it and the
/// program is stated here. The byte is a control: it holds on the single-cell lowering, so a red
/// row beside it is the wide lowering's and not this test's. 253 is the chain on a decomposition,
/// where the operands have an element each and their difference does not.
#[test]
fn every_witness_column_of_an_unsigned_ordering_is_pinned() {
    for bits in [8usize, 200, 253] {
        let ssa = main_program(
            &[Type::int(bits), Type::int(bits)],
            &[Type::int(1)],
            |e, params| vec![e.cmp(params[0], params[1], CmpKind::ULt)],
        );
        let compiled = Compiled::new(ssa)
            .unwrap_or_else(|error| panic!("ULt at {bits} bits does not compile: {error}"));

        // 7 < 3 is false, declared as the return so that the entry point's own check tests it.
        let (lhs, rhs) = (IntBits::from_u128(bits, 7), IntBits::from_u128(bits, 3));
        let answer = IntBits::from_u128(1, 0);
        assert_every_column_is_pinned(
            &format!("ULt at {bits} bits"),
            &compiled,
            &[&lhs, &rhs, &answer],
        );
    }
}

/// The mutation test past the width one field element carries, where the operands are **limbs**.
///
/// An entry point writes each integer to a single witness column and range-checks it there, so no
/// parameter is wider than the widest injectively-carried width, and neither is a declared return.
/// A limbed operand is therefore built inside the program: a narrow parameter is widened, and its
/// partner is a constant wide enough to reach the top limb. The sum and the difference leave
/// through their low limb and the ordering as one bit.
///
/// Neither is read back with an equality. `lower_eq` witnesses the inverse of the difference as a
/// hint, and that hint is free whenever the two sides are equal — which an honest check is — so an
/// equality would be a column this test finds unpinned whatever the chain does.
///
/// Each pair is chosen so that the chain runs its whole length. `(2^256 - 1) + 3` carries out of
/// every limb below the top one, as `2^300 - 3` borrows out of each, and `3 < 2^300` is decided
/// only in the top limb: every limb below it borrows nothing, so each of their carries is read and
/// found to be zero.
#[test]
fn every_witness_column_of_a_limbed_sum_difference_or_ordering_is_pinned() {
    let bits = 320usize;
    let high = BigUint::from(1u8) << 300;
    let low_limbs_full = (BigUint::from(1u8) << (4 * 64usize)) - 1u8;
    let parameter = IntBits::from_u128(64, 3);

    let sum = main_program(&[Type::int(64)], &[Type::int(64)], |e, params| {
        let addend = e.cast_to(CastTarget::Int(bits), params[0]);
        let augend = e.int_const(IntBits::from_biguint(bits, &low_limbs_full));
        let sum = e.bin(BinaryArithOpKind::UAdd, augend, addend);
        vec![e.cast_to(CastTarget::Int(64), sum)]
    });
    let sum_low_limb = IntBits::from_u128(64, 2);

    let difference = main_program(&[Type::int(64)], &[Type::int(64)], |e, params| {
        let subtrahend = e.cast_to(CastTarget::Int(bits), params[0]);
        let minuend = e.int_const(IntBits::from_biguint(bits, &high));
        let difference = e.bin(BinaryArithOpKind::USub, minuend, subtrahend);
        vec![e.cast_to(CastTarget::Int(64), difference)]
    });
    let low_limb = IntBits::from_u128(64, u128::from(u64::MAX - 2));

    let ordering = main_program(&[Type::int(64)], &[Type::int(1)], |e, params| {
        let left = e.cast_to(CastTarget::Int(bits), params[0]);
        let right = e.int_const(IntBits::from_biguint(bits, &high));
        vec![e.cmp(left, right, CmpKind::ULt)]
    });
    let less = IntBits::from_u128(1, 1);

    for (operation, ssa, answer) in [
        ("UAdd", sum, sum_low_limb),
        ("USub", difference, low_limb),
        ("ULt", ordering, less),
    ] {
        let compiled = Compiled::new(ssa)
            .unwrap_or_else(|error| panic!("{operation} at {bits} bits does not compile: {error}"));
        assert_every_column_is_pinned(
            &format!("{operation} at {bits} bits"),
            &compiled,
            &[&parameter, &answer],
        );
    }
}

/// The mutation test for a product whose operands are limbs, with one operand witnessed and with
/// both.
///
/// Built inside the program for the reason
/// [`every_witness_column_of_a_limbed_sum_difference_or_ordering_is_pinned`] gives. The left
/// operand is a widened parameter; the right one is a constant reaching the fourth limb, taken as
/// it is and then selected by a witnessed bit, which makes it a witnessed value held as limbs. The
/// two differ in what they cost: a partial product of a witnessed limb and a pure one is a linear
/// term, while one of two witnessed limbs is a column of its own that only its product constraint
/// pins.
///
/// The product leaves through its low limb. A missing range check on a limb of the answer or on a
/// carry is invisible here, whichever limb leaves: raising a carry by one lowers its limb by `2^h`
/// and raises the next by one, which the recombination cancels, so it takes two columns moving
/// together to exploit, and the structural audit in `wide_witness_ints` is what sees it.
#[test]
fn every_witness_column_of_a_limbed_product_is_pinned() {
    let bits = 320usize;
    let factor: BigUint = (BigUint::from(1u8) << 200) + 5u8;
    let parameter = IntBits::from_u128(64, 3);
    let low_limb = IntBits::from_u128(64, 15);
    let selected = IntBits::from_u128(1, 1);

    let pure_factor = factor.clone();
    let one_witnessed = main_program(&[Type::int(64)], &[Type::int(64)], move |e, params| {
        let multiplicand = e.cast_to(CastTarget::Int(bits), params[0]);
        let multiplier = e.int_const(IntBits::from_biguint(bits, &pure_factor));
        let product = e.bin(BinaryArithOpKind::UMul, multiplicand, multiplier);
        vec![e.cast_to(CastTarget::Int(64), product)]
    });
    let compiled = Compiled::new(one_witnessed)
        .unwrap_or_else(|error| panic!("UMul at {bits} bits does not compile: {error}"));
    assert_every_column_is_pinned(
        &format!("UMul by a constant at {bits} bits"),
        &compiled,
        &[&parameter, &low_limb],
    );

    let both_witnessed = main_program(
        &[Type::int(64), Type::int(1)],
        &[Type::int(64)],
        move |e, params| {
            let multiplicand = e.cast_to(CastTarget::Int(bits), params[0]);
            let constant = e.int_const(IntBits::from_biguint(bits, &factor));
            let zero = e.int_const(IntBits::zero(bits));
            let multiplier = e.select(params[1], constant, zero);
            let product = e.bin(BinaryArithOpKind::UMul, multiplicand, multiplier);
            vec![e.cast_to(CastTarget::Int(64), product)]
        },
    );
    let compiled = Compiled::new(both_witnessed)
        .unwrap_or_else(|error| panic!("UMul at {bits} bits does not compile: {error}"));
    assert_every_column_is_pinned(
        &format!("UMul of two witnessed values at {bits} bits"),
        &compiled,
        &[&parameter, &selected, &low_limb],
    );
}

/// The mutation test for the asserted ordering, in the band and on limbs.
///
/// **This is the one chain shape the mutation test cannot fully judge.** An asserted ordering fixes
/// its top borrow at one instead of witnessing it, so the top limb's identity has no column of its
/// own: dropping that limb's range check leaves every column pinned and this test green.
/// [`a_guarded_assertion_holds_only_where_its_guard_does`] is what fails then, by accepting a
/// failing assertion, and so does `wide_witness_ints`'s structural audit,
/// `every_carry_is_a_bit_and_every_limb_of_the_chain_is_bounded`. What this test pins is every
/// carry below the top.
#[test]
fn every_witness_column_of_an_asserted_ordering_is_pinned() {
    let accepted = IntBits::from_u128(1, 1);

    let band = main_program(
        &[Type::int(253), Type::int(253)],
        &[Type::int(1)],
        |e, params| {
            e.emit(OpCode::AssertCmp {
                kind: CmpKind::ULt,
                lhs: params[0],
                rhs: params[1],
            });
            vec![e.int_const(IntBits::from_u128(1, 1))]
        },
    );
    let compiled = Compiled::new(band)
        .unwrap_or_else(|error| panic!("an asserted int253 ordering does not compile: {error}"));
    assert_every_column_is_pinned(
        "an asserted ordering at 253 bits",
        &compiled,
        &[
            &IntBits::from_u128(253, 3),
            &IntBits::from_u128(253, 7),
            &accepted,
        ],
    );

    let bits = 320usize;
    let high = BigUint::from(1u8) << 300;
    let limbed = main_program(&[Type::int(64)], &[Type::int(1)], move |e, params| {
        let left = e.cast_to(CastTarget::Int(bits), params[0]);
        let right = e.int_const(IntBits::from_biguint(bits, &high));
        e.emit(OpCode::AssertCmp {
            kind: CmpKind::ULt,
            lhs: left,
            rhs: right,
        });
        vec![e.int_const(IntBits::from_u128(1, 1))]
    });
    let compiled = Compiled::new(limbed)
        .unwrap_or_else(|error| panic!("an asserted int320 ordering does not compile: {error}"));
    assert_every_column_is_pinned(
        "an asserted ordering at 320 bits",
        &compiled,
        &[&IntBits::from_u128(64, 3), &accepted],
    );
}

/// The unsigned sum and difference agree with the model where one element carries the operands.
///
/// 200 and 252 are the band past the funnel's narrow width that the single-cell lowering holds,
/// because the sum still fits an element. 253 is the one width where the operands fit and the sum
/// does not, so the chain runs there on a decomposition of each operand.
#[test]
fn a_sum_or_difference_agrees_with_the_model_at_the_limb_corners() {
    for bits in [200usize, 252, 253] {
        for kind in [BinaryArithOpKind::UAdd, BinaryArithOpKind::USub] {
            sweep_over(kind, bits, wide_corner_pairs(IntOp::from(kind), bits));
        }
    }
}

/// The unsigned ordering agrees with the model over the same widths, and its declared return is
/// bound to it.
///
/// The second half is the entry point's return check: witness generation is honest, so a run that
/// declares the other answer computes the right one and fails where the two are compared. It says
/// nothing about the ordering's own constraints, which
/// [`every_witness_column_of_an_unsigned_ordering_is_pinned`] is what probes.
///
/// The byte is the single-cell lowering the rest are measured against.
#[test]
fn an_unsigned_ordering_agrees_with_the_model_at_the_limb_corners() {
    for bits in [8usize, 200, 252, 253] {
        let ssa = main_program(
            &[Type::int(bits), Type::int(bits)],
            &[Type::int(1)],
            |e, params| vec![e.cmp(params[0], params[1], CmpKind::ULt)],
        );
        let compiled = Compiled::new(ssa)
            .unwrap_or_else(|error| panic!("ULt at {bits} bits does not compile: {error}"));

        let values = corners::wide_values(bits);
        for lhs in &values {
            for rhs in &values {
                let less = Limbed::Ordering
                    .model(lhs, rhs)
                    .expect("an ordering is always answered");
                let verdict = compiled.run(&input_block(&[lhs, rhs, &less]));
                assert!(
                    verdict.is_accepted(),
                    "ULt({lhs:?}, {rhs:?}) at {bits} bits is {less:?}: {verdict:?}"
                );
                let wrong = less.xor(&IntBits::from_u128(1, 1));
                assert!(
                    !compiled
                        .run(&input_block(&[lhs, rhs, &wrong]))
                        .is_accepted(),
                    "ULt({lhs:?}, {rhs:?}) at {bits} bits accepted the wrong answer {wrong:?}"
                );
            }
        }
    }
}

/// The operations the representation computes on limbs: the carry chain's three and the
/// schoolbook's product.
#[derive(Clone, Copy, Debug)]
enum Limbed {
    Sum,
    Difference,
    Ordering,
    Product,
}

impl Limbed {
    fn emit(self, e: &mut HLBlockEmitter<'_>, lhs: ValueId, rhs: ValueId) -> ValueId {
        match self {
            Limbed::Sum => e.bin(BinaryArithOpKind::UAdd, lhs, rhs),
            Limbed::Difference => e.bin(BinaryArithOpKind::USub, lhs, rhs),
            Limbed::Ordering => e.cmp(lhs, rhs, CmpKind::ULt),
            Limbed::Product => e.bin(BinaryArithOpKind::UMul, lhs, rhs),
        }
    }

    /// The model's answer, or [`None`] where it rejects the operands.
    fn model(self, lhs: &IntBits, rhs: &IntBits) -> Option<IntBits> {
        match self {
            Limbed::Sum => mavros_int_semantics::eval(IntOp::UAdd, lhs, rhs).value(),
            Limbed::Difference => mavros_int_semantics::eval(IntOp::USub, lhs, rhs).value(),
            Limbed::Product => mavros_int_semantics::eval(IntOp::UMul, lhs, rhs).value(),
            Limbed::Ordering => Some(IntBits::from_u128(
                1,
                u128::from(BigUint::from(lhs) < BigUint::from(rhs)),
            )),
        }
    }
}

/// The chain on limbs, over the corners a carry chain turns on, on both lanes.
///
/// A carry or a borrow crosses a limb boundary where a limb is full or empty, and leaves the value
/// at the top, so the corners are the ones either side of the lowest boundary and of the top limb's,
/// and the extremes. 254 has a top limb narrower than the rest and 320 a full one. The limb is the
/// **field's** witness limb, which is what the chain splits at; it is only on bn254 that it is also
/// the host word.
///
/// Every corner [`corners::wide_values`] has is [`the_limbed_corner_matrix_agrees_with_the_model`].
/// Those are placed at the host word, since the model knows no field, so they are the chain's
/// corners only where the two limbs coincide — which the matrix asserts before it runs.
#[test]
fn limbed_sums_differences_and_orderings_agree_with_the_model() {
    let limb = witness_limb_bits(FieldConfig::bn254());
    for bits in [254usize, 320] {
        let one = IntBits::from_u128(bits, 1);
        let top_limb = (bits - 1) / limb * limb;
        let values = [
            IntBits::zero(bits),
            one.clone(),
            IntBits::all_ones(limb).cast(bits),
            one.shifted_left(limb),
            one.shifted_left(top_limb),
            IntBits::all_ones(bits),
        ];
        for limbed in [Limbed::Sum, Limbed::Difference, Limbed::Ordering] {
            check_limbed_corners(limbed, bits, &values, Rhs::Witnessed);
        }
    }
}

/// [`limbed_sums_differences_and_orderings_agree_with_the_model`] over every limb corner.
#[test]
#[ignore = "around 2 400 compiled pairs; several minutes in a debug build"]
fn the_limbed_corner_matrix_agrees_with_the_model() {
    assert_eq!(
        witness_limb_bits(FieldConfig::bn254()),
        HOST_LIMB_BITS,
        "the model's limb corners are the chain's only while the two limbs coincide"
    );
    for bits in [254usize, 320] {
        let values = corners::wide_values(bits);
        for limbed in [Limbed::Sum, Limbed::Difference, Limbed::Ordering] {
            check_limbed_corners(limbed, bits, &values, Rhs::Witnessed);
        }
        let mut values = values;
        values.extend(half_width_corners(bits));
        check_limbed_corners(Limbed::Product, bits, &dedup(values), Rhs::Witnessed);
    }
}

/// The corners a product turns on, which the model's wide set does not have: either side of half
/// the width, so that the products of two of them land either side of `2^N`.
///
/// `2^⌊N/2⌋ · 2^⌈N/2⌉` is exactly `2^N`, the smallest product that overflows, and it overflows by
/// nothing but a carry out of the top column. At an even width `(2^(N/2) + 1)(2^(N/2) - 1)` is the
/// largest answer there is, which carries through every column on the way.
fn half_width_corners(bits: usize) -> Vec<IntBits> {
    let one = IntBits::from_u128(bits, 1);
    let values = [bits / 2, bits.div_ceil(2)].into_iter().flat_map(|half| {
        let power = one.shifted_left(half);
        [
            IntBits::all_ones(half).cast(bits),
            power.clone(),
            power.or(&one),
        ]
    });
    dedup(values.collect())
}

/// `values` with repeats removed, in order.
fn dedup(values: Vec<IntBits>) -> Vec<IntBits> {
    let mut out: Vec<IntBits> = Vec::with_capacity(values.len());
    for value in values {
        if !out.contains(&value) {
            out.push(value);
        }
    }
    out
}

/// A product agrees with the model where the operands have an element each and their product does
/// not, which is where the schoolbook runs on a decomposition of each operand.
///
/// 127 is the one width below the double lane whose product the single cell cannot hold. 129 and
/// 200 are past both narrow lanes, and 253 is the widest width an element carries.
#[test]
fn a_product_agrees_with_the_model_at_the_limb_corners() {
    for bits in [127usize, 129, 200, 253] {
        let mut values = corners::wide_values(bits);
        values.extend(half_width_corners(bits));
        let values = dedup(values);
        let pairs = values
            .iter()
            .flat_map(|a| values.iter().map(move |b| (a.clone(), b.clone())))
            .collect();
        sweep_over(BinaryArithOpKind::UMul, bits, pairs);
    }
}

/// The product on limbs, over the corners its carries and its overflow check turn on, on both
/// lanes, with both operands witnessed and with a constant multiplier.
///
/// A constant is cut into limbs at compile time, and a limb it does not reach drops out of the
/// plan along with every partial product and overflow term against it. So the constant half is
/// what checks that nothing needed dropped out with them: `2^top_limb` is a constant whose only
/// limb is the top one, whose overflow terms are all that stand between a large operand and a
/// wrong answer.
#[test]
fn limbed_products_agree_with_the_model() {
    let limb = witness_limb_bits(FieldConfig::bn254());
    for bits in [254usize, 320] {
        let one = IntBits::from_u128(bits, 1);
        let top_limb = (bits - 1) / limb * limb;
        let mut values = vec![
            IntBits::zero(bits),
            one.clone(),
            IntBits::all_ones(limb).cast(bits),
            one.shifted_left(limb),
            one.shifted_left(top_limb),
            IntBits::all_ones(bits),
        ];
        values.extend(half_width_corners(bits));
        let values = dedup(values);
        for rhs in [Rhs::Witnessed, Rhs::Constant] {
            check_limbed_corners(Limbed::Product, bits, &values, rhs);
        }
    }
}

/// Every pair of `values`, checked by [`check_limbed_pairs`] a program's worth at a time.
///
/// Split because of the host rather than the compiler. On aarch64, wasmtime's Cranelift (0.118)
/// trips its own branch-range assertion, `(label_offset - offset) <= kind.max_pos_range()` in
/// `machinst/buffer.rs`, compiling a function holding a few hundred pairs, while the VM lane runs
/// the same program to acceptance. Fifty sums, differences or orderings per program stay clear of
/// it. A product of two dense operands is several times their code, and fifty of those do not.
fn check_limbed_corners(limbed: Limbed, bits: usize, values: &[IntBits], rhs: Rhs) {
    let pairs: Vec<(IntBits, IntBits)> = values
        .iter()
        .flat_map(|a| values.iter().map(move |b| (a.clone(), b.clone())))
        .collect();
    let per_program = match limbed {
        Limbed::Product => 20,
        Limbed::Sum | Limbed::Difference | Limbed::Ordering => 50,
    };
    for pairs in pairs.chunks(per_program) {
        check_limbed_pairs(limbed, bits, pairs.to_vec(), rhs);
    }
}

/// How [`check_limbed_pairs`] builds its right operand.
#[derive(Clone, Copy, Debug)]
enum Rhs {
    /// Selected by a witnessed bit, as the left one is: a witnessed value held as limbs.
    Witnessed,

    /// The corner constant itself, which a lowering sees as a constant and may cut at compile time.
    Constant,
}

/// `limbed` over `pairs`, all in one program, against the model.
///
/// No parameter can be this wide, so each operand is a corner constant selected by a witnessed bit,
/// one per side: a witnessed value held as limbs, whose value is the constant. Every pair is then
/// one program's worth of arithmetic, and one program holds them all.
///
/// A pair the model rejects cannot sit beside the others unconditionally, since it would refuse
/// every run. Its operands are selected by a second parameter naming the pair instead, and are
/// zero on every run but the one that names it — where the pipeline has to refuse, and only there.
/// A constant right operand is not selected, so there it is the left one alone that goes to zero.
fn check_limbed_pairs(limbed: Limbed, bits: usize, pairs: Vec<(IntBits, IntBits)>, rhs: Rhs) {
    let answers: Vec<Option<IntBits>> = pairs.iter().map(|(a, b)| limbed.model(a, b)).collect();
    let zero = IntBits::zero(bits);
    let unnamed: Vec<IntBits> = pairs
        .iter()
        .map(|(_, right)| {
            let right = match rhs {
                Rhs::Witnessed => &zero,
                Rhs::Constant => right,
            };
            limbed
                .model(&zero, right)
                .expect("the operation accepts a zero left operand against this right one")
        })
        .collect();

    let (program_pairs, program_answers) = (pairs.clone(), answers.clone());
    let ssa = main_program(
        &[Type::int(1), Type::int(32), Type::int(1)],
        &[Type::int(1)],
        move |e, params| {
            let zero = e.int_const(IntBits::zero(bits));
            let mut all = None;
            for (index, (((lhs, right), answer), unnamed)) in program_pairs
                .iter()
                .zip(&program_answers)
                .zip(&unnamed)
                .enumerate()
            {
                // Each side has a selector of its own, so that `a op b` and `b op a` are different
                // values to the optimiser. With one selector the two are the same product, CSE
                // keeps one of them, and a lowering that treats its operands differently is only
                // ever run one way round.
                let (left_chosen, right_chosen, want) = match answer {
                    Some(want) => (params[0], params[2], want.clone()),
                    None => {
                        let name = e.int_const(IntBits::from_u128(32, index as u128));
                        let named = e.eq(params[1], name);
                        (named, named, unnamed.clone())
                    }
                };
                let lhs = e.int_const(lhs.clone());
                let lhs = e.select(left_chosen, lhs, zero);
                let right = e.int_const(right.clone());
                let rhs = match rhs {
                    Rhs::Witnessed => e.select(right_chosen, right, zero),
                    Rhs::Constant => right,
                };
                let got = limbed.emit(e, lhs, rhs);
                let want = e.int_const(want);
                let same = e.eq(got, want);
                all = Some(match all {
                    None => same,
                    Some(all) => e.bin(BinaryArithOpKind::And, all, same),
                });
            }
            vec![all.expect("there is at least one corner pair")]
        },
    );
    let compiled = Compiled::new(ssa).unwrap_or_else(|error| {
        panic!("{limbed:?} at {bits} bits, {rhs:?} right operand, does not compile: {error}")
    });

    let one = IntBits::from_u128(1, 1);
    let inputs =
        |selected: u128| input_block(&[&one, &IntBits::from_u128(32, selected), &one, &one]);

    let none = u128::from(u32::MAX);
    let verdict = compiled.run(&inputs(none));
    assert!(
        verdict.is_accepted(),
        "{limbed:?} at {bits} bits, {rhs:?} right operand, disagrees with the model on some pair it accepts: {verdict:?}"
    );
    let verdict = compiled
        .run_wasm(&inputs(none))
        .expect("the WASM lane builds");
    assert!(
        verdict.is_accepted(),
        "{limbed:?} at {bits} bits, {rhs:?} right operand, disagrees with the model on the WASM lane: {verdict:?}"
    );

    for (index, ((lhs, rhs), answer)) in pairs.iter().zip(&answers).enumerate() {
        if answer.is_some() {
            continue;
        }
        let verdict = compiled.run(&inputs(index as u128));
        assert!(
            verdict.is_refusal(),
            "{limbed:?}({lhs:?}, {rhs:?}) at {bits} bits is rejected by the model, but the \
             pipeline answered {verdict:?}"
        );
    }
}

/// A difference under a witnessed condition is checked only where the condition holds.
///
/// Inside a branch on a witness the operation is guarded, and a guard that is off means the
/// operands are whatever the branch not taken left behind — here a difference that would borrow
/// out of the top limb. The chain's checks go off with the guard, and the answer is zero.
///
/// 253 is the chain on a decomposition of a single element, and 320 on limbs.
#[test]
fn a_guarded_difference_is_checked_only_where_its_guard_holds() {
    for bits in [253usize, 320] {
        // Both operands share their top bits, so the difference is decided in the lowest limb and
        // a borrow out of it has to run the whole chain.
        let high = IntBits::from_biguint(bits, &(BigUint::from(1u8) << (bits - 3)));
        let ssa = main_program(
            &[Type::int(1), Type::int(64), Type::int(64)],
            &[Type::int(64)],
            move |e, params| {
                let lhs = e.cast_to(CastTarget::Int(bits), params[1]);
                let top = e.int_const(high.clone());
                let lhs = e.bin(BinaryArithOpKind::Or, lhs, top);
                let rhs = e.cast_to(CastTarget::Int(bits), params[2]);
                let rhs = e.bin(BinaryArithOpKind::Or, rhs, top);

                let (then_id, _) = e.add_block();
                let (else_id, _) = e.add_block();
                let (merge_id, _) = e.add_block();
                e.seal_and_switch(Terminator::JmpIf(params[0], then_id, else_id), then_id);
                let difference = e.bin(BinaryArithOpKind::USub, lhs, rhs);
                let low = e.cast_to(CastTarget::Int(64), difference);
                e.seal_and_switch(Terminator::Jmp(merge_id, vec![low]), else_id);
                let zero = e.int_const(IntBits::zero(64));
                e.seal_and_switch(Terminator::Jmp(merge_id, vec![zero]), merge_id);
                vec![e.add_parameter(Type::int(64))]
            },
        );
        let compiled = Compiled::new(ssa)
            .unwrap_or_else(|error| panic!("a guarded int{bits} USub does not compile: {error}"));

        let run = |taken: u128, lhs: u128, rhs: u128, answer: u128| {
            compiled.run(&input_block(&[
                &IntBits::from_u128(1, taken),
                &IntBits::from_u128(64, lhs),
                &IntBits::from_u128(64, rhs),
                &IntBits::from_u128(64, answer),
            ]))
        };

        let verdict = run(1, 7, 3, 4);
        assert!(
            verdict.is_accepted(),
            "int{bits}: taken, 7 - 3: {verdict:?}"
        );
        let verdict = run(1, 3, 7, 0);
        assert!(verdict.is_refusal(), "int{bits}: taken, 3 - 7: {verdict:?}");
        let verdict = run(0, 3, 7, 0);
        assert!(
            verdict.is_accepted(),
            "int{bits}: not taken, 3 - 7: {verdict:?}"
        );
    }
}

/// A product under a witnessed condition is checked only where the condition holds.
///
/// The multiplier is `2^(N - 2)` or-ed onto a parameter, so a multiplicand of three fits and one of
/// four overflows by exactly its top column. Where the branch is not taken that overflow is what the
/// branch not taken left behind, and the schoolbook's checks go off with the guard.
///
/// 253 is the schoolbook on a decomposition of a single element, and 320 on limbs.
#[test]
fn a_guarded_product_is_checked_only_where_its_guard_holds() {
    for bits in [253usize, 320] {
        let high = IntBits::from_biguint(bits, &(BigUint::from(1u8) << (bits - 2)));
        let ssa = main_program(
            &[Type::int(1), Type::int(64), Type::int(64)],
            &[Type::int(64)],
            move |e, params| {
                let lhs = e.cast_to(CastTarget::Int(bits), params[1]);
                let rhs = e.cast_to(CastTarget::Int(bits), params[2]);
                let top = e.int_const(high.clone());
                let rhs = e.bin(BinaryArithOpKind::Or, rhs, top);

                let (then_id, _) = e.add_block();
                let (else_id, _) = e.add_block();
                let (merge_id, _) = e.add_block();
                e.seal_and_switch(Terminator::JmpIf(params[0], then_id, else_id), then_id);
                let product = e.bin(BinaryArithOpKind::UMul, lhs, rhs);
                let low = e.cast_to(CastTarget::Int(64), product);
                e.seal_and_switch(Terminator::Jmp(merge_id, vec![low]), else_id);
                let zero = e.int_const(IntBits::zero(64));
                e.seal_and_switch(Terminator::Jmp(merge_id, vec![zero]), merge_id);
                vec![e.add_parameter(Type::int(64))]
            },
        );
        let compiled = Compiled::new(ssa)
            .unwrap_or_else(|error| panic!("a guarded int{bits} UMul does not compile: {error}"));

        let run = |taken: u128, lhs: u128, rhs: u128, answer: u128| {
            compiled.run(&input_block(&[
                &IntBits::from_u128(1, taken),
                &IntBits::from_u128(64, lhs),
                &IntBits::from_u128(64, rhs),
                &IntBits::from_u128(64, answer),
            ]))
        };

        // `3 * (2^(N - 2) + 5)`, whose low limb is fifteen.
        let verdict = run(1, 3, 5, 15);
        assert!(
            verdict.is_accepted(),
            "int{bits}: taken, 3 * (2^(N - 2) + 5): {verdict:?}"
        );
        let verdict = run(1, 4, 5, 20);
        assert!(
            verdict.is_refusal(),
            "int{bits}: taken, 4 * (2^(N - 2) + 5): {verdict:?}"
        );
        let verdict = run(0, 4, 5, 0);
        assert!(
            verdict.is_accepted(),
            "int{bits}: not taken, 4 * (2^(N - 2) + 5): {verdict:?}"
        );
    }
}

/// An assertion under a witnessed condition holds only where the condition does.
///
/// The ordering is the chain with its top borrow fixed at one, and equality one assertion per limb;
/// a guard that is off turns every check of either off with it. 253 is the chain on a decomposition,
/// where equality needs nothing of the representation, and 320 is both on limbs.
#[test]
fn a_guarded_assertion_holds_only_where_its_guard_does() {
    for (kind, holds, fails) in [
        (CmpKind::ULt, (3, 7), (7, 3)),
        (CmpKind::Eq, (7, 7), (7, 3)),
    ] {
        for bits in [253usize, 320] {
            let high = IntBits::from_biguint(bits, &(BigUint::from(1u8) << (bits - 3)));
            let ssa = main_program(
                &[Type::int(1), Type::int(64), Type::int(64)],
                &[Type::int(1)],
                move |e, params| {
                    let lhs = e.cast_to(CastTarget::Int(bits), params[1]);
                    let top = e.int_const(high.clone());
                    let lhs = e.bin(BinaryArithOpKind::Or, lhs, top);
                    let rhs = e.cast_to(CastTarget::Int(bits), params[2]);
                    let rhs = e.bin(BinaryArithOpKind::Or, rhs, top);

                    let (then_id, _) = e.add_block();
                    let (merge_id, _) = e.add_block();
                    e.seal_and_switch(Terminator::JmpIf(params[0], then_id, merge_id), then_id);
                    e.emit(OpCode::AssertCmp { kind, lhs, rhs });
                    e.seal_and_switch(Terminator::Jmp(merge_id, vec![]), merge_id);
                    vec![params[0]]
                },
            );
            let what = format!("a guarded int{bits} {kind:?} assertion");
            let compiled = Compiled::new(ssa)
                .unwrap_or_else(|error| panic!("{what} does not compile: {error}"));

            let run = |taken: u128, (lhs, rhs): (u128, u128)| {
                compiled.run(&input_block(&[
                    &IntBits::from_u128(1, taken),
                    &IntBits::from_u128(64, lhs),
                    &IntBits::from_u128(64, rhs),
                    &IntBits::from_u128(1, taken),
                ]))
            };
            let verdict = run(1, holds);
            assert!(verdict.is_accepted(), "{what}, taken, holding: {verdict:?}");
            let verdict = run(1, fails);
            assert!(verdict.is_refusal(), "{what}, taken, failing: {verdict:?}");
            let verdict = run(0, fails);
            assert!(
                verdict.is_accepted(),
                "{what}, not taken, failing: {verdict:?}"
            );
        }
    }
}

/// The whole matrix: every operation at every width the model sweeps, over every corner pair.
///
/// Around 32 000 operand pairs and half a minute and hence not default. Run it with
/// `cargo test -p mavros-compiler --test bigints -- --ignored`.
#[test]
#[ignore = "around 32 000 compiled-and-run operand pairs; roughly half a minute"]
fn the_whole_corner_matrix_agrees_with_the_model() {
    for kind in BinaryArithOpKind::ALL {
        let op = IntOp::from(kind);
        for &bits in corners::widths_for(op.is_signed()) {
            if op.is_shift() && (!bits.is_power_of_two() || bits > 64) {
                continue;
            }
            sweep(kind, bits);
        }
    }
}

// KNOWN DIVERGENCES
// ================================================================================================
//
// Each of these pins a refusal the pipeline makes today to make it clear when they are fixed.
// Every one of them is a width the type system admits and no lowering builds, so the compiler owes
// the program a diagnostic rather than a panic; the assertions are on the diagnostic's own text,
// which is what a reader of the refusal actually gets.

/// The rendered refusal for a program the pipeline declines to compile.
///
/// # Panics
///
/// If the program compiles, or fails for any reason other than a refusal — a crash reports as a
/// crash, and reading one as a refusal is the distinction this whole family of tests is about.
fn refusal_for(kind: BinaryArithOpKind, lhs_bits: usize, rhs_bits: usize) -> String {
    let Err(error) = BinaryOpOracle::new(kind, lhs_bits, rhs_bits) else {
        panic!("an int{lhs_bits} {kind:?} compiled rather than being refused");
    };
    assert!(
        matches!(error, DriverError::Refused(_)),
        "a width no lowering builds is a refusal rather than a crash: {error}"
    );
    error.to_string()
}

/// Assert that `text` contains every one of `expected`.
///
/// The oracle builds its programs in memory, so their locations name no file and each refusal
/// renders as its header and its notes with no source snippet between them. The caret label is
/// covered where there is a file to draw it under, in
/// [`an_oversized_cast_to_field_is_refused_with_a_diagnostic`].
fn contains_all(text: &str, expected: &[&str]) {
    for line in expected {
        assert!(text.contains(line), "no {line:?} in:\n{text}");
    }
}

/// A witnessed shift bounds its amount by the low `log2(bits)` bits of it, which is the bound
/// itself only at a power-of-two width. Retiring this means rebuilding the check as a real
/// `amount < bits` comparison.
#[test]
fn a_witness_shift_at_a_non_power_of_two_width_is_refused() {
    contains_all(
        &refusal_for(BinaryArithOpKind::UShl, 5, 5),
        &[
            "error: a witnessed int5 left shift is not supported",
            "= note: a witnessed shift bounds its amount by the low bits of it",
        ],
    );
}

/// A 128-bit witness `<<` lowers to the single field product `lhs * 2^n`, which reaches 2^255 and
/// wraps. Retiring this means splitting the shift limb-wise.
///
/// The amount here is an entry-point parameter, so the program states none and the refusal is on
/// the widest amount the width admits. An amount the program _does_ state is a different question,
/// answered in [`a_witness_shift_left_by_a_literal_amount_that_fits_is_lowered`].
#[test]
fn a_witness_shift_left_at_a_hundred_and_twenty_eight_bits_is_refused() {
    contains_all(
        &refusal_for(BinaryArithOpKind::UShl, 128, 128),
        &[
            "error: a witnessed int128 left shift is not supported",
            "= note: a witnessed shift forms `value * 2^amount` in one field element",
            "= note: an amount the program states as a literal is held to that amount",
        ],
    );
}

/// The same width, shifted by an amount written in the program, compiles and agrees.
///
/// `2^(128 + 120)` sits well inside the modulus, so the single field product is honest and its
/// range check means what it says. This is `noir-bignum`'s `get_double_modulus`, which is what the
/// three `passport` tests in the local corpus compile.
#[test]
fn a_witness_shift_left_by_a_literal_amount_that_fits_is_lowered() {
    let bits = 128;
    let amount = 120;
    let value = IntBits::from_u128(bits, 0xab);

    let ssa = main_program(&[Type::int(bits)], &[Type::int(bits)], |e, params| {
        let literal = e.int_const(IntBits::from_u128(bits, amount as u128));
        vec![e.bin(BinaryArithOpKind::UShl, params[0], literal)]
    });
    let compiled = Compiled::new(ssa).expect("a literal amount with headroom compiles");

    let verdict = compiled.run(&input_block(&[&value, &value.shifted_left(amount)]));
    assert!(verdict.is_accepted(), "int{bits} << {amount}: {verdict:?}");
}

/// A shift admitted by its literal amount still meets the 128-bit lowering's own divergence: an
/// operand whose shifted product leaves the width is **rejected at proving time** rather than
/// wrapped.
///
/// `wrap_shifted_product` truncates through an `Int(2 * bits)` intermediate, which at 128 bits is
/// an `Int(256)` nothing can express yet, so that width falls back to a trapping range check on the
/// product. Admitting the literal amount neither causes this nor fixes it — it is the shift's other
/// divergence, and a wide multiply is what retires it.
#[test]
fn a_literal_amount_does_not_make_the_hundred_and_twenty_eight_bit_shift_wrap() {
    let bits = 128;
    let amount = 120;

    // One bit too many to survive the shift: `2^8 << 120` is `2^128`, which the product's range
    // check refuses where Noir's `<<` would have discarded it.
    let value = IntBits::from_u128(bits, 1 << 8);

    let ssa = main_program(&[Type::int(bits)], &[Type::int(bits)], |e, params| {
        let literal = e.int_const(IntBits::from_u128(bits, amount as u128));
        vec![e.bin(BinaryArithOpKind::UShl, params[0], literal)]
    });
    let compiled = Compiled::new(ssa).expect("a literal amount with headroom compiles");

    let verdict = compiled.run(&input_block(&[&value, &value.shifted_left(amount)]));
    assert!(
        !verdict.is_accepted(),
        "an overflowing int{bits} << {amount} is rejected rather than wrapped: {verdict:?}"
    );
}

/// And the exemption stops where the product does: one bit more of amount leaves the field.
///
/// `widest_injective_int_bits` is 253 on bn254, so a 128-bit value shifts by up to 125 and no
/// further.
#[test]
fn a_literal_shift_amount_is_admitted_only_while_its_own_product_fits() {
    let bits = 128;
    let widest_amount = first_width_past_the_field() - 1 - bits;

    for (amount, refused) in [(widest_amount, false), (widest_amount + 1, true)] {
        let ssa = main_program(&[Type::int(bits)], &[Type::int(bits)], |e, params| {
            let literal = e.int_const(IntBits::from_u128(bits, amount as u128));
            vec![e.bin(BinaryArithOpKind::UShl, params[0], literal)]
        });

        assert_eq!(
            Compiled::new(ssa).is_err(),
            refused,
            "int{bits} << {amount} should {} be refused",
            if refused { "" } else { "not" }
        );
    }
}

/// A **pure** wide value past the modulus that no fold can compute, alone and as a product's
/// operand.
///
/// `3^200` built by a loop is an `int320` no fold answers, so R1CS generation computes it, and it
/// has no field element. It is carried there as its pattern, which is what lets the loop run at
/// all, and the schoolbook cuts it into limbs on the pure side, each an element again. A limb that
/// kept the bits above it would be a wrong coefficient in the product's constraints, which is what
/// the multiplied run checks, and a multiplier of sixteen overflows the width.
#[test]
fn a_pure_wide_value_a_loop_computes_is_carried_to_its_limbs() {
    let bits = 320usize;
    let power: BigUint = BigUint::from(3u8).pow(200u32);
    assert!(power.bits() > 254 && power.bits() < 320);
    let low = |value: &BigUint| IntBits::from_biguint(64, &(value % (BigUint::from(1u8) << 64)));

    for multiplied in [false, true] {
        let ssa = main_program(&[Type::int(64)], &[Type::int(64)], move |e, params| {
            let zero = e.int_const(IntBits::zero(32));
            let one = e.int_const(IntBits::from_u128(bits, 1));
            let (head, _) = e.add_block();
            let (back, _) = e.add_block();
            let (done, _) = e.add_block();
            e.seal_and_switch(Terminator::Jmp(head, vec![zero, one]), head);
            let count = e.add_parameter(Type::int(32));
            let power = e.add_parameter(Type::int(bits));
            let three = e.int_const(IntBits::from_u128(bits, 3));
            let power = e.bin(BinaryArithOpKind::UMul, power, three);
            let step = e.int_const(IntBits::from_u128(32, 1));
            let count = e.bin(BinaryArithOpKind::UAdd, count, step);
            let limit = e.int_const(IntBits::from_u128(32, 200));
            let more = e.cmp(count, limit, CmpKind::ULt);
            e.seal_and_switch(Terminator::JmpIf(more, back, done), back);
            e.seal_and_switch(Terminator::Jmp(head, vec![count, power]), done);
            let value = if multiplied {
                let factor = e.cast_to(CastTarget::Int(bits), params[0]);
                e.bin(BinaryArithOpKind::UMul, factor, power)
            } else {
                power
            };
            vec![e.cast_to(CastTarget::Int(64), value)]
        });
        let compiled = Compiled::new(ssa).unwrap_or_else(|error| {
            panic!("the loop, multiplied: {multiplied}, does not compile: {error}")
        });
        let inputs = |factor: u64, answer: &IntBits| {
            input_block(&[&IntBits::from_u128(64, u128::from(factor)), answer])
        };

        for factor in [1u64, 5] {
            let answer = if multiplied {
                low(&(&power * factor))
            } else {
                low(&power)
            };
            let verdict = compiled.run(&inputs(factor, &answer));
            assert!(
                verdict.is_accepted(),
                "multiplied: {multiplied}, factor {factor}: {verdict:?}"
            );
            let verdict = compiled
                .run_wasm(&inputs(factor, &answer))
                .expect("the WASM lane builds");
            assert!(
                verdict.is_accepted(),
                "multiplied: {multiplied}, factor {factor}, WASM: {verdict:?}"
            );
            let wrong = answer.xor(&IntBits::from_u128(64, 1));
            assert!(
                !compiled.run(&inputs(factor, &wrong)).is_accepted(),
                "multiplied: {multiplied}, factor {factor} accepted a wrong answer"
            );
        }
        if multiplied {
            let verdict = compiled.run(&inputs(16, &IntBits::zero(64)));
            assert!(verdict.is_refusal(), "16 * 3^200 overflows: {verdict:?}");
        }
    }
}

/// A witnessed sum, difference or product whose result nothing reads still rejects an overflow.
///
/// Noir rejects an overflowing checked operation whether or not anything reads it. The rejection of
/// a witnessed one is a range check its lowering puts on the result, and dead-code elimination runs
/// between taint inference and that lowering, so it has to keep the operation rather than delete
/// the check with it. The left operand carries every bit above the byte, so at the wide widths the
/// sum of two bytes past 255 overflows the whole value, and the byte widths are the control.
#[test]
fn a_dead_witnessed_operation_still_rejects_its_overflow() {
    let one = IntBits::from_u128(1, 1);
    for bits in [8usize, 200, 320] {
        for kind in [
            BinaryArithOpKind::UAdd,
            BinaryArithOpKind::USub,
            BinaryArithOpKind::UMul,
        ] {
            let ssa = main_program(
                &[Type::int(8), Type::int(8)],
                &[Type::int(1)],
                move |e, params| {
                    let lhs = e.cast_to(CastTarget::Int(bits), params[0]);
                    let rhs = e.cast_to(CastTarget::Int(bits), params[1]);
                    let lhs = if bits > 8 {
                        let above_the_byte =
                            IntBits::all_ones(bits).xor(&IntBits::all_ones(8).cast(bits));
                        let above_the_byte = e.int_const(above_the_byte);
                        e.bin(BinaryArithOpKind::Or, lhs, above_the_byte)
                    } else {
                        lhs
                    };
                    e.bin(kind, lhs, rhs);
                    vec![e.int_const(IntBits::from_u128(1, 1))]
                },
            );
            let compiled = Compiled::new(ssa).unwrap_or_else(|error| {
                panic!("a dead int{bits} {kind:?} does not compile: {error}")
            });
            let run = |lhs: u128, rhs: u128| {
                compiled.run(&input_block(&[
                    &IntBits::from_u128(8, lhs),
                    &IntBits::from_u128(8, rhs),
                    &one,
                ]))
            };

            // A wide difference cannot underflow from these operands, whose top bits are set, so
            // only the byte has an overflowing difference to try.
            let (fits, overflows) = match kind {
                BinaryArithOpKind::USub if bits == 8 => ((2, 1), Some((1, 2))),
                BinaryArithOpKind::USub => ((1, 2), None),
                _ => ((1, 1), Some((200, 100))),
            };
            let verdict = run(fits.0, fits.1);
            assert!(
                verdict.is_accepted(),
                "int{bits} {kind:?}{fits:?}: {verdict:?}"
            );
            if let Some((lhs, rhs)) = overflows {
                let verdict = run(lhs, rhs);
                assert!(
                    verdict.is_refusal(),
                    "int{bits} {kind:?}({lhs}, {rhs}) overflows, and nothing reads it: {verdict:?}"
                );
            }
        }
    }
}

/// The same for the signed three, which reach a different lowering but owe the same rejection.
///
/// Each pair is two's complement at the width: at 8 bits `100 + 100`, `-100 - 100` and `16 * 16`
/// each leave `-128..=127`, and at 64 bits `2^62 + 2^62`, `-2^63 - 1` and `2^32 * 2^31` leave the
/// signed range the same way.
#[test]
fn a_dead_witnessed_signed_operation_still_rejects_its_overflow() {
    let one = IntBits::from_u128(1, 1);
    // Two's complement at a width no wider than a `u128` holds.
    let negative = |bits: usize, magnitude: u128| (1u128 << bits) - magnitude;
    for bits in [8usize, 64] {
        let cases = [
            (
                BinaryArithOpKind::SAdd,
                (1, 1),
                if bits == 8 {
                    (100, 100)
                } else {
                    (1 << 62, 1 << 62)
                },
            ),
            (
                BinaryArithOpKind::SSub,
                (1, 2),
                if bits == 8 {
                    (negative(8, 100), 100)
                } else {
                    (negative(64, 1 << 63), 1)
                },
            ),
            (
                BinaryArithOpKind::SMul,
                (3, 5),
                if bits == 8 {
                    (16, 16)
                } else {
                    (1 << 32, 1 << 31)
                },
            ),
        ];
        for (kind, fits, overflows) in cases {
            let ssa = main_program(
                &[Type::int(bits), Type::int(bits)],
                &[Type::int(1)],
                move |e, params| {
                    e.bin(kind, params[0], params[1]);
                    vec![e.int_const(IntBits::from_u128(1, 1))]
                },
            );
            let compiled = Compiled::new(ssa).unwrap_or_else(|error| {
                panic!("a dead int{bits} {kind:?} does not compile: {error}")
            });
            let run = |(lhs, rhs): (u128, u128)| {
                compiled.run(&input_block(&[
                    &IntBits::from_u128(bits, lhs),
                    &IntBits::from_u128(bits, rhs),
                    &one,
                ]))
            };
            let verdict = run(fits);
            assert!(
                verdict.is_accepted(),
                "int{bits} {kind:?}{fits:?}: {verdict:?}"
            );
            let verdict = run(overflows);
            assert!(
                verdict.is_refusal(),
                "int{bits} {kind:?}{overflows:?} overflows, and nothing reads it: {verdict:?}"
            );
        }
    }
}

/// A 129-bit division is refused for its witnessed operands and for nothing else.
///
/// The program the oracle builds carries the operation twice: witness taint inference splits the
/// entry point by context, so a **pure** copy of the division sits beside the witnessed one. Only
/// the witnessed copy has no lowering, and that is what makes this the closest thing to end-to-end
/// evidence that the pure division's check is built at this width.
///
/// The count is the assertion. The witness refusal's own second note tells the reader the operation
/// is supported outside the witness domain, and a second refusal saying it is not would contradict
/// it in the same compile.
#[test]
fn a_wide_division_is_refused_for_its_witness_operands_alone() {
    let refusal = refusal_for(BinaryArithOpKind::UDiv, 129, 129);
    contains_all(
        &refusal,
        &[
            "error: a witnessed int129 division is not supported",
            "= note: the same operation is supported at this width outside the witness domain",
        ],
    );
    assert_eq!(
        refusal.matches("division is not supported").count(),
        1,
        "the pure copy of the same division is lowered, not refused:\n{refusal}"
    );
}

/// A witnessed bitwise operation computes at **every** width, as the unsigned sum, difference and
/// ordering do, but by a different route: limb by limb with no carry between them.
///
/// **The widths are the ones that used to be gaps, and each was a different gap.** `40` fell off
/// the two limb cases into a spread at the operand's own width, which the VM's `SpreadU32ToU64`
/// stops at 32. `96` reached a _different_ ceiling — the spread of a `65..=127`-bit operand is a
/// `130..=254`-bit value, and `Unspread`'s typing rule refused anything above `int128`. `200` was
/// past the narrow bound with no representation to split it. `254` is the first width the
/// representation splits **raggedly**, so its top limb is 62 bits and lands back in the first gap.
///
/// What closes all four is the same thing: the decomposition runs at the **half-limb**, which is
/// what the spread instruction bounds, and the top half-limb carries only the bits that are left.
#[test]
fn a_witnessed_bitwise_operation_computes_at_every_width() {
    for bits in [40usize, 96, 200, 253, 254, 320] {
        for kind in [
            BinaryArithOpKind::And,
            BinaryArithOpKind::Or,
            BinaryArithOpKind::Xor,
        ] {
            let ssa = main_program(
                &[Type::int(64), Type::int(64)],
                &[Type::int(64)],
                move |e, params| {
                    let lhs = e.cast_to(CastTarget::Int(bits), params[0]);
                    let rhs = e.cast_to(CastTarget::Int(bits), params[1]);
                    let answer = e.bin(kind, lhs, rhs);
                    vec![e.cast_to(CastTarget::Int(64), answer)]
                },
            );
            let compiled = Compiled::new(ssa)
                .unwrap_or_else(|error| panic!("an int{bits} {kind:?} is refused: {error}"));

            // Operands with bits in both halves of every limb, so a decomposition that dropped or
            // mis-placed one is visible in the answer rather than only in the top limb.
            let (x, y) = (0xF0F0_F0F0_0F0F_0F0Fu128, 0x00FF_00FF_FF00_FF00u128);
            // The operands are narrowed to `bits` on the way in and the answer widened back, so
            // below 64 the expected value is the operation on the *truncated* pair.
            let mask = if bits >= 64 {
                u128::from(u64::MAX)
            } else {
                (1u128 << bits) - 1
            };
            let (a, c) = (x & mask, y & mask);
            let want = match kind {
                BinaryArithOpKind::And => a & c,
                BinaryArithOpKind::Or => a | c,
                _ => a ^ c,
            };
            let verdict = compiled.run(&input_block(&[
                &IntBits::from_u128(64, x),
                &IntBits::from_u128(64, y),
                &IntBits::from_u128(64, want),
            ]));
            assert!(verdict.is_accepted(), "int{bits} {kind:?}: {verdict:?}");
        }
    }
}

/// One witnessed operand and one **pure** one, which take different halves of the decomposition.
///
/// `decompose_into_spread_limbs` branches on witness-ness: a witnessed operand has its high limbs
/// written as columns and its lowest derived, while a pure one is read straight out of the hint.
/// A mixed pair is the ordinary shape of `x & MASK`, and it is what puts a **wide** value through
/// the pure branch while a constraint still reads the answer.
#[test]
fn a_wide_bitwise_operation_takes_one_pure_operand() {
    for bits in [96usize, 200, 254, 320] {
        let input = 0xDEAD_BEEF_0BAD_F00Du64;
        // A bit in the **top** limb, placed on the witnessed operand with an `or`, which carries
        // nothing between limbs and so sets that bit and no other. Without it every limb of the
        // operand above the first is zero, and a decomposition that dropped one would be invisible.
        let high = BigUint::from(1u8) << (bits - 8);
        let mask = &high + BigUint::from(0x0F0F_0F0Fu32);
        let expected = &high + BigUint::from(input & 0x0F0F_0F0F);

        let (high_bit, mask_bits, want_bits) = (high, mask, expected);
        let ssa = main_program(&[Type::int(64)], &[Type::int(64)], move |e, params| {
            let widened = e.cast_to(CastTarget::Int(bits), params[0]);
            let high = e.int_const(IntBits::from_biguint(bits, &high_bit));
            let witnessed = e.bin(BinaryArithOpKind::Or, widened, high);
            // The mask meets the operand in the top limb and in the bottom, so a decomposition
            // that mislaid either end shows up in the answer.
            let mask = e.int_const(IntBits::from_biguint(bits, &mask_bits));
            let masked = e.bin(BinaryArithOpKind::And, witnessed, mask);
            // Compared **whole**. Narrowing the result to the return width would read its lowest
            // limb alone, and every limb above it could be dropped unseen.
            let want = e.int_const(IntBits::from_biguint(bits, &want_bits));
            let same = e.eq(masked, want);
            vec![e.cast_to(CastTarget::Int(64), same)]
        });
        let compiled = Compiled::new(ssa)
            .unwrap_or_else(|error| panic!("an int{bits} mixed bitwise is refused: {error}"));

        let verdict = compiled.run(&input_block(&[
            &IntBits::from_u128(64, u128::from(input)),
            &IntBits::from_u128(64, 1),
        ]));
        assert!(verdict.is_accepted(), "int{bits} x & MASK: {verdict:?}");
    }
}

/// A witnessed complement computes at every width, by the same limb-wise argument.
///
/// `!x ^ y` is the shape for two reasons. It needs no constant wider than the return — the low 64
/// bits of a wide complement are the complement of the low 64 bits, which is what the declared
/// return reads. And the answer **depends on the complement**: `!!x` is folded straight back to `x`
/// by the simplifier's `~~x -> x` rewrite, in a phase long before any pass this exercises, so a
/// double complement would assert only that a widening cast and a truncating cast round trip.
///
/// **253 and 254 are the pair that matters.** At 253 the operand is one field element and the
/// complement really is `2^bits - 1 - value` there; at 254 it is limbs, and the complement is one
/// per limb at that limb's own width. The same program either side of the threshold.
#[test]
fn a_witnessed_complement_computes_at_every_width() {
    for bits in [96usize, 200, 253, 254, 320] {
        let ssa = main_program(
            &[Type::int(64), Type::int(64)],
            &[Type::int(64)],
            move |e, params| {
                let wide = e.cast_to(CastTarget::Int(bits), params[0]);
                let other = e.cast_to(CastTarget::Int(bits), params[1]);
                let flipped = e.not(wide);
                let answer = e.bin(BinaryArithOpKind::Xor, flipped, other);
                vec![e.cast_to(CastTarget::Int(64), answer)]
            },
        );
        let compiled = Compiled::new(ssa)
            .unwrap_or_else(|error| panic!("an int{bits} complement is refused: {error}"));

        let (x, y) = (0xDEAD_BEEF_0BAD_F00Du64, 0xF0F0_F0F0_0F0F_0F0Fu64);
        let want = !x ^ y;
        let verdict = compiled.run(&input_block(&[
            &IntBits::from_u128(64, u128::from(x)),
            &IntBits::from_u128(64, u128::from(y)),
            &IntBits::from_u128(64, u128::from(want)),
        ]));
        assert!(verdict.is_accepted(), "int{bits} !x ^ y: {verdict:?}");
    }
}

/// A complement **outside** the witness domain, at the double lane and the wide one.
///
/// A pure complement is not rewritten into `(2^bits - 1) - value` in the field: the witness
/// lowering leaves it alone, exactly as it already left a pure `and` alone. Nothing in width
/// validation binds it either, at any width, because bit `i` of the answer depends on bit `i` of
/// the operand alone. The `xor` against a witnessed operand is what puts the answer somewhere a
/// constraint reads it: a pure value on its own is a hint, and a wrong hint that nothing checks is
/// not a failing test.
///
/// **What this does not reach is the opcodes.** A pure value that is not a constant cannot be
/// built from an entry point — witness-ness is inferred, and every non-constant value here is
/// tainted by it — so the operand is a constant and the complement is folded. That makes this a
/// test of the width rule and of a `2^bits`-wide complement through the constant evaluators;
/// `NotInt`, `NotInt128` and `NotIntn` are held to the semantic model in the VM crate instead, by
/// `the_complement_opcode_agrees_with_the_model` and `the_intn_complement_agrees_with_the_model`.
#[test]
fn a_complement_outside_the_witness_domain_computes_at_every_width() {
    /// Below `2^64`, so the wide constant's low word is `PURE` and its complement's is `!PURE`.
    const PURE: u64 = 0xDEAD;

    for bits in [96usize, 200, 253, 254, 320] {
        let ssa = main_program(&[Type::int(64)], &[Type::int(64)], move |e, params| {
            let pure = e.int_const(IntBits::from_biguint(bits, &BigUint::from(PURE)));
            let flipped = e.not(pure);
            let witnessed = e.cast_to(CastTarget::Int(bits), params[0]);
            let answer = e.bin(BinaryArithOpKind::Xor, flipped, witnessed);
            vec![e.cast_to(CastTarget::Int(64), answer)]
        });
        let compiled = Compiled::new(ssa)
            .unwrap_or_else(|error| panic!("an int{bits} pure complement is refused: {error}"));

        let x = 0xDEAD_BEEF_0BAD_F00Du64;
        let want = !PURE ^ x;
        let verdict = compiled.run(&input_block(&[
            &IntBits::from_u128(64, u128::from(x)),
            &IntBits::from_u128(64, u128::from(want)),
        ]));
        assert!(verdict.is_accepted(), "int{bits} !C ^ x: {verdict:?}");
    }
}

/// Every signed operation is capped at one host word, which is the frontier unit 13 moves.
#[test]
fn a_signed_operation_above_sixty_four_bits_is_refused() {
    contains_all(
        &refusal_for(BinaryArithOpKind::SShl, 128, 128),
        &[
            "error: a signed int128 left shift is not supported",
            "= note: a signed operand is read as two's complement in one integer cell",
        ],
    );
}

/// `main(a: int(from)) -> int(to) { a as int(to) }`, the narrowing written as a bare cast.
fn program_casting_down(from: usize, to: usize) -> HLSSA {
    main_program(&[Type::int(from)], &[Type::int(to)], move |e, params| {
        vec![e.cast_to(CastTarget::Int(to), params[0])]
    })
}

/// A bare narrowing cast of a witnessed value **truncates**, at every width one element carries.
///
/// A witness is a field element and a `Cast` carries it through unchanged, so something else has
/// to discard the bits above the target. `LowerWitnessNarrowingCast` puts the bit window that does
/// in front of it.
#[test]
fn a_bare_narrowing_cast_truncates_at_every_width() {
    for bits in [64usize, 96, 128, 129, 200, first_width_past_the_field() - 1] {
        let compiled = Compiled::new(program_casting_down(bits, 32))
            .unwrap_or_else(|error| panic!("an int{bits} narrowing: {error}"));

        // Bits far above the target, so what is checked is the truncation and not the value.
        let source = IntBits::from_u128(bits, (5u128 << 40) + 7);
        let truncated = IntBits::from_u128(32, 7);
        let verdict = compiled.run(&input_block(&[&source, &truncated]));
        assert!(
            verdict.is_accepted(),
            "int{bits} did not truncate to 7: {verdict:?}"
        );

        // And the answer is constrained rather than merely computed: declaring a different one is
        // refused by the wrapper's own return check.
        let wrong = IntBits::from_u128(32, 8);
        let refused = compiled.run(&input_block(&[&source, &wrong]));
        assert!(
            matches!(
                refused,
                Verdict::Trapped {
                    in_return_check: true,
                    ..
                }
            ),
            "int{bits} accepted a wrong truncation: {refused:?}"
        );
    }
}

/// A narrowing **between two widths the representation carries as limbs**, where the target is not
/// a whole number of limbs.
///
/// The limb list a narrowing builds is the **target's**, so its top limb is narrower than the
/// source limb it comes from. Cutting that limb with a `BitRange` gives the right bits and the
/// wrong type (as a window keeps its source's width) and the limb then meets an operand split at
/// its true width with nothing able to pair the two.
///
/// Limb-aligned targets never cut a limb at all, which is why 256 and 320 compile either way and
/// the widths here are deliberately not multiples of the limb.
#[test]
fn a_narrowing_between_two_widths_held_as_limbs() {
    let injective = first_width_past_the_field() - 1;
    let source = BigUint::from(1u8) << 200;
    for (from, to) in [(384usize, 300usize), (320, 254), (384, 319), (320, 256)] {
        let ssa = main_program(
            &[Type::int(injective)],
            &[Type::int(1)],
            move |e, params| {
                let widened = e.cast_to(CastTarget::Int(from), params[0]);
                let narrowed = e.cast_to(CastTarget::Int(to), widened);
                // The whole target width, so a limb dropped or mistyped is visible rather than hidden
                // below 64 bits.
                let want = e.int_const(IntBits::from_biguint(to, &(BigUint::from(1u8) << 200)));
                vec![e.cmp(narrowed, want, CmpKind::Eq)]
            },
        );
        let compiled = Compiled::new(ssa)
            .unwrap_or_else(|error| panic!("int{from} narrowed to int{to}: {error}"));

        let param = IntBits::from_biguint(injective, &source);
        let verdict = compiled.run(&input_block(&[&param, &IntBits::from_u128(1, 1)]));
        assert!(
            verdict.is_accepted(),
            "int{from} narrowed to int{to}: {verdict:?}"
        );

        // And it can go red: the same program asked for the opposite answer.
        let verdict = compiled.run(&input_block(&[&param, &IntBits::from_u128(1, 0)]));
        assert!(
            !verdict.is_accepted(),
            "int{from} narrowed to int{to}: a wrong answer was accepted"
        );
    }
}

/// And a value the range already puts inside the target costs no window.
///
/// The pair Noir's frontend emits — `bit_range(v, 0, n)` and then the cast — is the shape every
/// compiled program takes, and the window's own range transfer bounds its result by `2^n`. So the
/// rewrite must decline there, or every narrowing in the corpus would pay for a second window.
/// Stated as a comparison rather than a count so it survives any change that moves both.
#[test]
fn a_narrowing_cast_costs_no_window_where_the_value_is_already_inside_it() {
    let source = IntBits::from_u128(64, (5u128 << 40) + 7);
    let truncated = IntBits::from_u128(32, 7);

    let columns = |ssa| {
        let compiled = Compiled::new(ssa).expect("a narrowing compiles");
        let verdict = compiled.run(&input_block(&[&source, &truncated]));
        verdict
            .witness()
            .unwrap_or_else(|| panic!("the narrowing ran: {verdict:?}"))
            .len()
    };

    let bare = columns(program_casting_down(64, 32));
    let windowed = columns(main_program(
        &[Type::int(64)],
        &[Type::int(32)],
        |e, params| {
            let window = e.bit_range(params[0], 0, 32);
            vec![e.cast_to(CastTarget::Int(32), window)]
        },
    ));

    assert_eq!(
        bare, windowed,
        "the pair paid for a second window: {windowed} columns against {bare}"
    );
}

/// A widening cast is untouched by the rule: only a narrowing one has bits to discard.
#[test]
fn a_witnessed_widening_cast_is_not_refused() {
    let _ =
        Compiled::new(program_casting_down(64, 200)).expect("a widening cast is not a narrowing");
    let _ = Compiled::new(program_casting_down(200, 200))
        .expect("a cast to its own width narrows nothing");
}

// THE WIDE BYTECODE LANE
// ================================================================================================

/// A width above the two narrow lanes lowers to `_intn` opcodes rather than to a panic.
///
/// The `vm` conformance sweep says the wide bodies compute the right answers and the interpreter's
/// own round-trip test says the encoding survives dispatch. Neither says anything about _which_
/// opcode a width picks, which is the whole of the codegen half: a width that fell through to the
/// double lane would compute modulo the wrong power of two and every one of those tests would
/// still pass.
#[test]
fn a_wide_width_reaches_the_wide_lane() {
    let ssa = main_program(
        &[Type::int(256), Type::int(256)],
        &[Type::int(256)],
        |e, params| {
            let sum = e.bin(BinaryArithOpKind::UAdd, params[0], params[1]);
            let product = e.bin(BinaryArithOpKind::UMul, sum, params[1]);
            vec![e.bin(BinaryArithOpKind::And, product, params[0])]
        },
    );
    let listing = bytecode_listing(&ssa);

    for op in ["add_intn", "mul_intn", "and_intn"] {
        assert!(listing.contains(op), "no {op} in:\n{listing}");
    }
    // The width immediate is what tells the opcode how many cells each operand covers, so an
    // opcode that carried the wrong one would read the wrong cells while still disassembling.
    assert!(
        listing.contains("add_intn 256 "),
        "add_intn did not carry its width:\n{listing}"
    );
}

/// The three lanes are chosen by width, and each keeps its own opcodes.
///
/// One test rather than three because what is being checked is the partition: the point of failure
/// worth catching is a boundary that moved, and a boundary shows up as two widths agreeing that
/// should not.
#[test]
fn each_width_picks_its_own_lane() {
    for (bits, expected, forbidden) in [
        (64usize, "add_int ", "add_int128"),
        (128, "add_int128", "add_intn"),
        (129, "add_intn", "add_int128"),
        // Two cells, and so the double lane: its six width-dependent opcodes carry a width, so it
        // spans `65..=128` rather than the single width they were first written for.
        (96, "add_int128", "add_intn"),
        (65, "add_int128", "add_intn"),
    ] {
        let ssa = main_program(
            &[Type::int(bits), Type::int(bits)],
            &[Type::int(bits)],
            |e, params| vec![e.bin(BinaryArithOpKind::UAdd, params[0], params[1])],
        );
        let listing = bytecode_listing(&ssa);
        assert!(
            listing.contains(expected),
            "int{bits} did not reach {expected}:\n{listing}"
        );
        assert!(
            !listing.contains(forbidden),
            "int{bits} reached {forbidden}:\n{listing}"
        );
    }
}

/// Comparison, equality and the width cast reach the wide lane too.
#[test]
fn the_wide_lane_covers_comparison_and_the_width_cast() {
    let ssa = main_program(
        &[Type::int(320), Type::int(320)],
        &[Type::int(64)],
        |e, params| {
            // The comparison's value goes nowhere, which costs it nothing: `bytecode_listing`
            // generates code without running the passes, so there is no elimination to survive.
            let _less = e.cmp(params[0], params[1], CmpKind::ULt);
            vec![e.cast_to(CastTarget::Int(64), params[0])]
        },
    );
    let listing = bytecode_listing(&ssa);
    assert!(listing.contains("ult_intn"), "no ult_intn in:\n{listing}");
    assert!(listing.contains("cast_intn"), "no cast_intn in:\n{listing}");
}

/// Both directions of the field boundary reach the wide lane too.
///
/// The listing is what says which opcode a width picks, and it is the only thing that does: the
/// interpreter's own tests exercise the bodies directly, so a width that fell through to the
/// double lane would read two cells of a four-cell value and every one of them would still pass.
#[test]
fn the_wide_lane_covers_both_directions_of_the_field_boundary() {
    let ssa = main_program(&[Type::int(200)], &[Type::int(200)], |e, params| {
        let element = e.cast_to_field(params[0]);
        vec![e.cast_to(CastTarget::Int(200), element)]
    });
    let listing = bytecode_listing(&ssa);

    for op in ["cast_intn_to_field 200 ", "cast_field_to_intn 200 "] {
        assert!(listing.contains(op), "no {op:?} in:\n{listing}");
    }
}

/// An operand narrower than the result is an **ICE** in codegen, below the funnel that refuses it.
///
/// A shift is the one operation whose operands the model lets differ: `int_semantics::eval` reads
/// the amount at its own declared width and reduces its magnitude against the value's. Bytecode
/// cannot honour that above the cell lane, and `width_validation`'s `shift_amount_extent` rule is
/// what a program meets — see [`a_shift_by_a_narrower_literal_amount_is_refused`]. This drives
/// codegen directly, `bytecode_listing` running no passes, so it reaches the backstop underneath.
#[test]
#[should_panic(expected = "does not cover the 4 frame cell(s)")]
fn a_wide_shift_by_a_narrower_amount_is_an_ice_in_codegen() {
    let ssa = main_program(
        &[Type::int(256), Type::int(32)],
        &[Type::int(256)],
        |e, params| vec![e.bin(BinaryArithOpKind::UShl, params[0], params[1])],
    );
    let _ = bytecode_listing(&ssa);
}

/// The same backstop one lane down, where the stray cell is the result's own.
#[test]
#[should_panic(expected = "does not cover the 2 frame cell(s)")]
fn a_double_lane_shift_by_a_narrower_amount_is_an_ice_in_codegen() {
    let ssa = main_program(
        &[Type::int(96), Type::int(32)],
        &[Type::int(96)],
        |e, params| vec![e.bin(BinaryArithOpKind::UShl, params[0], params[1])],
    );
    let _ = bytecode_listing(&ssa);
}

/// `main(a: int(bits)) -> int(bits / 2) { unspread(a).odd }`, the inverse of a spread.
fn program_unspreading(bits: usize) -> HLSSA {
    main_program(
        &[Type::int(bits)],
        &[Type::int(bits / 2)],
        move |e, params| {
            let half = u8::try_from(bits / 2).expect("a spread half is at most 64 bits");
            let (odd, _even) = e.unspread(params[0], half);
            vec![odd]
        },
    )
}

/// An unspread wider than the spread it inverts is refused by the bytecode arm itself.
///
/// The type rule admits an operand up to 128 bits, and the opcode reads 64. Nothing emits a wider
/// one — every spread the bitwise lowering makes is a half-limb or narrower — so this is a guard on
/// a producer that does not exist yet, and drives codegen directly to reach it.
#[test]
#[should_panic(expected = "Unspread bytecode lowering for integer widths > 64 bits")]
fn an_unspread_past_what_its_opcode_reads_is_refused_in_codegen() {
    let _ = bytecode_listing(&program_unspreading(128));
}

/// The widest unspread the opcode reads still lowers to it.
#[test]
fn an_unspread_the_opcode_reads_lowers_to_it() {
    let listing = bytecode_listing(&program_unspreading(64));
    assert!(
        listing.contains("unspread_u64_to_u32"),
        "no unspread in:\n{listing}"
    );
}

/// The **bytecode** cell lane keeps taking a narrower amount, which is what makes the bound above
/// a cell count rather than a width.
///
/// Every width the cell lane holds is one frame cell, so an `int8` amount and an `int32` one are
/// read from the same single cell and `shift_amount` reduces whichever magnitude it finds.
#[test]
fn a_narrow_shift_by_a_narrower_amount_still_lowers_to_bytecode() {
    let ssa = main_program(
        &[Type::int(32), Type::int(8)],
        &[Type::int(32)],
        |e, params| vec![e.bin(BinaryArithOpKind::UShl, params[0], params[1])],
    );
    let listing = bytecode_listing(&ssa);
    assert!(listing.contains("shl_int "), "no shl_int in:\n{listing}");
}

/// The rule a program meets: a shift by an amount laid out in fewer cells than the operation reads.
///
/// A **literal** amount is the reachable shape and it is the one that used to pass the funnel and
/// ICE in codegen. The bound is a cell count rather than a width, so an amount that covers the same
/// cells still compiles.
#[test]
fn a_shift_by_a_narrower_literal_amount_is_refused() {
    let shift_by = |amount_bits: usize| {
        main_program(&[Type::int(128)], &[Type::int(128)], move |e, params| {
            let amount = e.int_const(IntBits::from_u128(amount_bits, 8));
            vec![e.bin(BinaryArithOpKind::UShl, params[0], amount)]
        })
    };

    let Err(error) = Compiled::new(shift_by(64)) else {
        panic!("an int128 shift by an int64 literal is refused rather than compiled");
    };
    assert!(
        matches!(error, DriverError::Refused(_)),
        "a shape no lowering builds is a refusal rather than a crash: {error}"
    );
    contains_all(
        &error.to_string(),
        &[
            "error: an int128 left shift by an int64 amount is not supported",
            "= note: an integer opcode addresses both operands at the result's own cell count",
        ],
    );

    // The same cells, so nothing is read past and the program compiles.
    let _ = Compiled::new(shift_by(128)).expect("an amount covering the operation's cells lowers");
}

// THE FIELD BOUNDARY
// ================================================================================================

/// The first width whose values are not all distinct field elements.
///
/// `2^bits <= p` is what makes an `Int -> Field` cast injective, and the modulus needs exactly this
/// many bits to be written down, so this is the first width at which two integers share an element.
///
/// Derived from the field's own bit size rather than the compiler's `widest_injective_int_bits` to
/// ensure that they agree,
fn first_width_past_the_field() -> usize {
    FieldConfig::bn254().field_bit_size() as usize
}

/// An oversized cast is a refusal of the program.
#[test]
fn an_oversized_cast_to_field_is_refused_with_a_diagnostic() {
    let bits = first_width_past_the_field();
    let source = format!("fn main(wide: int{bits}) -> Field {{\n    wide as Field\n}}\n");
    let mut file = tempfile::NamedTempFile::new().unwrap();
    std::io::Write::write_all(&mut file, source.as_bytes()).unwrap();
    std::io::Write::flush(&mut file).unwrap();

    let mut ssa = main_program(&[Type::int(bits)], &[Type::field()], |e, params| {
        vec![e.cast_to_field(params[0])]
    });
    locate_the_cast(
        &mut ssa,
        SourceLocation::new(
            file.path().to_string_lossy().as_ref(),
            SourcePosition::new(2, 5),
            SourcePosition::new(2, 18),
        ),
    );

    let Err(error) = Compiled::new(ssa) else {
        panic!("a cast the field cannot carry is refused");
    };
    assert!(
        matches!(error, DriverError::Refused(_)),
        "a program the compiler declines is a refusal rather than a crash: {error}"
    );

    let rendered = error.to_string();
    for expected in [
        &format!("error: an int{bits} value cannot be cast to Field"),
        "    wide as Field",
        "^^^^^^^^^^^^^",
        &format!("2^{bits} exceeds the field modulus"),
        "= note: the widest integer this field carries injectively is int",
    ] {
        assert!(
            rendered.contains(expected),
            "no {expected:?} in:\n{rendered}"
        );
    }
}

/// A constant operand is the case that decides where validation runs.
///
/// The constant folder answers an `Int -> Field` cast whenever the _value_ fits the modulus, so a
/// small `int254` constant folds to a field element and the cast stops existing. Validating after
/// any folding would therefore let this program through while refusing the same cast on a
/// parameter, which would make the rule a property of the optimizer rather than of the language.
#[test]
fn an_oversized_cast_of_a_small_constant_is_still_refused() {
    let bits = first_width_past_the_field();
    let ssa = main_program(&[], &[Type::field()], |e, _| {
        let value = e.int_const(IntBits::from_u128(bits, 5));
        vec![e.cast_to_field(value)]
    });

    let Err(DriverError::Refused(diagnostics)) = Compiled::new(ssa) else {
        panic!("a foldable cast is refused on what the program says, not on what survives folding");
    };
    assert_eq!(diagnostics.len(), 1, "{diagnostics:?}");
}

/// A cast at the widest width the field _does_ carry passes validation.
///
/// Stopped at [`Driver::make_struct_access_static`], which is the phase validation runs in.
#[test]
fn the_widest_injective_width_passes_validation() {
    let ssa = main_program(
        &[Type::int(first_width_past_the_field() - 1)],
        &[Type::field()],
        |e, params| vec![e.cast_to_field(params[0])],
    );

    assert!(
        validate_only(ssa).is_ok(),
        "validation refused the widest injective width"
    );
}

/// Every integer parameter of `main` is range-checked to its own width where it is written to a
/// witness column, so a wide entry point meets that decomposition before anything else.
///
/// It is a check on a **field element**, which is why it reaches as far as it does: the widest
/// integer the modulus carries injectively is a parameter like any other, and `main` needs no wide
/// calling convention to take one.
#[test]
fn the_widest_injective_width_is_an_entry_point_parameter() {
    let injective = first_width_past_the_field() - 1;

    for width in [129usize, 200, injective] {
        let ssa = main_program(&[Type::int(width)], &[Type::field()], |e, params| {
            vec![e.cast_to_field(params[0])]
        });
        let compiled = Compiled::new(ssa)
            .unwrap_or_else(|error| panic!("int{width} is not an entry-point parameter: {error}"));

        let value = IntBits::from_biguint(
            width,
            &((BigUint::from(1u8) << (width - 8)) + BigUint::from(7u8)),
        );
        let verdict = compiled.run(&input_block(&[&value, &value]));
        assert!(verdict.is_accepted(), "int{width} parameter: {verdict:?}");
    }
}

/// And the check it emits bites: a value above the declared width is refused at proving time.
///
/// This is the half `the_widest_injective_width_is_an_entry_point_parameter` cannot show. An honest
/// witness satisfies a decomposition that drops its top chunks just as happily as one that does
/// not, so what says the chunks are all there is a **dishonest** input.
#[test]
fn a_wide_entry_point_parameter_is_held_to_its_width() {
    for width in [129usize, 200] {
        let ssa = main_program(&[Type::int(width)], &[Type::field()], |e, params| {
            vec![e.cast_to_field(params[0])]
        });
        let compiled = Compiled::new(ssa).expect("a wide parameter compiles");

        // One bit above the declared width, which the entry point's range check is the only thing
        // standing between and the rest of the program.
        let past = Field::from(BigUint::from(1u8) << width);
        let verdict = compiled.run(&[
            InputValueOrdered::Field(past),
            InputValueOrdered::Field(past),
        ]);
        assert!(
            verdict.is_refusal(),
            "int{width} accepted a value above its width: {verdict:?}"
        );
    }
}

/// A bound no element can exceed is refused, because there is nothing left to decompose.
#[test]
fn a_range_check_past_what_the_field_carries_is_refused() {
    let bits = first_width_past_the_field();
    let ssa = main_program(&[Type::field()], &[Type::field()], |e, params| {
        e.rangecheck(params[0], bits);
        vec![params[0]]
    });

    let Err(error) = Compiled::new(ssa) else {
        panic!("a bound above the modulus is refused rather than compiled");
    };
    let DriverError::Refused(diagnostics) = &error else {
        panic!("a bound no lowering builds is a refusal rather than a crash: {error}");
    };
    assert_eq!(
        diagnostics.iter().map(|d| d.message()).collect::<Vec<_>>(),
        [&format!("a range check to {bits} bits is not supported")],
        "{error}"
    );
}

/// The `int320` round trip: widen from a narrow witness, carry it through selects, narrow back,
/// and compare against where it started.
///
/// The end-to-end half of the representation — the witness satisfies the R1CS and the value that
/// comes back is the one that went in. A select is an instruction and builds no phi, so this
/// carries the value without crossing a block boundary;
/// [`a_wide_value_carried_through_a_phi_comes_back`] is the shape that does.
#[test]
fn a_wide_value_survives_being_carried_and_comes_back() {
    let wide = 320usize;
    let ssa = main_program(&[Type::int(64)], &[Type::int(1)], |e, params| {
        let start = e.cast_to(CastTarget::Int(wide), params[0]);
        let no = e.int_const(IntBits::zero(1));
        let mut carried = start;
        for _ in 0..4 {
            carried = e.select(no, start, carried);
        }
        let narrowed = e.cast_to(CastTarget::Int(64), carried);
        vec![e.cmp(narrowed, params[0], CmpKind::Eq)]
    });

    let compiled =
        Compiled::new(ssa).unwrap_or_else(|error| panic!("an int{wide} round trip: {error}"));

    let value = IntBits::from_u128(64, (1u128 << 40) + 5);
    let one = IntBits::from_u128(1, 1);
    let verdict = compiled.run(&input_block(&[&value, &one]));
    assert!(verdict.is_accepted(), "int{wide} round trip: {verdict:?}");
}

/// The same value carried across a **block boundary**, which is a real phi rather than a select.
#[test]
fn a_wide_value_carried_through_a_phi_comes_back() {
    let wide = 320usize;
    let ssa = main_program(&[Type::int(128)], &[Type::int(64)], move |e, params| {
        let start = e.cast_to(CastTarget::Int(wide), params[0]);
        let counter = e.int_const(IntBits::zero(32));
        let limit = e.int_const(IntBits::from_u128(32, 3));
        let carried = e.build_loop(
            vec![
                (counter, Type::int(32)),
                (start, Type::witness_of(Type::int(wide))),
            ],
            |b, parameters| b.ult(parameters[0], limit),
            |b, parameters| {
                let one = b.int_const(IntBits::one(32));
                let next = b.uadd(parameters[0], one);
                // The wide value is passed straight back round, so the only thing the loop does to
                // it is carry it through the header's parameter list three times over.
                vec![next, parameters[1]]
            },
        );
        vec![e.cast_to(CastTarget::Int(64), carried[1])]
    });

    let compiled = Compiled::new(ssa).unwrap_or_else(|error| panic!("an int{wide} loop: {error}"));

    let value = IntBits::from_u128(128, (1u128 << 100) + 9);
    let low = IntBits::from_u128(64, 9);
    let verdict = compiled.run(&input_block(&[&value, &low]));
    assert!(
        verdict.is_accepted(),
        "int{wide} through a phi: {verdict:?}"
    );
}

/// A wide equality **assertion**, which is one assertion per limb rather than a conjunction.
///
/// The pair that matters differs only in a **high** limb: an assertion built from the low limbs
/// alone would call them equal.
#[test]
fn a_wide_equality_assertion_reads_every_limb() {
    let wide = 320usize;
    let ssa = main_program(
        &[Type::int(128), Type::int(128)],
        &[Type::int(64)],
        move |e, params| {
            let lhs = e.cast_to(CastTarget::Int(wide), params[0]);
            let rhs = e.cast_to(CastTarget::Int(wide), params[1]);
            e.emit(OpCode::AssertCmp {
                kind: CmpKind::Eq,
                lhs,
                rhs,
            });
            vec![e.cast_to(CastTarget::Int(64), lhs)]
        },
    );
    let compiled = Compiled::new(ssa).expect("a wide assertion compiles");

    let low = IntBits::from_u128(64, 9);
    let value = IntBits::from_u128(128, (1u128 << 100) + 9);
    let equal = compiled.run(&input_block(&[&value, &value, &low]));
    assert!(equal.is_accepted(), "int{wide} == itself: {equal:?}");

    // The same low limb, a different high one.
    let other = IntBits::from_u128(128, (1u128 << 101) + 9);
    let differing = compiled.run(&input_block(&[&value, &other, &low]));
    assert!(
        differing.is_refusal(),
        "a difference above the low limb went unasserted: {differing:?}"
    );
}

/// A wide value compared against a wide **pure** constant, which the representation never mapped.
///
/// `check_widths` makes both operands of a comparison the same width and a mixed pure/witness pair
/// is legal, so the constant arrives as one value beside a five-limb one. Zipping those two lists
/// would compare the low limb and pass silently over the rest — the answer would be right whenever
/// the low limbs decide it, which is most of the time and all of an honest sweep.
#[test]
fn a_wide_value_compared_against_a_wide_constant_reads_every_limb() {
    let wide = 320usize;

    // The two agree in their low limb and differ above it, which is the pair a truncated zip calls
    // equal.
    let low = IntBits::from_u128(64, 5);
    for (constant, expected) in [(5u128, 1u64), ((1u128 << 70) + 5, 0)] {
        let ssa = main_program(&[Type::int(64)], &[Type::int(1)], |e, params| {
            let widened = e.cast_to(CastTarget::Int(wide), params[0]);
            let other = e.int_const(IntBits::from_u128(wide, constant));
            vec![e.cmp(widened, other, CmpKind::Eq)]
        });
        let compiled = Compiled::new(ssa).expect("a wide comparison against a constant compiles");

        let answer = IntBits::from_u128(1, u128::from(expected));
        let verdict = compiled.run(&input_block(&[&low, &answer]));
        assert!(
            verdict.is_accepted(),
            "int{wide} == {constant} should be {expected}: {verdict:?}"
        );
    }
}

/// A constant the field cannot carry is carried anyway, because the limbs carry it.
///
/// This is what the representation buys that a single element cannot: `2^299` has no field element
/// — two integers that far apart share one — but each of its limbs does, so a constant at such a
/// width splits into limb constants and never needs an element of its own. `hlssa_to_r1cs::of_int`
/// refuses a magnitude at or above the modulus, and this is the shape that would have met it.
///
/// 254 is the first width the modulus does not carry and 300 is well past it; both answer.
#[test]
fn a_constant_the_field_cannot_carry_still_selects_and_narrows() {
    for bits in [254usize, 255, 300] {
        let ssa = main_program(&[Type::int(1)], &[Type::int(64)], |e, params| {
            let big = e.emit_constant(Constant::Int(IntBits::from_biguint(
                bits,
                &(BigUint::from(1u8) << (bits - 1)),
            )));
            let small = e.int_const(IntBits::from_u128(bits, 3));
            let chosen = e.select(params[0], big, small);
            vec![e.cast_to(CastTarget::Int(64), chosen)]
        });
        let compiled =
            Compiled::new(ssa).unwrap_or_else(|error| panic!("an int{bits} constant: {error}"));

        // `2^(bits-1)` has no bits below 64, so its low word is zero; the other arm is 3.
        let yes = IntBits::from_u128(1, 1);
        let no = IntBits::from_u128(1, 0);
        let zero = IntBits::from_u128(64, 0);
        let three = IntBits::from_u128(64, 3);

        let taken = compiled.run(&input_block(&[&yes, &zero]));
        assert!(taken.is_accepted(), "int{bits} constant taken: {taken:?}");
        let other = compiled.run(&input_block(&[&no, &three]));
        assert!(
            other.is_accepted(),
            "int{bits} constant passed over: {other:?}"
        );
    }
}

/// Every witness column a wide value occupies is pinned by a constraint.
///
/// The one test an honest witness cannot stand in for. A limb the reconstruction constraint does
/// not reach is free, and that is invisible to every run that only feeds correct inputs — the
/// prover never exercises the freedom it was left. Adding one to each column in turn is what finds
/// it: `2^h` is invertible mod `p`, so an unpinned limb lets a prover solve for any value at all.
///
/// **The source is wider than one limb and the result narrower**, which is what makes the test
/// bite. A `int64` source has one limb, and a result that reads every limb pins them through its
/// own assertion whether or not the decomposition constrains anything — either shape passes with
/// the reconstruction removed.
///
/// It also contains no equality. `lower_eq` witnesses the inverse of the difference as a hint, and
/// that hint is genuinely free whenever the difference is zero, so an honest `a == a` has an
/// unconstrained column at every width, this representation or not.
#[test]
fn every_column_of_a_wide_value_is_pinned() {
    let wide = 320usize;
    let ssa = main_program(&[Type::int(128)], &[Type::int(64)], |e, params| {
        let widened = e.cast_to(CastTarget::Int(wide), params[0]);
        vec![e.cast_to(CastTarget::Int(64), widened)]
    });
    let compiled = Compiled::new(ssa).expect("a wide round trip compiles");

    let value = IntBits::from_u128(128, (1u128 << 100) + 5);
    let low = IntBits::from_u128(64, 5);
    let verdict = compiled.run(&input_block(&[&value, &low]));
    let witness = verdict
        .witness()
        .unwrap_or_else(|| panic!("an int{wide} round trip: {verdict:?}"))
        .to_vec();

    for column in 0..witness.len() {
        let mut perturbed = witness.clone();
        perturbed[column] += Field::from(1u64);
        assert!(
            compiled.first_unsatisfied_constraint(&perturbed).is_some(),
            "column {column} of {} is unconstrained — adding one to it left every constraint \
             satisfied",
            witness.len()
        );
    }
}

/// Every column a wide **sequence** contributes is pinned, which no honest witness can show.
///
/// The scalar test above says the representation constrains its own limbs. This says the same of a
/// sequence: entering one mints `k` witness columns **per element**, each with a range check of its
/// own, and reading one back pins `k` lookups rather than a decomposition.
///
/// The element is derived from a **parameter** so the columns exist at all, and the sequence is
/// read at a **witness** index so the lookups are real. The declared return is narrower than the
/// element, so the program does not pin the high limbs through its own assertion.
///
/// **What this does not say:** a column that nothing pins is also a column nothing reads, so it is
/// eliminated rather than left free: delete one of the `k` lookups and the witness comes back
/// shorter with every column in it still pinned. A pinning test can only find a free column that
/// survives DCE.
#[test]
fn every_column_of_a_wide_sequence_is_pinned() {
    let wide = 320usize;
    let ssa = main_program(
        &[Type::int(253), Type::int(32)],
        &[Type::int(64)],
        move |e, params| {
            let element = e.cast_to(CastTarget::Int(wide), params[0]);
            let other = e.int_const(IntBits::from_biguint(
                wide,
                &((BigUint::from(1u8) << (wide - 8)) + BigUint::from(7u8)),
            ));
            let array = e.mk_seq(
                vec![element, other],
                SequenceTargetType::Array(2),
                Type::int(wide),
            );
            let read = e.array_get(array, params[1]);
            vec![e.cast_to(CastTarget::Int(64), read)]
        },
    );
    let compiled = Compiled::new(ssa).expect("a wide sequence compiles");

    let value = IntBits::from_biguint(253, &((BigUint::from(1u8) << 200) + BigUint::from(5u8)));
    let index = IntBits::from_u128(32, 0);
    let low = IntBits::from_u128(64, 5);
    let verdict = compiled.run(&input_block(&[&value, &index, &low]));
    let witness = verdict
        .witness()
        .unwrap_or_else(|| panic!("an int{wide} sequence: {verdict:?}"))
        .to_vec();

    for column in 0..witness.len() {
        let mut perturbed = witness.clone();
        perturbed[column] += Field::from(1u64);
        assert!(
            compiled.first_unsatisfied_constraint(&perturbed).is_some(),
            "column {column} of {} is unconstrained — adding one to it left every constraint \
             satisfied",
            witness.len()
        );
    }
}

/// `main(a: int64, i: int32) -> int64 { [a as int(bits), TOP][i] as int64 }`, where `TOP` is
/// `2^(bits - 8) + 7`.
///
/// The index is a parameter, so the lookup cannot be folded away before validation sees the
/// sequence it reads, and it is a witness, so the lookup is one the tape has to address.
fn program_indexing_a_sequence_of(bits: usize) -> HLSSA {
    main_program(
        &[Type::int(64), Type::int(32)],
        &[Type::int(64)],
        move |e, params| {
            let wide = e.cast_to(CastTarget::Int(bits), params[0]);
            // The constant occupies the **top** limb the width reaches, which makes a reader that
            // drops the high cells detectable: the tape's entry then disagrees with the witnessed
            // element and the lookup goes unsatisfied.
            let other = e.int_const(IntBits::from_biguint(
                bits,
                &((BigUint::from(1u8) << (bits - 8)) + BigUint::from(7u8)),
            ));
            let array = e.mk_seq(
                vec![wide, other],
                SequenceTargetType::Array(2),
                Type::int(bits),
            );
            let element = e.array_get(array, params[1]);
            vec![e.cast_to(CastTarget::Int(64), element)]
        },
    )
}

/// A sequence element wider than the field carries is **read**, by transposing the sequence.
///
/// Above the injective width an entry has no field element of its own, so the sequence is stored
/// one sequence per limb and each limb is an entry the tape already knows how to read. 254 is the
/// first such width; 320 is neither a limb multiple of it nor the same limb count, so the two
/// exercise different shapes.
#[test]
fn a_sequence_element_wider_than_the_field_is_transposed() {
    for bits in [first_width_past_the_field(), 320] {
        let compiled = Compiled::new(program_indexing_a_sequence_of(bits))
            .unwrap_or_else(|error| panic!("an int{bits} element is not read: {error}"));

        let value = IntBits::from_u128(64, 0xdead_beef);
        let index = IntBits::from_u128(32, 1);
        let expected = IntBits::from_u128(64, 7);
        let verdict = compiled.run(&input_block(&[&value, &index, &expected]));
        assert!(verdict.is_accepted(), "int{bits} element: {verdict:?}");

        // And the lane can go red: a declared return the program does not compute is refused.
        let wrong = IntBits::from_u128(64, 8);
        let verdict = compiled.run(&input_block(&[&value, &index, &wrong]));
        assert!(
            !verdict.is_accepted(),
            "int{bits} element: a wrong answer was accepted"
        );
    }
}

/// The band **between** the narrow lanes and the transpose, served by `ELEM_CELLS`.
///
/// Three widths reach the VM's lookup tape by three different paths, and only the third is this
/// unit's: one cell is `ELEM_WORD`, two are `ELEM_U128`, and from 129 bits up to the widest width
/// the field carries injectively an element is still **one** field element but spans more cells
/// than either tag names. That is `ELEM_CELLS`, and above it a sequence is transposed instead, so
/// no width outside `129..=253` selects it.
///
/// The element is compared **whole** rather than narrowed to its low limb. A narrowing return is
/// answered correctly by a reader that drops every high cell, and that is exactly the ablation
/// this test exists to catch: `read_cells_as_field` taking one cell instead of `cells` leaves the
/// entire workspace green without it.
#[test]
fn an_element_between_the_narrow_lanes_and_the_transpose_is_read_whole() {
    let injective = first_width_past_the_field() - 1;
    // The first width past the double lane, one in the middle, and the widest the band holds.
    for bits in [2 * HOST_LIMB_BITS + 1, 200, injective] {
        let top = IntBits::from_biguint(
            bits,
            &((BigUint::from(1u8) << (bits - 8))
                + (BigUint::from(1u8) << 130)
                + BigUint::from(7u8)),
        );
        let expected = top.clone();
        let ssa = main_program(
            &[Type::int(injective), Type::int(32)],
            &[Type::int(1)],
            move |e, params| {
                let element = e.cast_to(CastTarget::Int(bits), params[0]);
                let other = e.int_const(top.clone());
                let array = e.mk_seq(
                    vec![element, other],
                    SequenceTargetType::Array(2),
                    Type::int(bits),
                );
                let read = e.array_get(array, params[1]);
                let want = e.int_const(expected.clone());
                vec![e.cmp(read, want, CmpKind::Eq)]
            },
        );
        let compiled = Compiled::new(ssa)
            .unwrap_or_else(|error| panic!("an int{bits} element is not read: {error}"));

        let value = IntBits::from_biguint(injective, &(BigUint::from(1u8) << 200));
        let index = IntBits::from_u128(32, 1);
        let verdict = compiled.run(&input_block(&[&value, &index, &IntBits::from_u128(1, 1)]));
        assert!(verdict.is_accepted(), "int{bits} element: {verdict:?}");

        // And it can go red: slot 0 is the parameter, which is not the constant.
        let miss = IntBits::from_u128(32, 0);
        let verdict = compiled.run(&input_block(&[&value, &miss, &IntBits::from_u128(1, 1)]));
        assert!(
            !verdict.is_accepted(),
            "int{bits} element: a wrong answer was accepted"
        );
    }
}

/// An array of wide elements becomes a **slice** of them, which is `k` casts rather than one.
///
/// `Cast{ArrayToSlice}` is the one opcode in the slice family that survives to this pass without a
/// sequence test of its own: `SlicePop`, `SliceInsert` and `SliceRemove` are lowered before it
/// runs, and `SlicePush`/`SliceLen` are covered above.
#[test]
fn a_wide_array_becomes_a_slice() {
    let bits = 320usize;
    let top = |n: u128| {
        IntBits::from_biguint(
            bits,
            &((BigUint::from(1u8) << (bits - 8)) + BigUint::from(n)),
        )
    };
    let expected = top(2);
    let ssa = main_program(
        &[Type::int(253), Type::int(32)],
        &[Type::int(1)],
        move |e, params| {
            let element = e.cast_to(CastTarget::Int(bits), params[0]);
            let other = e.int_const(top(2));
            let array = e.mk_seq(
                vec![element, other],
                SequenceTargetType::Array(2),
                Type::int(bits),
            );
            let slice = e.cast_to(CastTarget::ArrayToSlice, array);
            let read = e.array_get(slice, params[1]);
            let want = e.int_const(expected.clone());
            vec![e.cmp(read, want, CmpKind::Eq)]
        },
    );
    let compiled = Compiled::new(ssa).expect("a wide array becomes a slice");

    let value = IntBits::from_biguint(253, &(BigUint::from(1u8) << 200));
    let index = IntBits::from_u128(32, 1);
    let verdict = compiled.run(&input_block(&[&value, &index, &IntBits::from_u128(1, 1)]));
    assert!(
        verdict.is_accepted(),
        "an int{bits} array to slice: {verdict:?}"
    );

    let miss = IntBits::from_u128(32, 0);
    let verdict = compiled.run(&input_block(&[&value, &miss, &IntBits::from_u128(1, 1)]));
    assert!(
        !verdict.is_accepted(),
        "an int{bits} array to slice: a wrong answer was accepted"
    );
}

/// The same for a sequence built from a **blob**, whose elements are constants.
///
/// This is the case a bound stated on the tape's reading could not have served. A blob of constants
/// is never witnessed, so it stays in the pure domain — and a 320-bit constant has no field element
/// for an entry to be read as. Transposing it makes every entry a narrow constant, which does.
#[test]
fn a_blob_sequence_of_wide_constants_is_read() {
    for bits in [first_width_past_the_field(), 320] {
        let ssa = main_program(&[Type::int(32)], &[Type::int(1)], move |e, params| {
            let blob = e.emit_constant(Constant::Blob(Blob::new(
                Type::int(bits),
                vec![
                    // Top-limb values, so a reading that drops the high limbs is detectable.
                    Constant::Int(IntBits::from_biguint(
                        bits,
                        &((BigUint::from(1u8) << (bits - 8)) + BigUint::from(3u8)),
                    )),
                    Constant::Int(IntBits::from_biguint(
                        bits,
                        &((BigUint::from(1u8) << (bits - 9)) + BigUint::from(4u8)),
                    )),
                ],
            )));
            let array = e.mk_seq_of_blob(Type::int(bits), blob);
            let element = e.array_get(array, params[0]);
            // Compared against the whole expected element rather than narrowed to 64 bits. A
            // narrowing return sees only the low limb, so a split that put the low window in every
            // limb would still answer correctly.
            let expected = e.int_const(IntBits::from_biguint(
                bits,
                &((BigUint::from(1u8) << (bits - 9)) + BigUint::from(4u8)),
            ));
            vec![e.cmp(element, expected, CmpKind::Eq)]
        });

        let compiled = Compiled::new(ssa)
            .unwrap_or_else(|error| panic!("an int{bits} blob element is not read: {error}"));
        let index = IntBits::from_u128(32, 1);
        let expected = IntBits::from_u128(1, 1);
        let verdict = compiled.run(&input_block(&[&index, &expected]));
        assert!(verdict.is_accepted(), "int{bits} blob element: {verdict:?}");

        let wrong = IntBits::from_u128(1, 0);
        let verdict = compiled.run(&input_block(&[&index, &wrong]));
        assert!(
            !verdict.is_accepted(),
            "int{bits} blob element: a wrong answer was accepted"
        );
    }
}

/// Every sequence shape carries a wide element, not just the one the other tests use.
#[test]
fn every_sequence_shape_carries_a_wide_element() {
    let bits = 320usize;
    // `MkRepeated`: every slot is the same wide element. The element is derived from a parameter
    // rather than being a constant because an all-constant wide element folds, and the fold has no
    // field element to land in — the divergence `docs/int-semantics.md` states for a pure integer
    // wider than the field carries injectively.
    let repeated = main_program(
        &[Type::int(64), Type::int(32)],
        &[Type::int(64)],
        move |e, params| {
            let element = e.cast_to(CastTarget::Int(bits), params[0]);
            let array = e.mk_repeated(element, SequenceTargetType::Array(3), 3, Type::int(bits));
            let read = e.array_get(array, params[1]);
            vec![e.cast_to(CastTarget::Int(64), read)]
        },
    );
    let compiled = Compiled::new(repeated).expect("a repeated wide element");
    let value = IntBits::from_u128(64, 0xfeed);
    let index = IntBits::from_u128(32, 2);
    let expected = IntBits::from_u128(64, 0xfeed);
    let verdict = compiled.run(&input_block(&[&value, &index, &expected]));
    assert!(verdict.is_accepted(), "MkRepeated: {verdict:?}");

    // A slice, which reaches its elements through its own opcode family rather than the tape, and
    // whose elements here are **constants**, which is the shape that folds if a read reassembles a
    // whole wide value instead of keeping its limbs.
    let sliced = main_program(&[Type::int(32)], &[Type::int(64)], move |e, params| {
        let top = |low: u128| {
            IntBits::from_biguint(
                bits,
                &((BigUint::from(1u8) << (bits - 8)) + BigUint::from(low)),
            )
        };
        let first = e.int_const(top(1));
        let second = e.int_const(top(2));
        let slice = e.mk_seq(
            vec![first, second],
            SequenceTargetType::Slice,
            Type::int(bits),
        );
        let read = e.array_get(slice, params[0]);
        vec![e.cast_to(CastTarget::Int(64), read)]
    });
    let compiled = Compiled::new(sliced).expect("a wide slice element");
    let index = IntBits::from_u128(32, 1);
    let expected = IntBits::from_u128(64, 2);
    let verdict = compiled.run(&input_block(&[&index, &expected]));
    assert!(verdict.is_accepted(), "slice of constants: {verdict:?}");
}

/// An `[int256; 4]` with a witness index, read and written.
///
/// Four slots rather than two, so a transpose that happened to line up at two does not pass; and
/// nested, because `limb_types` recurses through a sequence and the inner one has to transpose
/// under the outer.
#[test]
fn a_four_element_wide_array_is_read_written_and_nested() {
    let bits = 256usize;
    let top = |n: u128| {
        IntBits::from_biguint(
            bits,
            &((BigUint::from(1u8) << (bits - 8)) + BigUint::from(n)),
        )
    };

    // Read at a witness index.
    let read = main_program(
        &[Type::int(253), Type::int(32)],
        &[Type::int(64)],
        move |e, params| {
            let wide = e.cast_to(CastTarget::Int(bits), params[0]);
            let (a, b, c) = (
                e.int_const(top(1)),
                e.int_const(top(2)),
                e.int_const(top(3)),
            );
            let array = e.mk_seq(
                vec![wide, a, b, c],
                SequenceTargetType::Array(4),
                Type::int(bits),
            );
            let element = e.array_get(array, params[1]);
            vec![e.cast_to(CastTarget::Int(64), element)]
        },
    );
    let compiled = Compiled::new(read).expect("[int256; 4] read");
    let value = IntBits::from_biguint(253, &(BigUint::from(1u8) << 200));
    let index = IntBits::from_u128(32, 3);
    let expected = IntBits::from_u128(64, 3);
    let verdict = compiled.run(&input_block(&[&value, &index, &expected]));
    assert!(verdict.is_accepted(), "[int256; 4] read: {verdict:?}");

    // Written at a witness index, and read back at a constant one.
    let written = main_program(
        &[Type::int(253), Type::int(32)],
        &[Type::int(64)],
        move |e, params| {
            let wide = e.cast_to(CastTarget::Int(bits), params[0]);
            let (a, b, c, d) = (
                e.int_const(top(1)),
                e.int_const(top(2)),
                e.int_const(top(3)),
                e.int_const(top(4)),
            );
            let array = e.mk_seq(
                vec![a, b, c, d],
                SequenceTargetType::Array(4),
                Type::int(bits),
            );
            let set = e.array_set(array, params[1], wide);
            let two = e.int_const(IntBits::from_u128(32, 2));
            let element = e.array_get(set, two);
            vec![e.cast_to(CastTarget::Int(64), element)]
        },
    );
    let compiled = Compiled::new(written).expect("[int256; 4] write");
    // Written at slot 0, so slot 2 keeps its constant.
    let index = IntBits::from_u128(32, 0);
    let expected = IntBits::from_u128(64, 3);
    let verdict = compiled.run(&input_block(&[&value, &index, &expected]));
    assert!(verdict.is_accepted(), "[int256; 4] write: {verdict:?}");

    // Nested, with the outer row chosen at a witness index.
    let nested = main_program(
        &[Type::int(253), Type::int(32)],
        &[Type::int(64)],
        move |e, params| {
            let wide = e.cast_to(CastTarget::Int(bits), params[0]);
            let one = e.int_const(top(1));
            let inner_a = e.mk_seq(
                vec![wide, one],
                SequenceTargetType::Array(2),
                Type::int(bits),
            );
            let (two, three) = (e.int_const(top(2)), e.int_const(top(3)));
            let inner_b = e.mk_seq(
                vec![two, three],
                SequenceTargetType::Array(2),
                Type::int(bits),
            );
            let outer = e.mk_seq(
                vec![inner_a, inner_b],
                SequenceTargetType::Array(2),
                Type::int(bits).array_of(2),
            );
            let row = e.array_get(outer, params[1]);
            let zero = e.int_const(IntBits::zero(32));
            let element = e.array_get(row, zero);
            vec![e.cast_to(CastTarget::Int(64), element)]
        },
    );
    let compiled = Compiled::new(nested).expect("[[int256; 2]; 2]");
    let index = IntBits::from_u128(32, 1);
    let expected = IntBits::from_u128(64, 2);
    let verdict = compiled.run(&input_block(&[&value, &index, &expected]));
    assert!(verdict.is_accepted(), "[[int256; 2]; 2]: {verdict:?}");
}

/// A slice whose length changes at runtime, carrying wide elements.
///
/// The slice family is the half of this unit with no tape involvement: `SlicePush` and `SliceLen`
/// are limb-moving in their own right, and `SliceLen` reads the length off **one** limb sequence,
/// which is only correct because every length-changing operation is applied identically to all of
/// them. A push that reached some limb sequences and not others would leave them ragged, and the
/// length would then depend on which one was asked.
#[test]
fn a_wide_slice_grows_and_reports_its_length() {
    let bits = 320usize;
    let top = |n: u128| {
        IntBits::from_biguint(
            bits,
            &((BigUint::from(1u8) << (bits - 8)) + BigUint::from(n)),
        )
    };

    let pushed = main_program(
        &[Type::int(253), Type::int(32)],
        &[Type::int(64)],
        move |e, params| {
            let wide = e.cast_to(CastTarget::Int(bits), params[0]);
            let seed = e.int_const(top(1));
            let slice = e.mk_seq(vec![seed], SequenceTargetType::Slice, Type::int(bits));
            let longer = e.slice_push(slice, vec![wide], SliceOpDir::Back);
            let read = e.array_get(longer, params[1]);
            vec![e.cast_to(CastTarget::Int(64), read)]
        },
    );
    let compiled = Compiled::new(pushed).expect("a wide slice push");
    let value = IntBits::from_biguint(253, &(BigUint::from(1u8) << 200));
    // Slot 0 is the seed, whose low 64 bits are 1.
    let index = IntBits::from_u128(32, 0);
    let expected = IntBits::from_u128(64, 1);
    let verdict = compiled.run(&input_block(&[&value, &index, &expected]));
    assert!(verdict.is_accepted(), "a wide slice push: {verdict:?}");

    let length = main_program(&[Type::int(253)], &[Type::int(32)], move |e, params| {
        let wide = e.cast_to(CastTarget::Int(bits), params[0]);
        let slice = e.mk_seq(vec![wide], SequenceTargetType::Slice, Type::int(bits));
        let longer = e.slice_push(slice, vec![wide], SliceOpDir::Back);
        vec![e.slice_len(longer)]
    });
    let compiled = Compiled::new(length).expect("a wide slice length");
    let two = IntBits::from_u128(32, 2);
    let verdict = compiled.run(&input_block(&[&value, &two]));
    assert!(verdict.is_accepted(), "a wide slice length: {verdict:?}");
}

/// The third lane: the same programs compiled to WASM and run under wasmtime.
///
/// The corpus already checks this lane byte-identically at every width Noir can name, which is
/// strictly stronger than anything here, and it is silent above 128 bits because no corpus program
/// has a width there. So this says that the wide field boundary and the multi-cell representation
/// compute the same answers in the compiled module as in the interpreter and the constraint system.
///
/// Skipped only where the linker reported success and wrote no module; a WASM lane that fails to
/// compile fails here rather than being skipped. See `harness::compile_wasm`.
#[test]
fn the_wasm_lane_agrees_at_a_wide_width() {
    let injective = first_width_past_the_field() - 1;
    let mut ran = 0usize;

    // The field boundary, at a width only 6.3a's packing reaches, with the value in the top limb.
    for bits in [64usize, 128, 129, 200, injective] {
        let ssa = main_program(&[Type::field()], &[Type::field()], |e, params| {
            let wide = e.cast_to(CastTarget::Int(bits), params[0]);
            vec![e.cast_to_field(wide)]
        });
        let compiled = Compiled::new(ssa).expect("a wide round trip compiles");
        let value = Field::from((BigUint::from(1u8) << (bits - 8)) + BigUint::from(7u8));
        let inputs = [
            InputValueOrdered::Field(value),
            InputValueOrdered::Field(value),
        ];

        let Some(wasm) = compiled.run_wasm(&inputs) else {
            continue;
        };
        ran += 1;
        assert!(
            wasm.is_accepted(),
            "Field as int{bits} as Field in WASM: {wasm:?}"
        );
    }

    // And the multi-cell representation, above the width the field carries at all.
    let ssa = main_program(&[Type::int(128)], &[Type::int(64)], |e, params| {
        let widened = e.cast_to(CastTarget::Int(320), params[0]);
        vec![e.cast_to(CastTarget::Int(64), widened)]
    });
    let compiled = Compiled::new(ssa).expect("an int320 round trip compiles");
    let value = IntBits::from_u128(128, (1u128 << 100) + 5);
    let low = IntBits::from_u128(64, 5);
    if let Some(wasm) = compiled.run_wasm(&input_block(&[&value, &low])) {
        ran += 1;
        assert!(wasm.is_accepted(), "an int320 round trip in WASM: {wasm:?}");
    }

    // A wide **sequence** read at a witness index, which is the only thing that exercises the
    // compiled half of the lookup: `ELEM_CELLS` in the VM and the transposed tables in codegen are
    // both unreachable from the scalar programs above.
    for bits in [200usize, 320] {
        let compiled = Compiled::new(program_indexing_a_sequence_of(bits))
            .unwrap_or_else(|error| panic!("an int{bits} sequence compiles: {error}"));
        let value = IntBits::from_u128(64, 0xdead_beef);
        let index = IntBits::from_u128(32, 1);
        let expected = IntBits::from_u128(64, 7);
        if let Some(wasm) = compiled.run_wasm(&input_block(&[&value, &index, &expected])) {
            ran += 1;
            assert!(
                wasm.is_accepted(),
                "an int{bits} sequence element in WASM: {wasm:?}"
            );
        }
    }

    // And the bitwise family, which reaches a wide width by a different route than anything above:
    // the operation is split per limb before the lowering, and each limb is decomposed again into
    // half-limbs. Neither the split nor the ragged top limb is exercised by a scalar round trip.
    for bits in [200usize, 254] {
        let ssa = main_program(
            &[Type::int(64), Type::int(64)],
            &[Type::int(64)],
            move |e, params| {
                let lhs = e.cast_to(CastTarget::Int(bits), params[0]);
                let rhs = e.cast_to(CastTarget::Int(bits), params[1]);
                let answer = e.bin(BinaryArithOpKind::Xor, lhs, rhs);
                vec![e.cast_to(CastTarget::Int(64), answer)]
            },
        );
        let compiled = Compiled::new(ssa).expect("a wide bitwise compiles");
        let (x, y) = (0xF0F0_F0F0_0F0F_0F0Fu128, 0x00FF_00FF_FF00_FF00u128);
        let inputs = input_block(&[
            &IntBits::from_u128(64, x),
            &IntBits::from_u128(64, y),
            &IntBits::from_u128(64, x ^ y),
        ]);
        if let Some(wasm) = compiled.run_wasm(&inputs) {
            ran += 1;
            assert!(wasm.is_accepted(), "an int{bits} xor in WASM: {wasm:?}");
        }
    }

    // And the complement, on both sides of the representation threshold: at 200 it is one field
    // subtraction against an all-ones mask, and at 254 the value is limbs and the complement is one
    // per limb, the top one at the narrower width that limb actually carries.
    for bits in [200usize, 254] {
        let ssa = main_program(
            &[Type::int(64), Type::int(64)],
            &[Type::int(64)],
            move |e, params| {
                let wide = e.cast_to(CastTarget::Int(bits), params[0]);
                let other = e.cast_to(CastTarget::Int(bits), params[1]);
                let flipped = e.not(wide);
                let answer = e.bin(BinaryArithOpKind::Xor, flipped, other);
                vec![e.cast_to(CastTarget::Int(64), answer)]
            },
        );
        let compiled = Compiled::new(ssa).expect("a wide complement compiles");
        let (x, y) = (0xDEAD_BEEF_0BAD_F00Du64, 0xF0F0_F0F0_0F0F_0F0Fu64);
        let inputs = input_block(&[
            &IntBits::from_u128(64, u128::from(x)),
            &IntBits::from_u128(64, u128::from(y)),
            &IntBits::from_u128(64, u128::from(!x ^ y)),
        ]);
        if let Some(wasm) = compiled.run_wasm(&inputs) {
            ran += 1;
            assert!(
                wasm.is_accepted(),
                "an int{bits} complement in WASM: {wasm:?}"
            );
        }
    }

    assert!(
        ran == 12 || !compiled.wasm_is_available(),
        "the WASM lane was available and ran {ran} of 12"
    );
}

/// Run the pipeline as far as the phase width validation lives in, and no further.
fn validate_only(ssa: HLSSA) -> Result<(), DriverError> {
    let scratch = tempfile::TempDir::new().expect("a temporary directory for the pipeline's dumps");
    Driver::from_ssa(ssa, scratch.path().to_path_buf(), false).make_struct_access_static()
}

/// `main(value) { fn_ptr(value) }` calling `callee(value) { value as Field }` indirectly, with the
/// cast at `bits`.
fn program_calling_through_a_function_pointer(bits: usize) -> HLSSA {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let mut builder = HLSSABuilder::new(&mut ssa);

    let (callee_id, ()) = builder.add_function("callee".to_string(), |b| {
        b.function.add_return_type(Type::field());
        let entry = b.function.get_entry_id();
        let mut e = b
            .block(entry)
            .with_source_location(SourceLocation::synthetic("callee"));
        let value = e.add_parameter(Type::int(bits));
        let field = e.cast_to_field(value);
        e.terminate_return(vec![field]);
    });

    builder.modify_function(main_id, |b| {
        b.function.add_return_type(Type::field());
        let entry = b.function.get_entry_id();
        let mut e = b
            .block(entry)
            .with_source_location(SourceLocation::synthetic("indirect_main"));
        let value = e.add_parameter(Type::int(bits));
        let fn_ptr = e.emit_constant(Constant::FnPtr(callee_id));
        let results = e.call_indirect(fn_ptr, vec![value], 1);
        e.terminate_return(results);
    });

    ssa
}

/// A program that calls through a function pointer compiles.
///
/// Width validation needs `TypeInfo` and runs on the untransformed SSA, so `TypeInfo` has to be
/// able to type a `CallTarget::Dynamic` — which it does from the callee value's own
/// `TypeExpr::Function`, carrying the results a call through it produces. This is the shape any
/// `map`, `fold` or `fn(..) -> ..` parameter in the corpus lowers to, and it reaches validation
/// with its dynamic calls still in place.
#[test]
fn a_program_with_an_indirect_call_reaches_validation() {
    assert!(
        validate_only(program_calling_through_a_function_pointer(8)).is_ok(),
        "an indirect call is not a width problem"
    );
}

/// A cast in a function that only a function pointer reaches is refused like any other. The rule
/// walks every function, so nothing about how a callee is reached changes the answer.
#[test]
fn an_oversized_cast_behind_a_function_pointer_is_still_refused() {
    let ssa = program_calling_through_a_function_pointer(first_width_past_the_field());

    let Err(DriverError::Refused(diagnostics)) = validate_only(ssa) else {
        panic!("a cast the field cannot carry is refused wherever it is reached from");
    };
    assert_eq!(diagnostics.len(), 1, "{diagnostics:?}");
}

/// Point `ssa`'s single cast at `location`, leaving every other instruction where it is.
fn locate_the_cast(ssa: &mut HLSSA, location: SourceLocation) {
    let main = ssa.get_unique_entrypoint_id();
    for (_, block) in ssa.get_function_mut(main).get_blocks_mut() {
        let found: Vec<usize> = block
            .get_instructions()
            .enumerate()
            .filter(|(_, op)| matches!(op, OpCode::Cast { .. }))
            .map(|(index, _)| index)
            .collect();
        for index in found {
            block.set_instruction_source_location(index, location.clone());
        }
    }
}
