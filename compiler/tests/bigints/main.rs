//! The oracle's test suite: what the pipeline must agree with the model about.

mod goldilocks;
mod harness;

use harness::{
    BinaryOpOracle, Compiled, HarnessField, Verdict, bytecode_listing, input_block, main_program,
};
use mavros_artifacts::{Field, FieldConfig, InputValueOrdered};
use mavros_compiler::{
    compiler::{
        passes::shared::limbs::{widest_injective_int_bits, witness_limb_bits},
        ssa::{
            SourceLocation, Terminator, ValueId,
            hlssa::{
                BinaryArithOpKind, Blob, CastTarget, CmpKind, Constant, Endianness, HLSSA, OpCode,
                Radix, SequenceTargetType, SliceOpDir, Type,
                builder::{HLBlockEmitter, HLEmitter as _, HLSSABuilder},
            },
        },
    },
    driver::{Driver, Error as DriverError},
};
use mavros_int_semantics::{CmpOp, IntBits, IntOp, SignedValue, corners, int_bits::HOST_LIMB_BITS};
use num_bigint::BigUint;

// UTILITIES
// ================================================================================================

/// Every operand pair worth trying for one operation at one width: the corner values against each
/// other, with a shift's right operand drawn from the amounts instead.
fn corner_pairs(op: IntOp, bits: usize) -> Vec<(IntBits, IntBits)> {
    let rhs = if op.is_shift() {
        corners::shift_amounts(bits)
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
    let oracle = BinaryOpOracle::new(kind, bits)
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

/// Adds one to each column of an accepted witness in turn, and takes one from each column the
/// program computes, requiring each change to break a constraint: a column no constraint mentions
/// survives it.
///
/// `inputs` is the honest input block: the program's parameters followed by its declared returns.
///
/// **An input column is only stepped up**: a different input with the same answer is another honest
/// run, so taking one away need not fail. [`assert_changing_an_input_breaks_the_witness`] covers
/// inputs.
fn assert_every_column_is_pinned<F: HarnessField>(
    what: &str,
    compiled: &Compiled<F>,
    inputs: &[&IntBits],
) {
    let verdict = compiled.run(&harness::input_block_in(inputs));
    let witness = verdict
        .witness()
        .unwrap_or_else(|| panic!("{what}: {verdict:?}"))
        .to_vec();
    let input_columns: Vec<usize> = (0..inputs.len())
        .map(|index| compiled.input_column(index))
        .chain(compiled.guard_column())
        .collect();

    for column in 0..witness.len() {
        let steps: &[(F, &str)] = if input_columns.contains(&column) {
            &[(F::ONE, "adding one to")]
        } else {
            &[(F::ONE, "adding one to"), (-F::ONE, "taking one from")]
        };
        for &(step, direction) in steps {
            let mut perturbed = witness.clone();
            perturbed[column] += step;
            assert!(
                compiled.first_unsatisfied_constraint(&perturbed).is_some(),
                "{what}: column {column} of {} is unconstrained — \
                 {direction} it left every constraint satisfied",
                witness.len()
            );
        }
    }
}

/// Replaces input `index` of an accepted witness with `replacement`, leaving every other column as
/// the honest run wrote it, and requires the result to break a constraint.
///
/// `replacement` must change the answer. Unlike [`assert_every_column_is_pinned`], only the operand
/// moves, so this catches constraints that never read it.
fn assert_changing_an_input_breaks_the_witness(
    what: &str,
    compiled: &Compiled,
    inputs: &[&IntBits],
    index: usize,
    replacement: &IntBits,
) {
    let verdict = compiled.run(&input_block(inputs));
    let mut witness = verdict
        .witness()
        .unwrap_or_else(|| panic!("{what}: {verdict:?}"))
        .to_vec();
    let column = compiled.input_column(index);
    assert_eq!(
        witness[column],
        harness::input_field(inputs[index]),
        "{what}: input {index} is written to column {column}"
    );
    witness[column] = harness::input_field(replacement);
    assert!(
        compiled.first_unsatisfied_constraint(&witness).is_some(),
        "{what}: replacing input {index} with {replacement:?} left every constraint satisfied"
    );
}

/// The two-column mutation test: no pair of columns can trade a limb's place value.
///
/// Raising one column by `2^h`, for the witness limb width `h`, and lowering another by one keeps
/// every recombination `Σ limb_i · 2^(h·i)`, so only a range check on a limb or carry refuses it,
/// which [`assert_every_column_is_pinned`] cannot see. Where the column to lower is zero in the
/// honest witness, it would wrap, and the range check on it would mask a missing one on the raised
/// column, so that pair trades the other way: the first column loses `2^h` and the second gains
/// one. Only the place value `2^h` is probed. Column 0, the constant one, is left to
/// [`assert_every_column_is_pinned`]. Quadratic in the witness, so kept to small programs.
fn assert_no_two_columns_trade_a_limb<F: HarnessField>(
    what: &str,
    compiled: &Compiled<F>,
    inputs: &[&IntBits],
) {
    let verdict = compiled.run(&harness::input_block_in(inputs));
    let witness = verdict
        .witness()
        .unwrap_or_else(|| panic!("{what}: {verdict:?}"))
        .to_vec();
    let limb = witness_limb_bits(F::CONFIG);
    let place = F::of_biguint(&(BigUint::from(1u8) << limb));
    let one = F::ONE;
    let zero = F::ZERO;
    for traded in 1..witness.len() {
        for paired in 1..witness.len() {
            if traded == paired {
                continue;
            }
            let mut perturbed = witness.clone();
            let (step, direction) = if witness[paired] == zero {
                (-one, ("lowering", "raising"))
            } else {
                (one, ("raising", "lowering"))
            };
            perturbed[traded] += step * place;
            perturbed[paired] -= step;
            assert!(
                compiled.first_unsatisfied_constraint(&perturbed).is_some(),
                "{what}: {} column {traded} by 2^{limb} and {} column {paired} by one left every \
                 constraint satisfied",
                direction.0,
                direction.1
            );
        }
    }
}

/// Whether a run was refused by the program's constraints rather than by the entry point's return
/// check.
///
/// A trap at the return check says only that the declared answer differs from the VM's, which an
/// under-constrained circuit also produces. So a test of what the constraints reject declares the
/// answer the VM computes and asks for this.
///
/// Meaningful for a [`Compiled::run`] verdict only: a WASM trap carries no mavros stack, so on that
/// lane a return-check failure reads as any other refusal, and a WASM half asks for
/// [`Verdict::rejects`] instead.
fn refused_by_the_constraints(verdict: &Verdict) -> bool {
    verdict.is_refusal() || matches!(verdict, Verdict::Unsatisfied { .. })
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

/// The same claim at widths **no Noir program can name**, the shifts included: a shift's amount
/// check is a real `amount < bits`, and its table has more rows than there are amounts.
#[test]
fn every_operation_agrees_with_the_model_at_a_width_noir_cannot_express() {
    for kind in BinaryArithOpKind::ALL {
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
/// has a separate lowering: `lower_unsigned_mul` splits an `int128` into two limbs and multiplies
/// them schoolbook, while every narrower product is one field multiplication. Division and
/// remainder are checked by the same product, `q·d`, so they take its path too.
#[test]
fn arithmetic_agrees_at_the_top_of_the_two_cell_lane() {
    for kind in [
        BinaryArithOpKind::UAdd,
        BinaryArithOpKind::UMul,
        BinaryArithOpKind::UDiv,
        BinaryArithOpKind::URem,
    ] {
        sweep(kind, 128);
    }
}

/// A witnessed division and remainder past the host word and below the double lane, which the
/// single cell checks with one field product: 65 is the first width past the host word, and 126 the
/// widest whose `q·d` one bn254 element still holds.
#[test]
fn a_division_agrees_between_the_host_word_and_the_widest_single_cell_product() {
    let widest = widest_injective_int_bits(FieldConfig::bn254()) / 2;
    assert_eq!(widest, 126, "the widest single-cell product on bn254");
    for bits in [HOST_LIMB_BITS + 1, 96, widest] {
        for kind in [BinaryArithOpKind::UDiv, BinaryArithOpKind::URem] {
            sweep(kind, bits);
        }
    }
}

/// The ablation: the tests above can go red. One program is driven to each case:
///
/// - The model's value is accepted.
/// - A value the model did not produce is refused at the return check.
/// - Operands the model rejects are refused elsewhere.
#[test]
fn the_oracle_distinguishes_a_wrong_answer_from_a_refusal() {
    let byte = |v| IntBits::from_u128(8, v);
    let oracle = BinaryOpOracle::new(BinaryArithOpKind::UAdd, 8).unwrap();

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
/// It is the only check here an under-constrained circuit can fail, being the only one that judges
/// a witness the VM did not produce.
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
        (BinaryArithOpKind::UDiv, 127),
        (BinaryArithOpKind::URem, 127),
        (BinaryArithOpKind::UDiv, 200),
        (BinaryArithOpKind::URem, 200),
        (BinaryArithOpKind::UDiv, 253),
        (BinaryArithOpKind::URem, 253),
        (BinaryArithOpKind::SShr, 8),
    ] {
        let oracle = BinaryOpOracle::new(kind, bits)
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

        // 7 < 3 is false, declared as the return so that the entry point's check tests it.
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
/// No parameter or declared return can be wider than one element, so a limbed operand is built
/// inside the program: a widened narrow parameter beside a constant reaching the top limb. The sum
/// and the difference leave through their low limb, the ordering as one bit.
///
/// Nothing is read back with an equality: its inverse hint is free when the sides are equal, so it
/// always leaves a column unpinned.
///
/// Each pair runs the chain its whole length: `(2^256 - 1) + 3` carries out of every limb below the
/// top, `2^300 - 3` borrows out of each, and `3 < 2^300` is decided only in the top limb, so every
/// lower carry is read as zero.
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

/// The mutation test for a product of limbs, with one operand witnessed and with both.
///
/// Built inside the program for the reason
/// [`every_witness_column_of_a_limbed_sum_difference_or_ordering_is_pinned`] gives. The left
/// operand is a widened parameter; the right is a constant reaching the fourth limb, used as is and
/// then selected by a witnessed bit, which makes it a witnessed value held as limbs. The two differ
/// in cost: a partial product of a witnessed limb and a pure one is a linear term, while one of two
/// witnessed limbs is a new column that only its product constraint pins.
///
/// The product leaves through its low limb. A missing range check on a limb of the answer or on a
/// carry is invisible here whichever limb leaves: raising a carry by one lowers its limb by `2^h`
/// and raises the next by one, which the recombination cancels. Exploiting it takes two columns
/// moving together, which only the structural audit in `wide_witness_ints` sees.
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

/// The mutation test for a division of limbs, by a constant and by a witnessed divisor.
///
/// Built inside the program for the reason
/// [`every_witness_column_of_a_limbed_sum_difference_or_ordering_is_pinned`] gives. The dividend
/// is a constant reaching the top limb, selected by a witnessed bit so that it is a witnessed value
/// held as limbs; the divisor is a constant reaching the fourth limb or a widened parameter. A
/// large divisor leaves a short quotient and a long remainder, a small one the reverse, so between
/// them every limb of both answers is witnessed somewhere.
///
/// **Each operand the answers are checked against is also moved alone**, by
/// [`assert_changing_an_input_breaks_the_witness`], since neither check shows when one column is
/// perturbed: `q`, its partial products and its carries move together. Clearing the bit that
/// selects the dividend sends it to zero, which only `q·d + r == n` notices. The third program does
/// the same for `r < d`: its dividend is below its divisor, so the honest answer is `q = 0, r = n`,
/// and zeroing the divisor leaves `q·0 + r == n` true. Only the ordering can refuse it.
///
/// Both answers leave through their low limb.
#[test]
fn every_witness_column_of_a_limbed_division_is_pinned() {
    let bits = 320usize;
    let dividend: BigUint = (BigUint::from(1u8) << 300) + (BigUint::from(1u8) << 200) + 12345u32;
    let large_divisor: BigUint = (BigUint::from(1u8) << 200) + 5u8;
    let small_divisor = BigUint::from(3u8);
    let low = |value: BigUint| {
        let limb = value % (BigUint::from(1u8) << 64);
        IntBits::from_biguint(64, &limb)
    };
    let selected = IntBits::from_u128(1, 1);
    let parameter = IntBits::from_u128(64, 3);

    let pure_divisor = large_divisor.clone();
    let pure_dividend = dividend.clone();
    let by_a_constant = main_program(
        &[Type::int(1)],
        &[Type::int(64), Type::int(64)],
        move |e, params| {
            let constant = e.int_const(IntBits::from_biguint(bits, &pure_dividend));
            let zero = e.int_const(IntBits::zero(bits));
            let dividend = e.select(params[0], constant, zero);
            let divisor = e.int_const(IntBits::from_biguint(bits, &pure_divisor));
            let quotient = e.bin(BinaryArithOpKind::UDiv, dividend, divisor);
            let remainder = e.bin(BinaryArithOpKind::URem, dividend, divisor);
            vec![
                e.cast_to(CastTarget::Int(64), quotient),
                e.cast_to(CastTarget::Int(64), remainder),
            ]
        },
    );
    let compiled = Compiled::new(by_a_constant).unwrap_or_else(|error| {
        panic!("UDiv by a constant at {bits} bits does not compile: {error}")
    });
    let inputs = [
        &selected,
        &low(&dividend / &large_divisor),
        &low(&dividend % &large_divisor),
    ];
    let what = format!("UDiv and URem by a constant at {bits} bits");
    assert_every_column_is_pinned(&what, &compiled, &inputs);
    assert_changing_an_input_breaks_the_witness(
        &format!("{what}, the dividend sent to zero"),
        &compiled,
        &inputs,
        0,
        &IntBits::zero(1),
    );

    let pure_dividend = dividend.clone();
    let by_a_witness = main_program(
        &[Type::int(1), Type::int(64)],
        &[Type::int(64), Type::int(64)],
        move |e, params| {
            let constant = e.int_const(IntBits::from_biguint(bits, &pure_dividend));
            let zero = e.int_const(IntBits::zero(bits));
            let dividend = e.select(params[0], constant, zero);
            let divisor = e.cast_to(CastTarget::Int(bits), params[1]);
            let quotient = e.bin(BinaryArithOpKind::UDiv, dividend, divisor);
            let remainder = e.bin(BinaryArithOpKind::URem, dividend, divisor);
            vec![
                e.cast_to(CastTarget::Int(64), quotient),
                e.cast_to(CastTarget::Int(64), remainder),
            ]
        },
    );
    let compiled = Compiled::new(by_a_witness).unwrap_or_else(|error| {
        panic!("UDiv by a witness at {bits} bits does not compile: {error}")
    });
    assert_every_column_is_pinned(
        &format!("UDiv and URem by a witness at {bits} bits"),
        &compiled,
        &[
            &selected,
            &parameter,
            &low(&dividend / &small_divisor),
            &low(&dividend % &small_divisor),
        ],
    );

    let pure_divisor = large_divisor.clone();
    let smaller_than_the_divisor = main_program(
        &[Type::int(1), Type::int(64)],
        &[Type::int(64), Type::int(64)],
        move |e, params| {
            let dividend = e.cast_to(CastTarget::Int(bits), params[1]);
            let constant = e.int_const(IntBits::from_biguint(bits, &pure_divisor));
            let zero = e.int_const(IntBits::zero(bits));
            let divisor = e.select(params[0], constant, zero);
            let quotient = e.bin(BinaryArithOpKind::UDiv, dividend, divisor);
            let remainder = e.bin(BinaryArithOpKind::URem, dividend, divisor);
            vec![
                e.cast_to(CastTarget::Int(64), quotient),
                e.cast_to(CastTarget::Int(64), remainder),
            ]
        },
    );
    let compiled = Compiled::new(smaller_than_the_divisor).unwrap_or_else(|error| {
        panic!("UDiv of a smaller dividend at {bits} bits does not compile: {error}")
    });
    let inputs = [&selected, &parameter, &IntBits::zero(64), &parameter];
    let what = format!("UDiv and URem of a dividend smaller than the divisor at {bits} bits");
    assert_every_column_is_pinned(&what, &compiled, &inputs);
    assert_changing_an_input_breaks_the_witness(
        &format!("{what}, the divisor sent to zero"),
        &compiled,
        &inputs,
        0,
        &IntBits::zero(1),
    );
}

/// The mutation test for the asserted ordering, in the band and on limbs.
///
/// **This is the one chain shape the mutation test cannot fully judge.** An asserted ordering fixes
/// its top borrow at one instead of witnessing it, so the top limb's identity has no separate
/// column: dropping that limb's range check leaves every column pinned and this test green. Then
/// [`a_guarded_assertion_holds_only_where_its_guard_does`] fails by accepting a failing assertion,
/// and so does the structural audit in `wide_witness_ints`,
/// `every_carry_is_a_bit_and_every_limb_of_the_chain_is_bounded`. This test pins every carry below
/// the top.
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
/// 200 and 252 are past `narrow_int_bits`, in the band the single-cell lowering still holds because
/// the sum fits an element. 253 is the one width where the operands fit and the sum does not, so
/// the chain runs there on a decomposition of each operand.
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
/// The second half tests the entry point's return check: witness generation is honest, so a run
/// declaring the other answer computes the right one and fails where the two are compared. It says
/// nothing about the ordering's constraints, which
/// [`every_witness_column_of_an_unsigned_ordering_is_pinned`] probes.
///
/// The byte is the single-cell baseline the rest are measured against.
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

/// The operations the representation computes on limbs: the carry chain's three, the schoolbook's
/// product, the division built from it, and the shifts, each under either reading.
#[derive(Clone, Copy, Debug)]
enum Limbed {
    Sum,
    Difference,
    Ordering,
    Product,
    Quotient,
    Remainder,
    ShiftLeft,
    ShiftRight,
    SignedSum,
    SignedDifference,
    SignedOrdering,
    SignedProduct,
    SignedQuotient,
    SignedRemainder,
    SignedShiftLeft,
    SignedShiftRight,
}

impl Limbed {
    fn emit(self, e: &mut HLBlockEmitter<'_>, lhs: ValueId, rhs: ValueId) -> ValueId {
        match self {
            Limbed::Sum => e.bin(BinaryArithOpKind::UAdd, lhs, rhs),
            Limbed::Difference => e.bin(BinaryArithOpKind::USub, lhs, rhs),
            Limbed::Ordering => e.cmp(lhs, rhs, CmpKind::ULt),
            Limbed::Product => e.bin(BinaryArithOpKind::UMul, lhs, rhs),
            Limbed::Quotient => e.bin(BinaryArithOpKind::UDiv, lhs, rhs),
            Limbed::Remainder => e.bin(BinaryArithOpKind::URem, lhs, rhs),
            Limbed::ShiftLeft => e.bin(BinaryArithOpKind::UShl, lhs, rhs),
            Limbed::ShiftRight => e.bin(BinaryArithOpKind::UShr, lhs, rhs),
            Limbed::SignedSum => e.bin(BinaryArithOpKind::SAdd, lhs, rhs),
            Limbed::SignedDifference => e.bin(BinaryArithOpKind::SSub, lhs, rhs),
            Limbed::SignedOrdering => e.cmp(lhs, rhs, CmpKind::SLt),
            Limbed::SignedProduct => e.bin(BinaryArithOpKind::SMul, lhs, rhs),
            Limbed::SignedQuotient => e.bin(BinaryArithOpKind::SDiv, lhs, rhs),
            Limbed::SignedRemainder => e.bin(BinaryArithOpKind::SRem, lhs, rhs),
            Limbed::SignedShiftLeft => e.bin(BinaryArithOpKind::SShl, lhs, rhs),
            Limbed::SignedShiftRight => e.bin(BinaryArithOpKind::SShr, lhs, rhs),
        }
    }

    /// The right operand a pair takes on a run that does not name it: one that every left operand
    /// is accepted against, which is zero except for a divisor.
    fn idle_rhs(self, bits: usize) -> IntBits {
        match self {
            Limbed::Quotient
            | Limbed::Remainder
            | Limbed::SignedQuotient
            | Limbed::SignedRemainder => IntBits::from_u128(bits, 1),
            Limbed::Sum
            | Limbed::Difference
            | Limbed::Ordering
            | Limbed::Product
            | Limbed::ShiftLeft
            | Limbed::ShiftRight
            | Limbed::SignedSum
            | Limbed::SignedDifference
            | Limbed::SignedOrdering
            | Limbed::SignedProduct
            | Limbed::SignedShiftLeft
            | Limbed::SignedShiftRight => IntBits::zero(bits),
        }
    }

    /// The model's answer, or [`None`] where it rejects the operands.
    fn model(self, lhs: &IntBits, rhs: &IntBits) -> Option<IntBits> {
        match self {
            Limbed::Sum => mavros_int_semantics::eval(IntOp::UAdd, lhs, rhs).value(),
            Limbed::Difference => mavros_int_semantics::eval(IntOp::USub, lhs, rhs).value(),
            Limbed::Product => mavros_int_semantics::eval(IntOp::UMul, lhs, rhs).value(),
            Limbed::Quotient => mavros_int_semantics::eval(IntOp::UDiv, lhs, rhs).value(),
            Limbed::Remainder => mavros_int_semantics::eval(IntOp::URem, lhs, rhs).value(),
            Limbed::ShiftLeft | Limbed::SignedShiftLeft => {
                mavros_int_semantics::eval(IntOp::Shl, lhs, rhs).value()
            }
            Limbed::ShiftRight => mavros_int_semantics::eval(IntOp::UShr, lhs, rhs).value(),
            Limbed::SignedSum => mavros_int_semantics::eval(IntOp::SAdd, lhs, rhs).value(),
            Limbed::SignedDifference => mavros_int_semantics::eval(IntOp::SSub, lhs, rhs).value(),
            Limbed::SignedProduct => mavros_int_semantics::eval(IntOp::SMul, lhs, rhs).value(),
            Limbed::SignedQuotient => mavros_int_semantics::eval(IntOp::SDiv, lhs, rhs).value(),
            Limbed::SignedRemainder => mavros_int_semantics::eval(IntOp::SRem, lhs, rhs).value(),
            Limbed::SignedShiftRight => mavros_int_semantics::eval(IntOp::SShr, lhs, rhs).value(),
            Limbed::Ordering => Some(IntBits::from_u128(
                1,
                u128::from(BigUint::from(lhs) < BigUint::from(rhs)),
            )),
            Limbed::SignedOrdering => Some(IntBits::from_u128(
                1,
                u128::from(lhs.compare(CmpOp::SLt, rhs)),
            )),
        }
    }
}

/// The chain on limbs, over the corners a carry chain turns on, on both lanes.
///
/// The corners sit either side of the lowest limb boundary and of the top limb's, plus the
/// extremes. 254 has a narrow top limb, 320 a full one. The limb is the **field's** witness limb;
/// only on bn254 is it also the host word.
///
/// [`the_limbed_corner_matrix_agrees_with_the_model`] runs every corner in
/// [`corners::wide_values`], which sit at the host word; the matrix asserts the two limbs coincide.
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
#[ignore = "every limb corner of eight operations; well over an hour in a debug build"]
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
        let values = dedup(values);
        for limbed in [Limbed::Product, Limbed::Quotient, Limbed::Remainder] {
            check_limbed_corners(limbed, bits, &values, Rhs::Witnessed);
        }
        let amounts = limb_shift_amounts(bits, HOST_LIMB_BITS);
        for limbed in [Limbed::ShiftLeft, Limbed::ShiftRight] {
            check_limbed_shifts(limbed, bits, &values, &amounts, Rhs::Witnessed);
        }
    }
}

/// The sign-magnitude gadgets over every signed limb corner, and the powers of two whose product
/// lands either side of `INT_MIN`.
///
/// 127 is the gadgets on a decomposition of each operand, 128 an `i128`, 254 limbs with a narrow
/// top limb, and 320 a full one.
#[test]
#[ignore = "every signed limb corner of four operations at four widths; over an hour in a debug build"]
fn the_signed_limbed_corner_matrix_agrees_with_the_model() {
    for bits in [127usize, 128, 254, 320] {
        let signed = |v: SignedValue| IntBits::from_signed(bits, &v);
        let power = |place: usize| SignedValue::from(BigUint::from(1u8) << place);
        let (low, high) = (bits / 2, bits - 1 - bits / 2);
        let mut values = signed_limb_corners(bits);
        values.extend([
            signed(power(low)),
            signed(-power(low)),
            signed(power(high)),
            signed(-power(high)),
            signed(power(high) + SignedValue::from(1)),
        ]);
        let values = dedup(values);
        for limbed in [
            Limbed::SignedProduct,
            Limbed::SignedQuotient,
            Limbed::SignedRemainder,
        ] {
            check_limbed_corners(limbed, bits, &values, Rhs::Witnessed);
        }
        check_limbed_shifts(
            Limbed::SignedShiftRight,
            bits,
            &values,
            &limb_shift_amounts(bits, HOST_LIMB_BITS),
            Rhs::Witnessed,
        );
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

/// A quotient and a remainder agree with the model where the operands have an element each but the
/// product that checks them does not, so the representation divides on a decomposition of each.
///
/// 127 is the one width below the double lane the single cell cannot check, 129 and 200 are past
/// both narrow lanes, and 253 is the widest an element carries. A zero divisor must be refused.
///
/// Witness taint inference also gives each program a **pure** copy of the division, whose zero
/// check is built at the same width, so each program compiling shows both copies are lowered.
#[test]
fn a_quotient_or_remainder_agrees_with_the_model_at_the_limb_corners() {
    for bits in [127usize, 129, 200, 253] {
        let mut values = corners::wide_values(bits);
        values.extend(half_width_corners(bits));
        let values = dedup(values);
        let pairs: Vec<(IntBits, IntBits)> = values
            .iter()
            .flat_map(|a| values.iter().map(move |b| (a.clone(), b.clone())))
            .collect();
        for kind in [BinaryArithOpKind::UDiv, BinaryArithOpKind::URem] {
            sweep_over(kind, bits, pairs.clone());
        }
    }
}

/// The product on limbs, over the corners its carries and its overflow check turn on, on both
/// lanes, with both operands witnessed and with a constant multiplier.
///
/// The limbs a constant does not reach drop out of the plan with their partial products and
/// overflow terms. `2^top_limb` has only a top limb, so its overflow terms are all that stand
/// between a large operand and a wrong answer.
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

/// The quotient and the remainder on limbs, over the corners the schoolbook and the chain turn on,
/// on both lanes, with a witnessed divisor and with a constant one.
///
/// A constant divisor bounds both answers, so limbs it puts out of their reach are never witnessed
/// and `r < d` reads only the limbs below its top. `2^top_limb` leaves the most limbs out, and
/// `2^N - 1` leaves none. A constant zero divisor refuses the whole program, so it has a separate
/// test, [`a_constant_zero_divisor_is_refused_at_every_width`].
#[test]
fn limbed_quotients_and_remainders_agree_with_the_model() {
    let limb = witness_limb_bits(FieldConfig::bn254());
    for bits in [254usize, 320] {
        let one = IntBits::from_u128(bits, 1);
        let top_limb = (bits - 1) / limb * limb;
        let mut values = vec![
            IntBits::zero(bits),
            one.clone(),
            IntBits::from_u128(bits, 3),
            IntBits::all_ones(limb).cast(bits),
            one.shifted_left(limb),
            one.shifted_left(top_limb),
            IntBits::all_ones(bits),
        ];
        values.extend(half_width_corners(bits));
        let values = dedup(values);
        for limbed in [Limbed::Quotient, Limbed::Remainder] {
            check_limbed_corners(limbed, bits, &values, Rhs::Witnessed);
            let pairs: Vec<(IntBits, IntBits)> = values
                .iter()
                .flat_map(|a| {
                    values
                        .iter()
                        .filter(|b| !b.is_zero())
                        .map(move |b| (a.clone(), b.clone()))
                })
                .collect();
            for pairs in pairs.chunks(20) {
                check_limbed_pairs(limbed, bits, pairs.to_vec(), Rhs::Constant);
            }
        }
    }
}

/// A shift agrees with the model either side of where the single cell stops taking it.
///
/// 37 and 96 are single-cell widths of no special shape, where the amount check is a real
/// `amount < bits` and the table has more rows than there are amounts. 126 is the widest right shift
/// the single cell takes and 127 the widest left one; past each, and at 128 for both, the shift is
/// limb-wise on a decomposition of the one element each operand still is.
#[test]
fn a_shift_agrees_with_the_model_either_side_of_the_single_cell() {
    for bits in [37usize, 96, 126, 127, 128] {
        for kind in [BinaryArithOpKind::UShl, BinaryArithOpKind::UShr] {
            sweep(kind, bits);
        }
    }
    for kind in [BinaryArithOpKind::SShl, BinaryArithOpKind::SShr] {
        sweep(kind, 37);
    }
}

/// A shift agrees with the model where the operands have an element each and the single cell
/// cannot hold the shift, over the wide corners, which put the amount either side of a limb, of half
/// the width and of the width.
///
/// 129 is the first width past the double lane. 192 is three limbs, so its amount's decomposition
/// reaches past the width and the check is explicit. 200 has a narrow top limb, which a left shift
/// cuts to its width. 253 is the widest width an element carries.
#[test]
fn a_shift_agrees_with_the_model_at_the_limb_corners() {
    for bits in [129usize, 192, 200, 253] {
        for kind in [BinaryArithOpKind::UShl, BinaryArithOpKind::UShr] {
            sweep_over(kind, bits, wide_corner_pairs(IntOp::from(kind), bits));
        }
    }
}

/// The shift on limbs, by a witnessed amount and by a constant one, on both lanes.
///
/// The amounts are the wide corners plus every whole number of limbs and one either side of each:
/// where a witnessed amount's barrel moves a limb and a constant one stops cutting any. 254 has a
/// narrow top limb and a limbed amount; 256 is four limbs, whose amount decomposition alone bounds
/// it; 320 is five, whose barrel has a stage it only half uses. The operands reach every limb,
/// including one of alternating bits, so a piece put in the wrong place gives a different answer.
#[test]
fn limbed_shifts_agree_with_the_model() {
    let limb = witness_limb_bits(FieldConfig::bn254());
    for bits in [254usize, 256, 320] {
        let values = limb_reaching_values(bits, limb);
        let amounts = limb_shift_amounts(bits, limb);
        let legal: Vec<IntBits> = amounts
            .iter()
            .filter(|amount| BigUint::from(*amount) < BigUint::from(bits))
            .cloned()
            .collect();
        for limbed in [Limbed::ShiftLeft, Limbed::ShiftRight] {
            check_limbed_shifts(limbed, bits, &values, &amounts, Rhs::Witnessed);
            check_limbed_shifts(limbed, bits, &values, &legal, Rhs::Constant);
        }
    }
}

/// The shift at the widest widths, where the barrel is deepest: 1000 is 16 limbs with a narrow top
/// one, 16383 is 256 with a narrow top one, and 16384 is 256 full ones, eight stages deep.
///
/// The two widest test one pair per program: a wasm function holds at most 50 000 locals, and four
/// 16383-bit shifts by a witnessed amount exceed that.
#[test]
#[ignore = "shifts of up to 16384 bits; about half an hour in a debug build"]
fn a_shift_agrees_with_the_model_at_the_widest_widths() {
    let limb = witness_limb_bits(FieldConfig::bn254());
    for bits in [1000usize, 16383, 16384] {
        let one = IntBits::from_u128(bits, 1);
        let every_other =
            IntBits::from_biguint(bits, &(((BigUint::from(1u8) << bits) - 1u8) / 3u8));
        let (values, amounts, per_program) = if bits <= 1000 {
            let values = vec![one, IntBits::all_ones(bits), every_other];
            (values, few_shift_amounts(bits, limb), 4)
        } else {
            let amounts = [1, limb - 1, limb + 1, bits / 2 + 3, bits - 1, bits]
                .into_iter()
                .map(|amount| IntBits::from_u128(bits, amount as u128))
                .collect();
            (vec![every_other], amounts, 1)
        };
        let legal: Vec<IntBits> = amounts[..amounts.len() - 1].to_vec();
        for limbed in [Limbed::ShiftLeft, Limbed::ShiftRight] {
            for (rhs, amounts) in [
                (Rhs::Witnessed, &amounts),
                (Rhs::Constant, &legal),
                (Rhs::AgainstPure, &amounts),
            ] {
                let pairs: Vec<(IntBits, IntBits)> = values
                    .iter()
                    .flat_map(|value| {
                        amounts
                            .iter()
                            .map(move |amount| (value.clone(), amount.clone()))
                    })
                    .collect();
                for pairs in pairs.chunks(per_program) {
                    check_limbed_pairs(limbed, bits, pairs.to_vec(), rhs);
                }
            }
        }
    }
}

/// A pure operand shifted by a witnessed amount, on both lanes: `1 << n` and similar.
///
/// A limb the pure operand does not reach is a known zero whose halves are never witnessed; every
/// other limb's split is a pure limb times a witnessed power of two. `1` skips every limb but the
/// lowest, and alternating bits reach every limb. 200 is one element split on the pure side, 320
/// is limbs. The amounts are only [`few_shift_amounts`]; [`limbed_shifts_agree_with_the_model`]
/// covers every limb corner for a witnessed operand.
#[test]
fn a_pure_value_shifted_by_a_witnessed_amount_agrees_with_the_model() {
    let limb = witness_limb_bits(FieldConfig::bn254());
    for bits in [200usize, 320] {
        let every_other = ((BigUint::from(1u8) << bits) - 1u8) / 3u8;
        let values = [
            IntBits::from_u128(bits, 1),
            IntBits::from_biguint(bits, &every_other),
        ];
        let amounts = few_shift_amounts(bits, limb);
        for limbed in [Limbed::ShiftLeft, Limbed::ShiftRight] {
            check_limbed_shifts(limbed, bits, &values, &amounts, Rhs::AgainstPure);
        }
    }
}

/// A witnessed value shifted by a **pure** amount that a loop computes, which no fold answers.
///
/// Such an amount is known wherever the constraints are built, so the shift witnesses nothing about
/// it: its check is a comparison, and each limb is split by a pure power of two. The loop shifts by
/// every multiple of 37 below the width, which moves whole limbs and splits them at a different
/// place each time, and accumulates the answers, which the program asserts equal to the model's.
/// Clearing the bit that selects the operand makes every answer zero, which the assertion refuses.
/// One more iteration reaches past the width, which the comparison refuses. 200 is the operand as
/// one element, 320 as limbs.
#[test]
fn a_shift_by_a_pure_amount_a_loop_computes_agrees_with_the_model() {
    let step = 37usize;
    for bits in [200usize, 320] {
        let every_other = ((BigUint::from(1u8) << bits) - 1u8) / 3u8;
        let operand = IntBits::from_biguint(bits, &every_other);
        let legal = (bits - 1) / step + 1;
        for kind in [BinaryArithOpKind::UShl, BinaryArithOpKind::UShr] {
            let program = |iterations: usize, expected: IntBits| {
                let operand = operand.clone();
                main_program(&[Type::int(1)], &[], move |e, params| {
                    let pattern = e.int_const(operand);
                    let zero = e.int_const(IntBits::zero(bits));
                    let value = e.select(params[0], pattern, zero);
                    let start = e.int_const(IntBits::zero(32));
                    let (head, _) = e.add_block();
                    let (body, _) = e.add_block();
                    let (done, _) = e.add_block();
                    e.seal_and_switch(Terminator::Jmp(head, vec![start, zero]), head);
                    let count = e.add_parameter(Type::int(32));
                    let sum = e.add_parameter(Type::int(bits));
                    let limit = e.int_const(IntBits::from_u128(32, iterations as u128));
                    let more = e.cmp(count, limit, CmpKind::ULt);
                    e.seal_and_switch(Terminator::JmpIf(more, body, done), body);
                    let index = e.cast_to(CastTarget::Int(bits), count);
                    let stride = e.int_const(IntBits::from_u128(bits, step as u128));
                    let amount = e.bin(BinaryArithOpKind::UMul, index, stride);
                    let shifted = e.bin(kind, value, amount);
                    let next_sum = e.bin(BinaryArithOpKind::Xor, sum, shifted);
                    let one = e.int_const(IntBits::from_u128(32, 1));
                    let next = e.bin(BinaryArithOpKind::UAdd, count, one);
                    e.seal_and_switch(Terminator::Jmp(head, vec![next, next_sum]), done);
                    let expected = e.int_const(expected);
                    e.assert_eq(sum, expected);
                    vec![]
                })
            };
            let what = format!("int{bits} {kind:?} by a pure amount");

            let expected = (0..legal).fold(IntBits::zero(bits), |sum, index| {
                let amount = IntBits::from_u128(bits, (index * step) as u128);
                let shifted = mavros_int_semantics::eval(IntOp::from(kind), &operand, &amount)
                    .value()
                    .expect("an amount below the width is accepted");
                sum.xor(&shifted)
            });
            let compiled = Compiled::new(program(legal, expected))
                .unwrap_or_else(|error| panic!("{what} does not compile: {error}"));
            let selected = |bit: u128| input_block(&[&IntBits::from_u128(1, bit)]);
            let verdict = compiled.run(&selected(1));
            assert!(verdict.is_accepted(), "{what}: {verdict:?}");
            let verdict = compiled
                .run_wasm(&selected(1))
                .expect("the WASM lane builds");
            assert!(verdict.is_accepted(), "{what}, WASM: {verdict:?}");
            let verdict = compiled.run(&selected(0));
            assert!(
                !verdict.is_accepted(),
                "{what}, the operand zero: {verdict:?}"
            );

            let past = Compiled::new(program(legal + 1, IntBits::zero(bits)));
            let refused = match past {
                Err(_) => true,
                Ok(compiled) => !compiled.run(&selected(1)).is_accepted(),
            };
            assert!(refused, "{what} past the width is accepted");
        }
    }
}

/// A signed value at a width that is not a power of two, shifted by a pure amount a loop computes,
/// under a witnessed condition.
///
/// A pure amount's factor is built on the pure side, modulo the width, so the out-of-range 40
/// against 37 bits leaves an untaken branch with an in-range factor, and the amount check refuses
/// it only where the branch is taken.
#[test]
fn a_guarded_signed_shift_by_a_pure_amount_is_checked_where_the_guard_holds() {
    let bits = 37usize;
    let operand = IntBits::from_u128(bits, 0x1_2345_6789);
    for kind in [BinaryArithOpKind::SShl, BinaryArithOpKind::SShr] {
        for base in [3u128, 40] {
            check_guarded_loop_shift(kind, &operand, base);
        }
    }
}

/// An unsigned value past the single cell, shifted by a pure amount a loop computes, under a
/// witnessed condition.
///
/// The amount is known wherever the constraints are built, guard or not, so the limb-wise shift
/// compares it under the guard and splits and moves the limbs by it on every path, without zeroing
/// it where the guard is off. The comparison refuses the width and past it where the branch is
/// taken, and the split and the barrel have to hold for those amounts where it is not. 128 is one
/// element, 200 one element with a narrow top limb, and 320 limbs.
#[test]
fn a_pure_amount_under_a_witnessed_guard_is_checked_where_the_guard_holds() {
    for bits in [128usize, 200, 320] {
        let every_other = ((BigUint::from(1u8) << bits) - 1u8) / 3u8;
        let operand = IntBits::from_biguint(bits, &every_other);
        for kind in [BinaryArithOpKind::UShl, BinaryArithOpKind::UShr] {
            for base in [70, bits as u128, bits as u128 + 50] {
                check_guarded_loop_shift(kind, &operand, base);
            }
        }
    }
}

/// `operand` shifted by `base` under a witnessed condition, inside a loop that makes the amount a
/// pure value that is never a literal, checked against the model with the branch taken and not.
///
/// The operand is witnessed by folding it into a narrow input the run sets to zero, which lets it
/// be wider than a program input, and the program answers whether the shift equals the model's.
/// Not taken, it is accepted whatever the amount; taken, it answers yes, or is refused where the
/// model rejects the shift.
fn check_guarded_loop_shift(kind: BinaryArithOpKind, operand: &IntBits, base: u128) {
    let bits = operand.bits();
    let amount = IntBits::from_u128(bits, base);
    let expected = mavros_int_semantics::eval(IntOp::from(kind), operand, &amount).value();
    let pattern = operand.clone();
    let model = expected.clone().unwrap_or_else(|| IntBits::zero(bits));
    let ssa = main_program(
        &[Type::int(1), Type::int(64)],
        &[Type::int(1)],
        move |e, params| {
            let narrow = e.cast_to(CastTarget::Int(bits), params[1]);
            let pattern = e.int_const(pattern.clone());
            let operand = e.bin(BinaryArithOpKind::Xor, narrow, pattern);
            let model = e.int_const(model.clone());
            let no = e.int_const(IntBits::zero(1));
            let start = e.int_const(IntBits::zero(32));
            let (head, _) = e.add_block();
            let (body, _) = e.add_block();
            let (taken, _) = e.add_block();
            let (skipped, _) = e.add_block();
            let (merge, _) = e.add_block();
            let (done, _) = e.add_block();
            e.seal_and_switch(Terminator::Jmp(head, vec![start, no]), head);
            let count = e.add_parameter(Type::int(32));
            let last = e.add_parameter(Type::int(1));
            let limit = e.int_const(IntBits::from_u128(32, 2));
            let more = e.cmp(count, limit, CmpKind::ULt);
            e.seal_and_switch(Terminator::JmpIf(more, body, done), body);

            // Always `base`, but only once the loop has run, so never a literal.
            let thousand = e.int_const(IntBits::from_u128(32, 1000));
            let nothing = e.bin(BinaryArithOpKind::UDiv, count, thousand);
            let nothing = e.cast_to(CastTarget::Int(bits), nothing);
            let base = e.int_const(IntBits::from_u128(bits, base));
            let amount = e.bin(BinaryArithOpKind::UAdd, nothing, base);
            e.seal_and_switch(Terminator::JmpIf(params[0], taken, skipped), taken);
            let shifted = e.bin(kind, operand, amount);
            let agrees = e.cmp(shifted, model, CmpKind::Eq);
            e.seal_and_switch(Terminator::Jmp(merge, vec![agrees]), skipped);
            e.seal_and_switch(Terminator::Jmp(merge, vec![no]), merge);
            let merged = e.add_parameter(Type::int(1));
            let one = e.int_const(IntBits::from_u128(32, 1));
            let next = e.bin(BinaryArithOpKind::UAdd, count, one);
            e.seal_and_switch(Terminator::Jmp(head, vec![next, merged]), done);
            vec![last]
        },
    );
    let what = format!("int{bits} {kind:?} by the pure amount {base}");
    let compiled =
        Compiled::new(ssa).unwrap_or_else(|error| panic!("{what} does not compile: {error}"));
    let run = |taken: u128, answer: u128| {
        compiled.run(&input_block(&[
            &IntBits::from_u128(1, taken),
            &IntBits::zero(64),
            &IntBits::from_u128(1, answer),
        ]))
    };

    let verdict = run(0, 0);
    assert!(verdict.is_accepted(), "{what}, not taken: {verdict:?}");
    let verdict = run(1, 1);
    if expected.is_some() {
        assert!(verdict.is_accepted(), "{what}, taken: {verdict:?}");
    } else {
        assert!(verdict.is_refusal(), "{what}, taken: {verdict:?}");
    }
}

/// A few shift amounts at `bits`: none, either side of the first limb, one in the middle that moves
/// limbs and cuts them, the widest legal one, and the width itself, which the model rejects. The
/// last is last, so that dropping it leaves the legal ones.
fn few_shift_amounts(bits: usize, limb: usize) -> Vec<IntBits> {
    [0, 1, limb - 1, limb, limb + 1, bits / 2 + 3, bits - 1, bits]
        .into_iter()
        .map(|amount| IntBits::from_u128(bits, amount as u128))
        .collect()
}

/// Operands of `bits` that reach every limb: the top bit alone, the lowest limb full, every bit,
/// and every other bit.
fn limb_reaching_values(bits: usize, limb: usize) -> Vec<IntBits> {
    let one = IntBits::from_u128(bits, 1);
    let top_limb = (bits - 1) / limb * limb;
    let every_other = ((BigUint::from(1u8) << bits) - 1u8) / 3u8;
    vec![
        one.clone(),
        IntBits::all_ones(limb).cast(bits),
        one.shifted_left(top_limb),
        one.shifted_left(bits - 1),
        IntBits::all_ones(bits),
        IntBits::from_biguint(bits, &every_other),
    ]
}

/// The wide shift amounts at `bits`, and every whole number of `limb`-wide limbs with one either
/// side of it.
fn limb_shift_amounts(bits: usize, limb: usize) -> Vec<IntBits> {
    let mut amounts = corners::wide_shift_amounts(bits);
    for whole in (limb..bits).step_by(limb) {
        for amount in [whole - 1, whole, whole + 1] {
            amounts.push(IntBits::from_u128(bits, amount as u128));
        }
    }
    dedup(amounts)
}

/// Every value shifted by every amount, checked by [`check_limbed_pairs`] a program's worth at a
/// time.
fn check_limbed_shifts(
    limbed: Limbed,
    bits: usize,
    values: &[IntBits],
    amounts: &[IntBits],
    rhs: Rhs,
) {
    let pairs: Vec<(IntBits, IntBits)> = values
        .iter()
        .flat_map(|value| {
            amounts
                .iter()
                .map(move |amount| (value.clone(), amount.clone()))
        })
        .collect();
    for pairs in pairs.chunks(20) {
        check_limbed_pairs(limbed, bits, pairs.to_vec(), rhs);
    }
}

/// A shift by a constant at or past the width is refused at compile time: the model rejects it
/// whatever it shifts, so the lowering asserts what cannot hold. 128 is the element, 200 the element
/// with a narrow top limb, and 320 limbs.
#[test]
fn a_constant_shift_amount_past_the_width_is_refused() {
    for bits in [128usize, 200, 320] {
        for kind in [BinaryArithOpKind::UShl, BinaryArithOpKind::UShr] {
            for amount in [bits, bits + 1] {
                let ssa = main_program(&[Type::int(64)], &[Type::int(64)], move |e, params| {
                    let value = e.cast_to(CastTarget::Int(bits), params[0]);
                    let amount = e.int_const(IntBits::from_u128(bits, amount as u128));
                    let shifted = e.bin(kind, value, amount);
                    vec![e.cast_to(CastTarget::Int(64), shifted)]
                });
                let Err(error) = Compiled::new(ssa) else {
                    panic!("int{bits} {kind:?} by the constant {amount} compiled");
                };
                let error = error.to_string();
                assert!(error.contains("assert_cmp eq failed"), "{error}");
            }
        }
    }
}

/// A shift under a witnessed condition is checked only where the condition holds.
///
/// Where the branch is not taken the amount is one past the width, and the lowering zeroes it
/// rather than letting its checks see it; a constant amount past the width is the same case with
/// an assertion. 128 is the limb-wise shift on one element's decomposition, 200 adds the amount
/// check a narrow top limb needs, and 320 is limbs. Here the amount's upper limbs are known zeros;
/// [`a_wide_amount_is_held_below_its_second_limb_only_where_its_guard_holds`] sets one.
#[test]
fn a_guarded_shift_is_checked_only_where_its_guard_holds() {
    for bits in [128usize, 200, 320] {
        let high = IntBits::from_u128(bits, 1).shifted_left(bits - 1);
        for kind in [BinaryArithOpKind::UShl, BinaryArithOpKind::UShr] {
            for constant in [None, Some(bits)] {
                let top = high.clone();
                let ssa = main_program(
                    &[Type::int(1), Type::int(64), Type::int(64)],
                    &[Type::int(64)],
                    move |e, params| {
                        let value = e.cast_to(CastTarget::Int(bits), params[1]);
                        let top = e.int_const(top.clone());
                        let value = e.bin(BinaryArithOpKind::Or, value, top);
                        let amount = match constant {
                            Some(amount) => e.int_const(IntBits::from_u128(bits, amount as u128)),
                            None => e.cast_to(CastTarget::Int(bits), params[2]),
                        };

                        let (then_id, _) = e.add_block();
                        let (else_id, _) = e.add_block();
                        let (merge_id, _) = e.add_block();
                        e.seal_and_switch(Terminator::JmpIf(params[0], then_id, else_id), then_id);
                        let shifted = e.bin(kind, value, amount);
                        let low = e.cast_to(CastTarget::Int(64), shifted);
                        e.seal_and_switch(Terminator::Jmp(merge_id, vec![low]), else_id);
                        let zero = e.int_const(IntBits::zero(64));
                        e.seal_and_switch(Terminator::Jmp(merge_id, vec![zero]), merge_id);
                        vec![e.add_parameter(Type::int(64))]
                    },
                );
                let what = format!("a guarded int{bits} {kind:?} by {constant:?}");
                let compiled = Compiled::new(ssa)
                    .unwrap_or_else(|error| panic!("{what} does not compile: {error}"));

                let run = |taken: u128, amount: u128, answer: u128| {
                    compiled.run(&input_block(&[
                        &IntBits::from_u128(1, taken),
                        &IntBits::from_u128(64, 5),
                        &IntBits::from_u128(64, amount),
                        &IntBits::from_u128(64, answer),
                    ]))
                };

                let past = bits as u128;
                let verdict = run(0, past, 0);
                assert!(
                    verdict.is_accepted(),
                    "{what}: not taken, by {past}: {verdict:?}"
                );
                let verdict = run(1, past, 0);
                assert!(
                    verdict.is_refusal(),
                    "{what}: taken, by {past}: {verdict:?}"
                );
                if constant.is_none() {
                    let verdict = run(0, u128::from(u64::MAX), 0);
                    assert!(
                        verdict.is_accepted(),
                        "{what}: not taken, by 2^64 - 1: {verdict:?}"
                    );

                    let value = IntBits::from_u128(bits, 5).or(&high);
                    let amount = IntBits::from_u128(bits, 3);
                    let answer = mavros_int_semantics::eval(IntOp::from(kind), &value, &amount)
                        .value()
                        .expect("a shift by three is accepted");
                    let answer = usize::try_from(&answer.cast(64)).unwrap() as u128;
                    let verdict = run(1, 3, answer);
                    assert!(verdict.is_accepted(), "{what}: taken, by 3: {verdict:?}");
                }
            }
        }
    }
}

/// A witnessed amount held as limbs is held to zero above its lowest limb only where its guard holds.
///
/// A bit in an upper limb names a shift past any width, so where the branch is taken the program is
/// refused; where it is not, the bit is whatever that branch left behind and has to be admitted. The
/// amount is a small input with bit 200 set, which only a runtime value can put there: a widened
/// input's upper limbs are known zeros, and the constraint on a known zero holds either way.
#[test]
fn a_wide_amount_is_held_below_its_second_limb_only_where_its_guard_holds() {
    let bits = 320usize;
    let high = IntBits::from_u128(bits, 1).shifted_left(200);
    for kind in [BinaryArithOpKind::UShl, BinaryArithOpKind::UShr] {
        let high = high.clone();
        let ssa = main_program(
            &[Type::int(1), Type::int(64), Type::int(64)],
            &[Type::int(64)],
            move |e, params| {
                let value = e.cast_to(CastTarget::Int(bits), params[1]);
                let amount = e.cast_to(CastTarget::Int(bits), params[2]);
                let high = e.int_const(high.clone());
                let amount = e.bin(BinaryArithOpKind::Xor, amount, high);

                let (then_id, _) = e.add_block();
                let (else_id, _) = e.add_block();
                let (merge_id, _) = e.add_block();
                e.seal_and_switch(Terminator::JmpIf(params[0], then_id, else_id), then_id);
                let shifted = e.bin(kind, value, amount);
                let low = e.cast_to(CastTarget::Int(64), shifted);
                e.seal_and_switch(Terminator::Jmp(merge_id, vec![low]), else_id);
                let zero = e.int_const(IntBits::zero(64));
                e.seal_and_switch(Terminator::Jmp(merge_id, vec![zero]), merge_id);
                vec![e.add_parameter(Type::int(64))]
            },
        );
        let what = format!("a guarded int{bits} {kind:?} by an amount past its lowest limb");
        let compiled =
            Compiled::new(ssa).unwrap_or_else(|error| panic!("{what} does not compile: {error}"));
        let run = |taken: u128| {
            compiled.run(&input_block(&[
                &IntBits::from_u128(1, taken),
                &IntBits::from_u128(64, 5),
                &IntBits::from_u128(64, 3),
                &IntBits::zero(64),
            ]))
        };

        let verdict = run(0);
        assert!(verdict.is_accepted(), "{what}, not taken: {verdict:?}");
        let verdict = run(1);
        assert!(verdict.is_refusal(), "{what}, taken: {verdict:?}");
    }
}

/// The mutation test for the limb-wise shift, by a witnessed amount and by a constant one.
///
/// The operand is a constant reaching every limb, selected by a witnessed bit so that it is a
/// witnessed value; the amount is a widened parameter or the constant 70, which moves every limb
/// one place and splits each at six. The answer leaves through its low limb. 128 is one element,
/// whose amount decomposition alone bounds it; 200 is one element that needs the explicit check;
/// 320 is limbs.
///
/// A `>>` by the constant reaches the limb-wise shift only at 320. At widths one element carries,
/// the `Simplifier` that runs before taint inference turns it into a `BitRange`, so that lowering
/// is what gets pinned there, and the `<<` by the constant pins the cut.
///
/// **The amount is also moved alone**, by [`assert_changing_an_input_breaks_the_witness`]: the
/// split and the barrel read it only through its decomposition, so `n = h·Σe_t·2^t + r` has to
/// refuse a decomposition left behind by a different amount, and no single column of the honest
/// run moving shows that. The operand is moved alone too, which the split must notice with the
/// halves of the old one still in place.
#[test]
fn every_witness_column_of_a_limbed_shift_is_pinned() {
    let selected = IntBits::from_u128(1, 1);
    let amount = 70u128;
    for bits in [128usize, 200, 320] {
        let every_other = ((BigUint::from(1u8) << bits) - 1u8) / 3u8;
        let operand = IntBits::from_biguint(bits, &every_other);
        for kind in [BinaryArithOpKind::UShl, BinaryArithOpKind::UShr] {
            let answer = mavros_int_semantics::eval(
                IntOp::from(kind),
                &operand,
                &IntBits::from_u128(bits, amount),
            )
            .value()
            .expect("a shift by 70 is accepted")
            .cast(64);

            for constant in [false, true] {
                let pattern = operand.clone();
                let ssa = main_program(
                    &[Type::int(1), Type::int(64)],
                    &[Type::int(64)],
                    move |e, params| {
                        let value = e.int_const(pattern.clone());
                        let zero = e.int_const(IntBits::zero(bits));
                        let value = e.select(params[0], value, zero);
                        let by = if constant {
                            e.int_const(IntBits::from_u128(bits, amount))
                        } else {
                            e.cast_to(CastTarget::Int(bits), params[1])
                        };
                        let shifted = e.bin(kind, value, by);
                        vec![e.cast_to(CastTarget::Int(64), shifted)]
                    },
                );
                let what = format!(
                    "{kind:?} at {bits} bits by {}",
                    if constant { "a constant" } else { "a witness" }
                );
                let compiled = Compiled::new(ssa)
                    .unwrap_or_else(|error| panic!("{what} does not compile: {error}"));
                let by = IntBits::from_u128(64, amount);
                let inputs = [&selected, &by, &answer];
                assert_every_column_is_pinned(&what, &compiled, &inputs);
                assert_changing_an_input_breaks_the_witness(
                    &format!("{what}, the operand sent to zero"),
                    &compiled,
                    &inputs,
                    0,
                    &IntBits::zero(1),
                );
                if !constant {
                    assert_changing_an_input_breaks_the_witness(
                        &format!("{what}, the amount moved by one"),
                        &compiled,
                        &inputs,
                        1,
                        &IntBits::from_u128(64, amount + 1),
                    );
                }
            }
        }
    }
}

/// A division whose answer is known before it runs still refuses a zero divisor.
///
/// Zero divided by anything leaves every limb of both answers a known zero, so neither is witnessed
/// and the schoolbook holds only the dividend's limbs. A value divided by itself is one, with no
/// quotient or remainder witnessed at all. In both cases the lowering's check of the divisor is
/// the only thing refusing a zero divisor, as a division by a witness carries no other check. Each
/// runs unguarded and in a taken branch, the two paths to that check.
#[test]
fn a_division_with_a_known_answer_still_refuses_a_zero_divisor() {
    for bits in [200usize, 320] {
        for (what, same) in [("zero by d", false), ("d by d", true)] {
            for kind in [BinaryArithOpKind::UDiv, BinaryArithOpKind::URem] {
                for guarded in [false, true] {
                    let ssa = main_program(
                        &[Type::int(1), Type::int(64)],
                        &[Type::int(64)],
                        move |e, params| {
                            let divisor = e.cast_to(CastTarget::Int(bits), params[1]);
                            let dividend = if same {
                                divisor
                            } else {
                                e.int_const(IntBits::zero(bits))
                            };
                            if !guarded {
                                let answer = e.bin(kind, dividend, divisor);
                                return vec![e.cast_to(CastTarget::Int(64), answer)];
                            }

                            let (then_id, _) = e.add_block();
                            let (else_id, _) = e.add_block();
                            let (merge_id, _) = e.add_block();
                            e.seal_and_switch(
                                Terminator::JmpIf(params[0], then_id, else_id),
                                then_id,
                            );
                            let answer = e.bin(kind, dividend, divisor);
                            let low = e.cast_to(CastTarget::Int(64), answer);
                            e.seal_and_switch(Terminator::Jmp(merge_id, vec![low]), else_id);
                            let zero = e.int_const(IntBits::zero(64));
                            e.seal_and_switch(Terminator::Jmp(merge_id, vec![zero]), merge_id);
                            vec![e.add_parameter(Type::int(64))]
                        },
                    );
                    let what = format!("int{bits} {kind:?}, {what}, guarded={guarded}");
                    let compiled = Compiled::new(ssa)
                        .unwrap_or_else(|error| panic!("{what} does not compile: {error}"));

                    let answer = u128::from(same && kind == BinaryArithOpKind::UDiv);
                    let run = |divisor: u128, answer: u128| {
                        compiled.run(&input_block(&[
                            &IntBits::from_u128(1, 1),
                            &IntBits::from_u128(64, divisor),
                            &IntBits::from_u128(64, answer),
                        ]))
                    };
                    let verdict = run(7, answer);
                    assert!(verdict.is_accepted(), "{what}, by seven: {verdict:?}");
                    let verdict = run(0, 0);
                    assert!(verdict.is_refusal(), "{what}, by zero: {verdict:?}");
                }
            }
        }
    }
}

/// A division whose answers are known limb by limb still answers a witnessed value.
///
/// `0 / d` cannot fold before lowering, as it must still refuse a zero divisor, and the lowering
/// then knows every answer limb. The function declares a witnessed return, which a pure constant
/// answer would contradict.
#[test]
fn a_division_known_to_be_zero_keeps_a_witnessed_answer() {
    for bits in [200usize, 320] {
        for kind in [BinaryArithOpKind::UDiv, BinaryArithOpKind::URem] {
            let ssa = main_program(&[Type::int(64)], &[Type::int(1)], move |e, params| {
                let divisor = e.cast_to(CastTarget::Int(bits), params[0]);
                let zero = e.int_const(IntBits::zero(bits));
                let answer = e.bin(kind, zero, divisor);
                vec![e.eq(answer, zero)]
            });
            let what = format!("int{bits} {kind:?} of zero");
            let compiled = Compiled::new(ssa)
                .unwrap_or_else(|error| panic!("{what} does not compile: {error}"));
            let one = IntBits::from_u128(1, 1);
            let verdict = compiled.run(&input_block(&[&IntBits::from_u128(64, 7), &one]));
            assert!(verdict.is_accepted(), "{what}, by seven: {verdict:?}");
            let verdict = compiled.run(&input_block(&[&IntBits::zero(64), &one]));
            assert!(verdict.is_refusal(), "{what}, by zero: {verdict:?}");
        }
    }
}

/// A constant zero divisor is refused when the program is compiled, at every width.
///
/// `LowerPureGuards` asserts the divisor is not zero; with a constant divisor that assertion is a
/// constant, and R1CS generation reports it failing.
#[test]
fn a_constant_zero_divisor_is_refused_at_every_width() {
    for bits in [64usize, 128, 200, 320] {
        for kind in [BinaryArithOpKind::UDiv, BinaryArithOpKind::URem] {
            let ssa = main_program(&[Type::int(64)], &[Type::int(64)], move |e, params| {
                let dividend = e.cast_to(CastTarget::Int(bits), params[0]);
                let zero = e.int_const(IntBits::zero(bits));
                let answer = e.bin(kind, dividend, zero);
                vec![e.cast_to(CastTarget::Int(64), answer)]
            });
            let Err(error) = Compiled::new(ssa) else {
                panic!("int{bits} {kind:?} by a constant zero compiled");
            };
            let error = error.to_string();
            assert!(error.contains("assert_cmp eq failed: 1 != 0"), "{error}");
        }
    }
}

/// Every pair of `values`, checked by [`check_limbed_pairs`] one program's worth at a time.
///
/// Split because of the host, not the compiler. On aarch64, wasmtime's Cranelift (0.118) trips an
/// internal branch-range assertion, `(label_offset - offset) <= kind.max_pos_range()` in
/// `machinst/buffer.rs`, compiling a function with a few hundred pairs, while the VM lane accepts
/// the same program. Fifty sums, differences or orderings per program stay clear of it. A product
/// of two dense operands is several times their code, and fifty of those do not.
fn check_limbed_corners(limbed: Limbed, bits: usize, values: &[IntBits], rhs: Rhs) {
    let pairs: Vec<(IntBits, IntBits)> = values
        .iter()
        .flat_map(|a| values.iter().map(move |b| (a.clone(), b.clone())))
        .collect();
    // A sign-magnitude gadget is the unsigned one plus a negation of each operand and of the
    // answer, and twenty of them at 320 bits make a WASM function past the reach of an aarch64
    // branch, which Cranelift refuses.
    let per_program = match limbed {
        Limbed::SignedProduct | Limbed::SignedQuotient | Limbed::SignedRemainder => 8,
        Limbed::Product
        | Limbed::Quotient
        | Limbed::Remainder
        | Limbed::ShiftLeft
        | Limbed::ShiftRight
        | Limbed::SignedShiftLeft
        | Limbed::SignedShiftRight => 20,
        Limbed::Sum
        | Limbed::Difference
        | Limbed::Ordering
        | Limbed::SignedSum
        | Limbed::SignedDifference
        | Limbed::SignedOrdering => 50,
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

    /// Selected by a witnessed bit, as [`Rhs::Witnessed`] is, against a left operand that is the
    /// corner constant itself: a pure value put through a witnessed one, as `1 << n` is.
    AgainstPure,
}

/// `limbed` over `pairs`, all in one program, against the model.
///
/// Each operand is a corner constant selected by a witnessed bit: a witnessed value held as limbs.
///
/// A pair the model rejects would refuse every run, so its operands are selected by a second
/// parameter naming the pair and are idle otherwise: zero, or one for a divisor
/// ([`Limbed::idle_rhs`]). The pipeline must refuse on that run only. A constant right operand is
/// not selected, so only the left goes idle, to zero or to all ones where zero is rejected (as
/// `0 - INT_MIN` is); a pure left operand likewise leaves only the right to go idle.
fn check_limbed_pairs(limbed: Limbed, bits: usize, pairs: Vec<(IntBits, IntBits)>, rhs: Rhs) {
    let answers: Vec<Option<IntBits>> = pairs.iter().map(|(a, b)| limbed.model(a, b)).collect();
    let zero = IntBits::zero(bits);
    let idle_rhs: Vec<IntBits> = pairs.iter().map(|_| limbed.idle_rhs(bits)).collect();
    let idle_lhs: Vec<IntBits> = pairs
        .iter()
        .map(|(_, right)| match rhs {
            Rhs::Constant => [zero.clone(), IntBits::all_ones(bits)]
                .into_iter()
                .find(|left| limbed.model(left, right).is_some())
                .expect("the operation accepts zero or all ones against this constant"),
            Rhs::Witnessed | Rhs::AgainstPure => zero.clone(),
        })
        .collect();
    let unnamed: Vec<IntBits> = pairs
        .iter()
        .zip(&idle_lhs)
        .zip(&idle_rhs)
        .map(|(((left, right), idle_left), idle)| {
            let (left, right) = match rhs {
                Rhs::Witnessed => (idle_left, idle),
                Rhs::Constant => (idle_left, right),
                Rhs::AgainstPure => (left, idle),
            };
            limbed
                .model(left, right)
                .expect("the operation accepts an idle operand against this other one")
        })
        .collect();

    let (program_pairs, program_answers) = (pairs.clone(), answers.clone());
    let ssa = main_program(
        &[Type::int(1), Type::int(32), Type::int(1)],
        &[Type::int(1)],
        move |e, params| {
            let mut all = None;
            for (index, (((((lhs, right), answer), unnamed), idle_left), idle)) in program_pairs
                .iter()
                .zip(&program_answers)
                .zip(&unnamed)
                .zip(&idle_lhs)
                .zip(&idle_rhs)
                .enumerate()
            {
                // Each side has a separate selector, so `a op b` and `b op a` are different
                // values to the optimizer. With one selector the two are the same product, CSE
                // keeps one, and a lowering that treats its operands differently only ever runs
                // one way round.
                let (left_chosen, right_chosen, want) = match answer {
                    Some(want) => (params[0], params[2], want.clone()),
                    None => {
                        let name = e.int_const(IntBits::from_u128(32, index as u128));
                        let named = e.eq(params[1], name);
                        (named, named, unnamed.clone())
                    }
                };
                let lhs = e.int_const(lhs.clone());
                let lhs = match rhs {
                    Rhs::AgainstPure => lhs,
                    Rhs::Witnessed | Rhs::Constant => {
                        let idle_left = e.int_const(idle_left.clone());
                        e.select(left_chosen, lhs, idle_left)
                    }
                };
                let right = e.int_const(right.clone());
                let rhs = match rhs {
                    Rhs::Witnessed | Rhs::AgainstPure => {
                        let idle = e.int_const(idle.clone());
                        e.select(right_chosen, right, idle)
                    }
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
        let verdict = compiled
            .run_wasm(&inputs(index as u128))
            .expect("the WASM lane builds");
        assert!(
            verdict.rejects(),
            "{limbed:?}({lhs:?}, {rhs:?}) at {bits} bits is rejected by the model, but the WASM \
             lane answered {verdict:?}"
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
/// The multiplier is `2^(N - 2)` or-ed onto a parameter, so a multiplicand of three fits and four
/// overflows by exactly its top column; untaken, the schoolbook's checks are off with the guard.
/// 253 is the schoolbook on one element's decomposition, 320 on limbs.
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

/// A division under a witnessed condition is checked only where the condition holds.
///
/// Where the branch is not taken the divisor is whatever that branch left behind, here zero, which
/// no quotient or remainder satisfies. The checks go off with the guard, the pure side divides by
/// one instead, and the answer is zero.
///
/// 253 is the division on a decomposition of each operand, and 320 on limbs.
#[test]
fn a_guarded_division_is_checked_only_where_its_guard_holds() {
    for kind in [BinaryArithOpKind::UDiv, BinaryArithOpKind::URem] {
        for bits in [253usize, 320] {
            let high = BigUint::from(1u8) << (bits - 3);
            let top = IntBits::from_biguint(bits, &high);
            let ssa = main_program(
                &[Type::int(1), Type::int(64), Type::int(64)],
                &[Type::int(64)],
                move |e, params| {
                    let lhs = e.cast_to(CastTarget::Int(bits), params[1]);
                    let top = e.int_const(top.clone());
                    let lhs = e.bin(BinaryArithOpKind::Or, lhs, top);
                    let rhs = e.cast_to(CastTarget::Int(bits), params[2]);

                    let (then_id, _) = e.add_block();
                    let (else_id, _) = e.add_block();
                    let (merge_id, _) = e.add_block();
                    e.seal_and_switch(Terminator::JmpIf(params[0], then_id, else_id), then_id);
                    let answer = e.bin(kind, lhs, rhs);
                    let low = e.cast_to(CastTarget::Int(64), answer);
                    e.seal_and_switch(Terminator::Jmp(merge_id, vec![low]), else_id);
                    let zero = e.int_const(IntBits::zero(64));
                    e.seal_and_switch(Terminator::Jmp(merge_id, vec![zero]), merge_id);
                    vec![e.add_parameter(Type::int(64))]
                },
            );
            let what = format!("a guarded int{bits} {kind:?}");
            let compiled = Compiled::new(ssa)
                .unwrap_or_else(|error| panic!("{what} does not compile: {error}"));

            let run = |taken: u128, lhs: u128, rhs: u128, answer: u128| {
                compiled.run(&input_block(&[
                    &IntBits::from_u128(1, taken),
                    &IntBits::from_u128(64, lhs),
                    &IntBits::from_u128(64, rhs),
                    &IntBits::from_u128(64, answer),
                ]))
            };

            let dividend = &high | BigUint::from(12345u32);
            let exact: BigUint = match kind {
                BinaryArithOpKind::UDiv => &dividend / 7u8,
                _ => &dividend % 7u8,
            };
            let low = u128::from(
                u64::try_from(exact % (BigUint::from(1u8) << 64)).expect("a low limb fits"),
            );
            let verdict = run(1, 12345, 7, low);
            assert!(
                verdict.is_accepted(),
                "{what}, taken, by seven: {verdict:?}"
            );
            let verdict = run(1, 12345, 0, 0);
            assert!(verdict.is_refusal(), "{what}, taken, by zero: {verdict:?}");
            let verdict = run(0, 12345, 0, 0);
            assert!(
                verdict.is_accepted(),
                "{what}, not taken, by zero: {verdict:?}"
            );
        }
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

/// The whole matrix: every operation under both readings, at every width the model sweeps, over
/// every corner pair.
///
/// Ignored by default for its size: `cargo test -p mavros-compiler --test bigints -- --ignored`.
#[test]
#[ignore = "every corner pair of every operation, compiled and run"]
fn the_whole_corner_matrix_agrees_with_the_model() {
    for kind in BinaryArithOpKind::ALL {
        for &bits in corners::widths() {
            sweep(kind, bits);
        }
    }
}

// THE WITNESSED SIGNED FAMILY
// ================================================================================================

/// The signed corners a carry chain turns on at `bits`: [`signed_corners`], and either sign of a
/// value at the lowest limb boundary and at the top limb's, so that a carry or a borrow crosses a
/// limb and leaves the top under every combination of signs.
fn signed_limb_corners(bits: usize) -> Vec<IntBits> {
    let limb = witness_limb_bits(FieldConfig::bn254());
    let top_limb = (bits - 1) / limb * limb;
    let signed = |v: SignedValue| IntBits::from_signed(bits, &v);
    let power = |place: usize| SignedValue::from(BigUint::from(1u8) << place);
    let mut values = signed_corners(bits);
    values.extend([
        IntBits::all_ones(limb).cast(bits),
        signed(power(limb)),
        signed(-power(limb)),
        signed(power(top_limb)),
        signed(-power(top_limb)),
    ]);
    dedup(values)
}

/// The signed sum, difference and ordering agree with the model either side of the carry chain.
///
/// 200 is the single cell past the host word, and 252 its last width, where the sign packing comes
/// closest to the modulus; 253 is the chain on a decomposition of each operand; 257 has a one-bit
/// top limb, which is also its sign bit; 320 is limbs with a full top limb.
#[test]
fn a_signed_sum_difference_or_ordering_agrees_with_the_model_either_side_of_the_chain() {
    for bits in [200usize, 252, 253, 257, 320] {
        let values = signed_limb_corners(bits);
        for limbed in [
            Limbed::SignedSum,
            Limbed::SignedDifference,
            Limbed::SignedOrdering,
        ] {
            check_limbed_corners(limbed, bits, &values, Rhs::Witnessed);
        }
    }
}

/// A known operand of the signed chain is read at compile time: its top limb's sign and the flip an
/// ordering applies to it are constants, whichever side it is on.
///
/// `INT_MIN` is among the constants: against it the idle left operand is `-1`, as `0 - INT_MIN`
/// overflows where `-1 - INT_MIN` is `INT_MAX`.
#[test]
fn a_signed_chain_reads_a_known_operand_at_compile_time() {
    let bits = 320;
    let values = signed_corners(bits);
    for limbed in [
        Limbed::SignedSum,
        Limbed::SignedDifference,
        Limbed::SignedOrdering,
    ] {
        check_limbed_corners(limbed, bits, &values, Rhs::Constant);
        check_limbed_corners(limbed, bits, &values, Rhs::AgainstPure);
    }
}

/// The signed chain against a **pure** operand that a loop computes, which no fold answers.
///
/// Such an operand is known wherever the constraints are built but not at compile time, so its
/// sign and the flip an ordering applies to it are pure arithmetic rather than cuts. Each iteration
/// takes a negative value `INT_MIN + i·2^(N - 10)` against a positive witnessed one, adding and
/// subtracting them and ordering them both ways, and the program asserts what the model says of
/// every iteration. Clearing the bit that selects the witnessed value changes every sum and
/// difference, which the assertion refuses. 253 is the chain on a decomposition, and 320 limbs.
#[test]
fn a_signed_chain_against_a_pure_operand_a_loop_computes_agrees_with_the_model() {
    let iterations = 3u128;
    for bits in [253usize, 320] {
        let witnessed = IntBits::from_biguint(bits, &((BigUint::from(1u8) << (bits - 60)) + 5u8));
        let stride = IntBits::from_biguint(bits, &(BigUint::from(1u8) << (bits - 10)));
        let sign = IntBits::from_signed(bits, &IntBits::signed_min(bits));
        let pure_at = |index: u128| {
            let index = IntBits::from_u128(bits, index);
            let scaled = mavros_int_semantics::eval(IntOp::UMul, &index, &stride)
                .value()
                .expect("the stride times the index fits");
            scaled.xor(&sign)
        };
        let model = |op, lhs: &IntBits, rhs: &IntBits| {
            mavros_int_semantics::eval(op, lhs, rhs)
                .value()
                .expect("every iteration fits")
        };
        let expected = (1..=iterations).fold(IntBits::zero(bits), |acc, index| {
            let pure = pure_at(index);
            acc.xor(&model(IntOp::SAdd, &witnessed, &pure)).xor(&model(
                IntOp::SSub,
                &pure,
                &witnessed,
            ))
        });

        let (program_witnessed, program_stride) = (witnessed.clone(), stride.clone());
        let ssa = main_program(&[Type::int(1)], &[], move |e, params| {
            let chosen = e.int_const(program_witnessed);
            let zero = e.int_const(IntBits::zero(bits));
            let value = e.select(params[0], chosen, zero);
            let stride = e.int_const(program_stride);
            let sign = e.int_const(sign);
            let start = e.int_const(IntBits::from_u128(32, 1));
            let yes = e.int_const(IntBits::one(1));
            let no = e.int_const(IntBits::zero(1));
            let (head, _) = e.add_block();
            let (body, _) = e.add_block();
            let (done, _) = e.add_block();
            e.seal_and_switch(Terminator::Jmp(head, vec![start, zero, yes, no]), head);
            let count = e.add_parameter(Type::int(32));
            let acc = e.add_parameter(Type::int(bits));
            let every_less = e.add_parameter(Type::int(1));
            let any_greater = e.add_parameter(Type::int(1));
            let limit = e.int_const(IntBits::from_u128(32, iterations + 1));
            let more = e.cmp(count, limit, CmpKind::ULt);
            e.seal_and_switch(Terminator::JmpIf(more, body, done), body);
            let index = e.cast_to(CastTarget::Int(bits), count);
            let scaled = e.bin(BinaryArithOpKind::UMul, index, stride);
            let pure = e.bin(BinaryArithOpKind::Xor, scaled, sign);
            let sum = e.bin(BinaryArithOpKind::SAdd, value, pure);
            let difference = e.bin(BinaryArithOpKind::SSub, pure, value);
            let next_acc = e.bin(BinaryArithOpKind::Xor, acc, sum);
            let next_acc = e.bin(BinaryArithOpKind::Xor, next_acc, difference);
            let less = e.cmp(pure, value, CmpKind::SLt);
            let next_less = e.bin(BinaryArithOpKind::And, every_less, less);
            let greater = e.cmp(value, pure, CmpKind::SLt);
            let next_greater = e.bin(BinaryArithOpKind::Or, any_greater, greater);
            let one = e.int_const(IntBits::from_u128(32, 1));
            let next = e.bin(BinaryArithOpKind::UAdd, count, one);
            e.seal_and_switch(
                Terminator::Jmp(head, vec![next, next_acc, next_less, next_greater]),
                done,
            );
            let expected = e.int_const(expected);
            e.assert_eq(acc, expected);
            e.assert_eq(every_less, yes);
            e.assert_eq(any_greater, no);
            vec![]
        });
        let what = format!("int{bits} signed chain against a pure operand");
        let compiled =
            Compiled::new(ssa).unwrap_or_else(|error| panic!("{what} does not compile: {error}"));
        let selected = |bit: u128| input_block(&[&IntBits::from_u128(1, bit)]);
        let verdict = compiled.run(&selected(1));
        assert!(verdict.is_accepted(), "{what}: {verdict:?}");
        let verdict = compiled
            .run_wasm(&selected(1))
            .expect("the WASM lane builds");
        assert!(verdict.is_accepted(), "{what}, WASM: {verdict:?}");
        let verdict = compiled.run(&selected(0));
        assert!(
            !verdict.is_accepted(),
            "{what}, the witnessed value zero: {verdict:?}"
        );
    }
}

/// A signed left shift is the unsigned one: the same map on the bit pattern, with the amount read
/// unsigned. 100 is the single cell, and 320 is limbs.
#[test]
fn a_signed_shift_left_agrees_with_the_model_past_the_host_word() {
    for bits in [100usize, 320] {
        let amounts = few_shift_amounts(bits, HOST_LIMB_BITS);
        check_limbed_shifts(
            Limbed::SignedShiftLeft,
            bits,
            &signed_corners(bits),
            &amounts,
            Rhs::Witnessed,
        );
    }
}

/// A signed product, quotient, remainder and right shift agree with the model past the host word.
///
/// 100 is the single cell. 128 takes the sign-magnitude gadgets over a decomposition of each
/// operand, as the signed single cell has no two-limb form. 320 is limbs, against a constant right
/// operand (for a shift, a known amount) as well as a witnessed one. A constant zero divisor is
/// refused whatever it divides, so it cannot run beside the others.
#[test]
fn a_signed_product_quotient_or_right_shift_agrees_with_the_model_past_the_host_word() {
    for bits in [100usize, 128, 320] {
        let values = signed_corners(bits);
        for limbed in [
            Limbed::SignedProduct,
            Limbed::SignedQuotient,
            Limbed::SignedRemainder,
        ] {
            check_limbed_corners(limbed, bits, &values, Rhs::Witnessed);
            if bits == 320 {
                let pairs: Vec<(IntBits, IntBits)> = values
                    .iter()
                    .flat_map(|a| {
                        values
                            .iter()
                            .filter(|b| matches!(limbed, Limbed::SignedProduct) || !b.is_zero())
                            .map(move |b| (a.clone(), b.clone()))
                    })
                    .collect();
                for pairs in pairs.chunks(8) {
                    check_limbed_pairs(limbed, bits, pairs.to_vec(), Rhs::Constant);
                }
            }
        }
        let mut amounts = few_shift_amounts(bits, HOST_LIMB_BITS);
        check_limbed_shifts(
            Limbed::SignedShiftRight,
            bits,
            &values,
            &amounts,
            Rhs::Witnessed,
        );
        // A known amount fills the vacated bits with the sign rather than complementing around the
        // unsigned shift. One at the width is refused at compile time, and so is left out.
        if bits == 320 {
            amounts.pop();
            check_limbed_shifts(
                Limbed::SignedShiftRight,
                bits,
                &values,
                &amounts,
                Rhs::Constant,
            );
        }
    }
}

/// A signed product's magnitude is held to `2^(N - 1)` exactly where its sign is negative, and
/// below it otherwise, as is a quotient's, whose one way past is `INT_MIN / -1`.
///
/// `2^h · 2^(N - 1 - h)` is `2^(N - 1)`, which fits only negated, and one more than the second factor
/// passes it under either sign. 126 is the single cell's last width, 127 the gadgets' first, 128 the
/// width the unsigned single cell takes and the signed one does not, and 254 limbs.
#[test]
fn a_signed_product_or_quotient_is_held_to_its_sign_at_int_min() {
    for bits in [126usize, 127, 128, 254] {
        let signed = |v: SignedValue| IntBits::from_signed(bits, &v);
        let power = |place: usize| SignedValue::from(BigUint::from(1u8) << place);
        let (low, high) = (bits / 2, bits - 1 - bits / 2);
        let min = signed(IntBits::signed_min(bits));
        let one = signed(SignedValue::from(1));
        let minus_one = signed(SignedValue::from(-1));
        let products = vec![
            (signed(power(low)), signed(-power(high))),
            (signed(-power(low)), signed(power(high))),
            (signed(power(low)), signed(power(high))),
            (signed(-power(low)), signed(-power(high))),
            (signed(-power(low)), signed(power(high) + 1)),
            (min.clone(), one.clone()),
            (min.clone(), minus_one.clone()),
            (signed(IntBits::signed_min(bits) + 1), minus_one.clone()),
        ];
        check_limbed_pairs(Limbed::SignedProduct, bits, products, Rhs::Witnessed);

        let divisions = vec![
            (min.clone(), minus_one.clone()),
            (min.clone(), one),
            (min.clone(), min.clone()),
            (signed(IntBits::signed_min(bits) + 1), minus_one),
        ];
        for limbed in [Limbed::SignedQuotient, Limbed::SignedRemainder] {
            check_limbed_pairs(limbed, bits, divisions.clone(), Rhs::Witnessed);
        }
    }
}

/// A signed product, quotient and remainder whose signs are known at compile time.
///
/// The witnessed operand is a widened 64-bit parameter, whose limbs above it are known zeros, so
/// its sign is known to be clear; the other is a constant. The answer's sign is then known too, and
/// a negative one complements the answer's limbs linearly rather than by a product with a cut bit.
/// The program asserts the model's answers for one input, and so refuses another.
#[test]
fn a_signed_operation_on_operands_of_known_sign_agrees_with_the_model() {
    let bits = 320usize;
    let signed = |v: SignedValue| IntBits::from_signed(bits, &v);
    let power = |place: usize| SignedValue::from(BigUint::from(1u8) << place);
    let input = (1u128 << 40) + 9;
    let x = IntBits::from_u128(bits, input);
    let constants = [
        signed(SignedValue::from(-3)),
        signed(-(power(200) + SignedValue::from(7))),
        signed(SignedValue::from(5)),
    ];
    let mut checks = Vec::new();
    for c in &constants {
        for (kind, lhs, rhs) in [
            (BinaryArithOpKind::SMul, &x, c),
            (BinaryArithOpKind::SDiv, &x, c),
            (BinaryArithOpKind::SRem, &x, c),
            (BinaryArithOpKind::SDiv, c, &x),
            (BinaryArithOpKind::SRem, c, &x),
        ] {
            let want = mavros_int_semantics::eval(IntOp::from(kind), lhs, rhs)
                .value()
                .expect("the model accepts every pair");
            checks.push((kind, lhs == &x, c.clone(), want));
        }
    }

    let ssa = main_program(&[Type::int(64)], &[], move |e, params| {
        let x = e.cast_to(CastTarget::Int(bits), params[0]);
        for (kind, x_first, constant, want) in &checks {
            let constant = e.int_const(constant.clone());
            let (lhs, rhs) = if *x_first {
                (x, constant)
            } else {
                (constant, x)
            };
            let got = e.bin(*kind, lhs, rhs);
            let want = e.int_const(want.clone());
            e.assert_eq(got, want);
        }
        vec![]
    });
    let what = format!("int{bits} signed operations of known sign");
    let compiled =
        Compiled::new(ssa).unwrap_or_else(|error| panic!("{what} does not compile: {error}"));
    let run = |value: u128| input_block(&[&IntBits::from_u128(64, value)]);
    let verdict = compiled.run(&run(input));
    assert!(verdict.is_accepted(), "{what}: {verdict:?}");
    let verdict = compiled
        .run_wasm(&run(input))
        .expect("the WASM lane builds");
    assert!(verdict.is_accepted(), "{what}, WASM: {verdict:?}");
    let verdict = compiled.run(&run(input + 1));
    assert!(!verdict.is_accepted(), "{what}, another input: {verdict:?}");
}

/// A signed answer the gadgets know at compile time is still a witnessed result.
///
/// Every limb of `0 % x`, `0 / x` and `x * 0` is a known zero, which is delivered as a witnessed
/// constant. Returned as it is, a result that reached the program as a pure value would disagree
/// with the witnessed return the function declares. A zero `x` is still a zero divisor.
#[test]
fn a_signed_answer_known_at_compile_time_stays_witnessed() {
    let bits = 320usize;
    let ssa = main_program(
        &[Type::int(64)],
        &[Type::int(64), Type::int(64), Type::int(64)],
        move |e, params| {
            let x = e.cast_to(CastTarget::Int(bits), params[0]);
            let zero = e.int_const(IntBits::zero(bits));
            [
                (BinaryArithOpKind::SRem, zero, x),
                (BinaryArithOpKind::SDiv, zero, x),
                (BinaryArithOpKind::SMul, x, zero),
            ]
            .into_iter()
            .map(|(kind, lhs, rhs)| {
                let answer = e.bin(kind, lhs, rhs);
                e.cast_to(CastTarget::Int(64), answer)
            })
            .collect()
        },
    );
    let what = format!("int{bits} signed answers known to be zero");
    let compiled =
        Compiled::new(ssa).unwrap_or_else(|error| panic!("{what} does not compile: {error}"));
    let zero = IntBits::zero(64);
    let run = |value: u128| input_block(&[&IntBits::from_u128(64, value), &zero, &zero, &zero]);
    let verdict = compiled.run(&run(5));
    assert!(verdict.is_accepted(), "{what}: {verdict:?}");
    let verdict = compiled.run_wasm(&run(5)).expect("the WASM lane builds");
    assert!(verdict.is_accepted(), "{what}, WASM: {verdict:?}");
    let verdict = compiled.run(&run(0));
    assert!(verdict.is_refusal(), "{what}, a zero divisor: {verdict:?}");
}

/// A negative value the gadgets know at compile time, shifted right by a known amount, is still a
/// witnessed result.
///
/// The operand is a constant put into the witness domain, so every limb is known and its sign is
/// known to be set. An answer limb that is one piece of it, or that the shift empties, is then a
/// constant before the sign fills it, and has to be one afterwards too: arithmetic on two
/// constants would reach the result as a pure value where the function returns a witnessed one.
#[test]
fn a_known_negative_value_shifted_right_by_a_known_amount_stays_witnessed() {
    let bits = 320usize;
    let power = SignedValue::from(BigUint::from(1u8) << 310);
    let value = IntBits::from_signed(bits, &(-(power + SignedValue::from(5))));
    for amount in [70u128, 250, 300, 319] {
        let shift = IntBits::from_u128(bits, amount);
        let want = mavros_int_semantics::eval(IntOp::SShr, &value, &shift)
            .value()
            .expect("the model accepts the shift")
            .cast(64);
        let (program_value, program_shift) = (value.clone(), shift.clone());
        let ssa = main_program(&[Type::int(1)], &[Type::int(64)], move |e, params| {
            let constant = e.int_const(program_value);
            let witnessed = e.cast_to(CastTarget::WitnessOf, constant);
            let amount = e.int_const(program_shift);
            let answer = e.bin(BinaryArithOpKind::SShr, witnessed, amount);
            let one = e.int_const(IntBits::one(1));
            e.assert_eq(params[0], one);
            vec![e.cast_to(CastTarget::Int(64), answer)]
        });
        let what = format!("int{bits} SShr of a known negative value by {amount}");
        let compiled =
            Compiled::new(ssa).unwrap_or_else(|error| panic!("{what} does not compile: {error}"));
        let verdict = compiled.run(&input_block(&[&IntBits::one(1), &want]));
        assert!(verdict.is_accepted(), "{what}: {verdict:?}");
    }
}

/// The mutation test for the sign-magnitude gadgets: the product, the quotient and remainder, and
/// the right shift by a witnessed amount and by a known one, each on limbs.
///
/// Each operand is a constant selected by a witnessed bit, so that it is a witnessed value held as
/// limbs: a negative left one, and a right one of either sign. Between them every magnitude is
/// negated, the answers take either sign, and each sign bit is cut. Every answer leaves through its
/// low limb.
///
/// A division also carries the guard IR's checks against a zero divisor and `INT_MIN / -1`, which
/// compare limb by limb, and an equality's inverse is free where the two limbs agree. So no limb of
/// a divisor here is zero or all ones, and no limb of its dividend is `INT_MIN`'s.
#[test]
fn every_witness_column_of_a_signed_product_quotient_or_right_shift_is_pinned() {
    let bits = 320usize;
    let signed = |v: SignedValue| IntBits::from_signed(bits, &v);
    let power = |place: usize| SignedValue::from(BigUint::from(1u8) << place);
    let small = power(70) + SignedValue::from(3);
    let lhs = signed(-(power(200) + SignedValue::from(12345)));
    let dividend = signed(-(power(300) + power(200) + SignedValue::from(12345)));
    let divisor = power(260) + power(200) + power(130) + power(70) + SignedValue::from(3);
    let low = |value: &IntBits| value.cast(64);
    let selected = IntBits::from_u128(1, 1);
    let amount = IntBits::from_u128(64, 67);

    for (kind, lhs, rhs) in [
        (BinaryArithOpKind::SMul, lhs.clone(), signed(-small.clone())),
        (BinaryArithOpKind::SMul, lhs.clone(), signed(small)),
        (
            BinaryArithOpKind::SDiv,
            dividend.clone(),
            signed(-divisor.clone()),
        ),
        (BinaryArithOpKind::SRem, dividend, signed(divisor)),
    ] {
        let want = mavros_int_semantics::eval(IntOp::from(kind), &lhs, &rhs)
            .value()
            .expect("the model accepts the pair");
        let (program_lhs, program_rhs) = (lhs, rhs);
        let ssa = main_program(
            &[Type::int(1), Type::int(1)],
            &[Type::int(64)],
            move |e, params| {
                let zero = e.int_const(IntBits::zero(bits));
                let one = e.int_const(IntBits::one(bits));
                let left = e.int_const(program_lhs);
                let left = e.select(params[0], left, zero);
                let right = e.int_const(program_rhs);
                let right = e.select(params[1], right, one);
                let answer = e.bin(kind, left, right);
                vec![e.cast_to(CastTarget::Int(64), answer)]
            },
        );
        let what = format!("{kind:?} at {bits} bits");
        let compiled =
            Compiled::new(ssa).unwrap_or_else(|error| panic!("{what} does not compile: {error}"));
        assert_every_column_is_pinned(&what, &compiled, &[&selected, &selected, &low(&want)]);
    }

    let shifted = mavros_int_semantics::eval(IntOp::SShr, &lhs, &IntBits::from_u128(bits, 67))
        .value()
        .expect("the model accepts the shift");
    for witnessed_amount in [true, false] {
        let program_lhs = lhs.clone();
        let ssa = main_program(
            &[Type::int(1), Type::int(64)],
            &[Type::int(64)],
            move |e, params| {
                let zero = e.int_const(IntBits::zero(bits));
                let left = e.int_const(program_lhs);
                let left = e.select(params[0], left, zero);
                let amount = if witnessed_amount {
                    e.cast_to(CastTarget::Int(bits), params[1])
                } else {
                    e.int_const(IntBits::from_u128(bits, 67))
                };
                let answer = e.bin(BinaryArithOpKind::SShr, left, amount);
                vec![e.cast_to(CastTarget::Int(64), answer)]
            },
        );
        let what = format!("SShr at {bits} bits, witnessed amount: {witnessed_amount}");
        let compiled =
            Compiled::new(ssa).unwrap_or_else(|error| panic!("{what} does not compile: {error}"));
        assert_every_column_is_pinned(&what, &compiled, &[&selected, &amount, &low(&shifted)]);
    }
}

/// A signed product, quotient, remainder or right shift under a witnessed condition is checked only
/// where the condition holds.
///
/// Where the branch is not taken the operands are whatever it left behind, and each run here leaves
/// something the taken branch refuses: a product whose magnitudes overflow even unsigned, which
/// leaves the schoolbook's top column past its width, a zero divisor, `INT_MIN / -1`, and an amount
/// at the width. 253 is the gadgets on a decomposition of each operand, and 320 on limbs.
#[test]
fn a_guarded_signed_operation_is_checked_only_where_its_guard_holds() {
    for bits in [253usize, 320] {
        let signed = |v: i128| IntBits::from_signed(bits, &SignedValue::from(v));
        let min = IntBits::from_signed(bits, &IntBits::signed_min(bits));
        let high = BigUint::from(1u8) << (bits - 3);
        let low = |value: &IntBits| value.cast(64);
        for (kind, holds, fails) in [
            (
                BinaryArithOpKind::SMul,
                (
                    signed(-3),
                    IntBits::from_biguint(bits, &(high.clone() + 5u8)),
                ),
                (
                    signed(-16),
                    IntBits::from_biguint(bits, &(high.clone() + 5u8)),
                ),
            ),
            (
                BinaryArithOpKind::SDiv,
                (signed(-7), signed(2)),
                (signed(-7), signed(0)),
            ),
            (
                BinaryArithOpKind::SRem,
                (signed(-7), signed(2)),
                (min.clone(), signed(-1)),
            ),
            (
                BinaryArithOpKind::SDiv,
                (min.clone(), signed(1)),
                (min.clone(), signed(-1)),
            ),
            (
                BinaryArithOpKind::SShr,
                (signed(-7), signed(1)),
                (signed(-7), IntBits::from_u128(bits, bits as u128)),
            ),
        ] {
            let want = mavros_int_semantics::eval(IntOp::from(kind), &holds.0, &holds.1)
                .value()
                .expect("the model accepts the holding pair");
            let ssa = main_program(
                &[Type::int(1), Type::int(1)],
                &[Type::int(64)],
                move |e, params| {
                    let pair = |e: &mut HLBlockEmitter<'_>, (lhs, rhs): &(IntBits, IntBits)| {
                        (e.int_const(lhs.clone()), e.int_const(rhs.clone()))
                    };
                    let (holds_lhs, holds_rhs) = pair(e, &holds);
                    let (fails_lhs, fails_rhs) = pair(e, &fails);
                    let lhs = e.select(params[1], holds_lhs, fails_lhs);
                    let rhs = e.select(params[1], holds_rhs, fails_rhs);

                    let (then_id, _) = e.add_block();
                    let (else_id, _) = e.add_block();
                    let (merge_id, _) = e.add_block();
                    e.seal_and_switch(Terminator::JmpIf(params[0], then_id, else_id), then_id);
                    let answer = e.bin(kind, lhs, rhs);
                    let answer = e.cast_to(CastTarget::Int(64), answer);
                    e.seal_and_switch(Terminator::Jmp(merge_id, vec![answer]), else_id);
                    let zero = e.int_const(IntBits::zero(64));
                    e.seal_and_switch(Terminator::Jmp(merge_id, vec![zero]), merge_id);
                    vec![e.add_parameter(Type::int(64))]
                },
            );
            let what = format!("a guarded int{bits} {kind:?}");
            let compiled = Compiled::new(ssa)
                .unwrap_or_else(|error| panic!("{what} does not compile: {error}"));
            let zero = IntBits::zero(64);
            for wasm in [false, true] {
                // A refusal is a trap on the VM, and a witness the constraints refuse on WASM.
                let refused = |verdict: &Verdict| {
                    if wasm {
                        verdict.rejects()
                    } else {
                        verdict.is_refusal()
                    }
                };
                let run = |taken: u128, holding: u128, answer: &IntBits| {
                    let inputs = input_block(&[
                        &IntBits::from_u128(1, taken),
                        &IntBits::from_u128(1, holding),
                        answer,
                    ]);
                    if wasm {
                        compiled.run_wasm(&inputs).expect("the WASM lane builds")
                    } else {
                        compiled.run(&inputs)
                    }
                };
                let lane = if wasm { "WASM" } else { "VM" };
                let verdict = run(1, 1, &low(&want));
                assert!(
                    verdict.is_accepted(),
                    "{what}, {lane}, taken, holding: {verdict:?}"
                );
                let verdict = run(1, 0, &zero);
                assert!(
                    refused(&verdict),
                    "{what}, {lane}, taken, failing: {verdict:?}"
                );
                let verdict = run(0, 0, &zero);
                assert!(
                    verdict.is_accepted(),
                    "{what}, {lane}, not taken, failing: {verdict:?}"
                );
            }
        }
    }
}

/// A signed product, quotient, remainder and right shift against a **pure** operand a loop
/// computes, which no fold answers.
///
/// Such an operand is known wherever the constraints are built but not at compile time, so its
/// magnitude and its sign are pure arithmetic rather than a negation through the chain and a cut.
/// Each iteration takes the negative `-(i·2^(N/2) + 3)` against a positive witnessed value, as
/// each side of a product and of a division, and shifts it by a witnessed amount; the program
/// asserts what the model says of every iteration. Clearing the bit that selects the witnessed
/// value changes every answer, which the assertion refuses. 253 is a decomposition, and 320 limbs.
#[test]
fn a_signed_product_or_quotient_against_a_pure_operand_a_loop_computes_agrees_with_the_model() {
    let iterations = 3u128;
    for bits in [253usize, 320] {
        let witnessed =
            IntBits::from_biguint(bits, &((BigUint::from(1u8) << (bits / 2 - 5)) + 5u8));
        let stride = IntBits::from_biguint(bits, &(BigUint::from(1u8) << (bits / 2)));
        let amount = IntBits::from_u128(bits, 67);
        let pure_at = |index: u128| {
            let magnitude: SignedValue =
                SignedValue::from(index) * IntBits::to_signed(&stride) + SignedValue::from(3);
            IntBits::from_signed(bits, &-magnitude)
        };
        let model = |op, lhs: &IntBits, rhs: &IntBits| {
            mavros_int_semantics::eval(op, lhs, rhs)
                .value()
                .expect("every iteration fits")
        };
        let expected = (1..=iterations).fold(IntBits::zero(bits), |acc, index| {
            let pure = pure_at(index);
            [
                model(IntOp::SMul, &witnessed, &pure),
                model(IntOp::SDiv, &pure, &witnessed),
                model(IntOp::SRem, &pure, &witnessed),
                model(IntOp::SDiv, &witnessed, &pure),
                model(IntOp::SShr, &pure, &amount),
            ]
            .iter()
            .fold(acc, |acc, answer| acc.xor(answer))
        });

        let (program_witnessed, program_stride) = (witnessed.clone(), stride.clone());
        let ssa = main_program(&[Type::int(1), Type::int(64)], &[], move |e, params| {
            let chosen = e.int_const(program_witnessed);
            let zero = e.int_const(IntBits::zero(bits));
            let value = e.select(params[0], chosen, zero);
            let amount = e.cast_to(CastTarget::Int(bits), params[1]);
            let stride = e.int_const(program_stride);
            let three = e.int_const(IntBits::from_u128(bits, 3));
            let start = e.int_const(IntBits::from_u128(32, 1));
            let (head, _) = e.add_block();
            let (body, _) = e.add_block();
            let (done, _) = e.add_block();
            e.seal_and_switch(Terminator::Jmp(head, vec![start, zero]), head);
            let count = e.add_parameter(Type::int(32));
            let acc = e.add_parameter(Type::int(bits));
            let limit = e.int_const(IntBits::from_u128(32, iterations + 1));
            let more = e.cmp(count, limit, CmpKind::ULt);
            e.seal_and_switch(Terminator::JmpIf(more, body, done), body);
            let index = e.cast_to(CastTarget::Int(bits), count);
            let scaled = e.bin(BinaryArithOpKind::UMul, index, stride);
            let magnitude = e.bin(BinaryArithOpKind::UAdd, scaled, three);
            let pure = e.bin(BinaryArithOpKind::SSub, zero, magnitude);
            let mut next_acc = acc;
            for (kind, lhs, rhs) in [
                (BinaryArithOpKind::SMul, value, pure),
                (BinaryArithOpKind::SDiv, pure, value),
                (BinaryArithOpKind::SRem, pure, value),
                (BinaryArithOpKind::SDiv, value, pure),
                (BinaryArithOpKind::SShr, pure, amount),
            ] {
                let answer = e.bin(kind, lhs, rhs);
                next_acc = e.bin(BinaryArithOpKind::Xor, next_acc, answer);
            }
            let one = e.int_const(IntBits::from_u128(32, 1));
            let next = e.bin(BinaryArithOpKind::UAdd, count, one);
            e.seal_and_switch(Terminator::Jmp(head, vec![next, next_acc]), done);
            let expected = e.int_const(expected);
            e.assert_eq(acc, expected);
            vec![]
        });
        let what = format!("int{bits} signed product and quotient against a pure operand");
        let compiled =
            Compiled::new(ssa).unwrap_or_else(|error| panic!("{what} does not compile: {error}"));
        let amount = IntBits::from_u128(64, 67);
        let selected = |bit: u128| input_block(&[&IntBits::from_u128(1, bit), &amount]);
        let verdict = compiled.run(&selected(1));
        assert!(verdict.is_accepted(), "{what}: {verdict:?}");
        let verdict = compiled
            .run_wasm(&selected(1))
            .expect("the WASM lane builds");
        assert!(verdict.is_accepted(), "{what}, WASM: {verdict:?}");
        let verdict = compiled.run(&selected(0));
        assert!(
            !verdict.is_accepted(),
            "{what}, the witnessed value zero: {verdict:?}"
        );
    }
}

/// The mutation test for the signed chain, whose columns past the unsigned one's are the carry out
/// of the top limb and the sign bits of both operands and of the answer.
///
/// The left operand is negative and the right one positive, so the two sign bits the check reads
/// differ, and the sum is negative while the difference is not. Both are built from a sign
/// extension of the parameter, so neither top limb is known at compile time and both operands' sign
/// bits are columns. 253 is the chain on a decomposition and 320 on limbs.
#[test]
fn every_witness_column_of_a_signed_sum_difference_or_ordering_is_pinned() {
    for bits in [253usize, 320] {
        let negative = IntBits::from_signed(bits, &-(SignedValue::from(1u8) << (bits - 2)));
        let parameter = IntBits::from_u128(64, 3);
        let operands = move |e: &mut HLBlockEmitter<'_>, param: ValueId| {
            let positive = e.fresh_value();
            e.emit(OpCode::SExt {
                result: positive,
                value: param,
                from_bits: 64,
                to_bits: bits,
            });
            let high = e.int_const(negative.clone());
            (e.bin(BinaryArithOpKind::Or, positive, high), positive)
        };

        let sum = main_program(&[Type::int(64)], &[Type::int(64)], |e, params| {
            let (lhs, rhs) = operands(e, params[0]);
            let sum = e.bin(BinaryArithOpKind::SAdd, lhs, rhs);
            vec![e.cast_to(CastTarget::Int(64), sum)]
        });
        let difference = main_program(&[Type::int(64)], &[Type::int(64)], |e, params| {
            let (lhs, rhs) = operands(e, params[0]);
            let difference = e.bin(BinaryArithOpKind::SSub, rhs, lhs);
            vec![e.cast_to(CastTarget::Int(64), difference)]
        });
        let ordering = main_program(&[Type::int(64)], &[Type::int(1)], |e, params| {
            let (lhs, rhs) = operands(e, params[0]);
            vec![e.cmp(lhs, rhs, CmpKind::SLt)]
        });

        // `(-2^(N-2) | 3) + 3` and `3 - (-2^(N-2) | 3)`, read at their low limb.
        for (operation, ssa, answer) in [
            ("SAdd", sum, IntBits::from_u128(64, 6)),
            ("SSub", difference, IntBits::from_u128(64, 0)),
            ("SLt", ordering, IntBits::from_u128(1, 1)),
        ] {
            let what = format!("{operation} at {bits} bits");
            let compiled = Compiled::new(ssa)
                .unwrap_or_else(|error| panic!("{what} does not compile: {error}"));
            assert_every_column_is_pinned(&what, &compiled, &[&parameter, &answer]);
        }
    }
}

/// A signed sum or difference under a witnessed condition is checked only where the condition
/// holds.
///
/// Its overflow is the one check relating the top carry to the sign bits, which goes off with the
/// guard; the answer is zero there. The constant sits five inside a boundary, so three fits and
/// seven overflows, out of the top for a sum and below the bottom for a difference.
#[test]
fn a_guarded_signed_sum_or_difference_is_checked_only_where_its_guard_holds() {
    for bits in [253usize, 320] {
        let max = IntBits::signed_max(bits);
        let min = IntBits::signed_min(bits);
        for (kind, base, low) in [
            (
                BinaryArithOpKind::SAdd,
                max.clone() - 5,
                IntBits::from_signed(bits, &(max - 2)).cast(64),
            ),
            (
                BinaryArithOpKind::SSub,
                min.clone() + 5,
                IntBits::from_signed(bits, &(min + 2)).cast(64),
            ),
        ] {
            let base = IntBits::from_signed(bits, &base);
            let ssa = main_program(
                &[Type::int(1), Type::int(64), Type::int(1)],
                &[Type::int(64)],
                move |e, params| {
                    // Both operands' top limbs are unknown at compile time, so both sign bits are
                    // cut: the left one a selection, and the right one a sign extension.
                    let lhs = e.int_const(base.clone());
                    let zero = e.int_const(IntBits::zero(bits));
                    let lhs = e.select(params[2], lhs, zero);
                    let rhs = e.fresh_value();
                    e.emit(OpCode::SExt {
                        result: rhs,
                        value: params[1],
                        from_bits: 64,
                        to_bits: bits,
                    });

                    let (then_id, _) = e.add_block();
                    let (else_id, _) = e.add_block();
                    let (merge_id, _) = e.add_block();
                    e.seal_and_switch(Terminator::JmpIf(params[0], then_id, else_id), then_id);
                    let answer = e.bin(kind, lhs, rhs);
                    let answer = e.cast_to(CastTarget::Int(64), answer);
                    e.seal_and_switch(Terminator::Jmp(merge_id, vec![answer]), else_id);
                    let zero = e.int_const(IntBits::zero(64));
                    e.seal_and_switch(Terminator::Jmp(merge_id, vec![zero]), merge_id);
                    vec![e.add_parameter(Type::int(64))]
                },
            );
            let what = format!("a guarded int{bits} {kind:?}");
            let compiled = Compiled::new(ssa)
                .unwrap_or_else(|error| panic!("{what} does not compile: {error}"));

            let zero = IntBits::zero(64);
            for wasm in [false, true] {
                // A refusal is a trap on the VM, and a witness the constraints refuse on WASM.
                let refused = |verdict: &Verdict| {
                    if wasm {
                        verdict.rejects()
                    } else {
                        verdict.is_refusal()
                    }
                };
                let run = |taken: u128, rhs: u128, answer: &IntBits| {
                    let inputs = input_block(&[
                        &IntBits::from_u128(1, taken),
                        &IntBits::from_u128(64, rhs),
                        &IntBits::from_u128(1, 1),
                        answer,
                    ]);
                    if wasm {
                        compiled.run_wasm(&inputs).expect("the WASM lane builds")
                    } else {
                        compiled.run(&inputs)
                    }
                };
                let lane = if wasm { "WASM" } else { "VM" };
                let verdict = run(1, 3, &low);
                assert!(
                    verdict.is_accepted(),
                    "{what}, {lane}, taken, fits: {verdict:?}"
                );
                let verdict = run(1, 7, &zero);
                assert!(
                    refused(&verdict),
                    "{what}, {lane}, taken, overflows: {verdict:?}"
                );
                let verdict = run(0, 7, &zero);
                assert!(
                    verdict.is_accepted(),
                    "{what}, {lane}, not taken, overflows: {verdict:?}"
                );
            }
        }
    }
}

/// An asserted signed ordering is the opposite of the unsigned one wherever the signs differ, and
/// under a witnessed condition holds only where the condition does.
///
/// The left operand is negative and the right one positive, so `lhs < rhs` holds signed and fails
/// unsigned, and the reverse fails signed and holds unsigned. 200 is the single cell, 253 the chain
/// on a decomposition, and 320 limbs.
#[test]
fn a_guarded_signed_assertion_holds_only_where_its_guard_does() {
    for bits in [200usize, 253, 320] {
        let sign = IntBits::from_signed(bits, &-(SignedValue::from(1u8) << (bits - 1)));
        for swapped in [false, true] {
            let sign = sign.clone();
            let ssa = main_program(
                &[Type::int(1), Type::int(64)],
                &[Type::int(1)],
                move |e, params| {
                    let positive = e.cast_to(CastTarget::Int(bits), params[1]);
                    let top = e.int_const(sign.clone());
                    let negative = e.bin(BinaryArithOpKind::Or, positive, top);
                    let (lhs, rhs) = if swapped {
                        (positive, negative)
                    } else {
                        (negative, positive)
                    };

                    let (then_id, _) = e.add_block();
                    let (merge_id, _) = e.add_block();
                    e.seal_and_switch(Terminator::JmpIf(params[0], then_id, merge_id), then_id);
                    e.emit(OpCode::AssertCmp {
                        kind: CmpKind::SLt,
                        lhs,
                        rhs,
                    });
                    e.seal_and_switch(Terminator::Jmp(merge_id, vec![]), merge_id);
                    vec![params[0]]
                },
            );
            let what = format!("a guarded int{bits} signed assertion, swapped: {swapped}");
            let compiled = Compiled::new(ssa)
                .unwrap_or_else(|error| panic!("{what} does not compile: {error}"));

            for wasm in [false, true] {
                // A refusal is a trap on the VM, and a witness the constraints refuse on WASM.
                let refused = |verdict: &Verdict| {
                    if wasm {
                        verdict.rejects()
                    } else {
                        verdict.is_refusal()
                    }
                };
                let run = |taken: u128| {
                    let inputs = input_block(&[
                        &IntBits::from_u128(1, taken),
                        &IntBits::from_u128(64, 7),
                        &IntBits::from_u128(1, taken),
                    ]);
                    if wasm {
                        compiled.run_wasm(&inputs).expect("the WASM lane builds")
                    } else {
                        compiled.run(&inputs)
                    }
                };
                let lane = if wasm { "WASM" } else { "VM" };
                let verdict = run(1);
                if swapped {
                    assert!(refused(&verdict), "{what}, {lane}, taken: {verdict:?}");
                } else {
                    assert!(verdict.is_accepted(), "{what}, {lane}, taken: {verdict:?}");
                }
                let verdict = run(0);
                assert!(
                    verdict.is_accepted(),
                    "{what}, {lane}, not taken: {verdict:?}"
                );
            }
        }
    }
}

/// A witnessed sign extension agrees with the model at every target width.
///
/// Each source is a corner constant selected by a witnessed bit against its neighbour, so one
/// program holds them all and the two runs reach every one; the answer is compared whole, so every
/// limb of it is read. 64 to 200 and 100 to 253 add the sign as a place value in one element; 253
/// to 254 fills limbs from a decomposition; 64 to 257 fills a one-bit top limb; 257 to 320 reads
/// the sign out of a one-bit top limb; and 300 to 1000 fills eleven limbs past the source.
#[test]
fn a_witnessed_sign_extension_agrees_with_the_model() {
    for (from, to) in [
        (64usize, 200usize),
        (100, 253),
        (253, 254),
        (64, 257),
        (257, 320),
        (300, 1000),
    ] {
        let values = signed_corners(from);
        let program_values = values.clone();
        let ssa = main_program(&[Type::int(1)], &[Type::int(1)], move |e, params| {
            let mut all = None;
            for (index, value) in program_values.iter().enumerate() {
                let next = &program_values[(index + 1) % program_values.len()];
                let chosen = e.int_const(value.clone());
                let other = e.int_const(next.clone());
                let source = e.select(params[0], other, chosen);
                let extended = e.fresh_value();
                e.emit(OpCode::SExt {
                    result: extended,
                    value: source,
                    from_bits: from,
                    to_bits: to,
                });
                let want = e.int_const(value.sign_extend(to));
                let other_want = e.int_const(next.sign_extend(to));
                let want = e.select(params[0], other_want, want);
                let same = e.eq(extended, want);
                all = Some(match all {
                    None => same,
                    Some(all) => e.bin(BinaryArithOpKind::And, all, same),
                });
            }
            vec![all.expect("there is at least one corner")]
        });
        let what = format!("a witnessed sign extension from int{from} to int{to}");
        let compiled =
            Compiled::new(ssa).unwrap_or_else(|error| panic!("{what} does not compile: {error}"));

        let one = IntBits::from_u128(1, 1);
        for flag in [0u128, 1] {
            let flag = IntBits::from_u128(1, flag);
            let verdict = compiled.run(&input_block(&[&flag, &one]));
            assert!(
                verdict.is_accepted(),
                "{what}, selector {flag:?}: {verdict:?}"
            );
            let wrong = compiled.run(&input_block(&[&flag, &IntBits::zero(1)]));
            assert!(
                !wrong.is_accepted(),
                "{what}, selector {flag:?}: a wrong answer was accepted"
            );
        }
        let verdict = compiled
            .run_wasm(&input_block(&[&IntBits::zero(1), &one]))
            .expect("the WASM lane builds");
        assert!(
            verdict.is_accepted(),
            "{what} on the WASM lane: {verdict:?}"
        );
    }
}

/// A witnessed sign extension whose source's top limb is known at compile time agrees with the
/// model, and so with what the same extension of the same value computes elsewhere.
///
/// Its sign is then known, and the fill is constants rather than a cut: clear for a value widened
/// from a parameter, whose limbs above the parameter are known zeros, and set for a witnessed
/// negative constant, every limb of which is known. The answer is read whole and through a shift,
/// so that every limb of it reaches a constraint.
#[test]
fn a_sign_extension_of_a_known_top_limb_agrees_with_the_model() {
    let (from, to) = (320usize, 1000usize);
    let negative = IntBits::from_signed(from, &-(SignedValue::from(5u8) << 200usize));
    for set in [false, true] {
        let constant = negative.clone();
        let ssa = main_program(&[Type::int(64)], &[Type::int(1)], move |e, params| {
            let source = if set {
                let constant = e.int_const(constant.clone());
                e.cast_to(CastTarget::WitnessOf, constant)
            } else {
                e.cast_to(CastTarget::Int(from), params[0])
            };
            let extended = e.fresh_value();
            e.emit(OpCode::SExt {
                result: extended,
                value: source,
                from_bits: from,
                to_bits: to,
            });
            // The parameter is or-ed into both sides, so that the witnessed constant meets a
            // witnessed value limb by limb.
            let low = e.cast_to(CastTarget::Int(to), params[0]);
            let (extended, want) = if set {
                let want = e.int_const(constant.sign_extend(to));
                (
                    e.bin(BinaryArithOpKind::Or, extended, low),
                    e.bin(BinaryArithOpKind::Or, want, low),
                )
            } else {
                (extended, low)
            };
            let same = e.eq(extended, want);
            let amount = e.int_const(IntBits::from_u128(to, 900));
            let top = e.bin(BinaryArithOpKind::UShr, extended, amount);
            let top_want = e.bin(BinaryArithOpKind::UShr, want, amount);
            let top_same = e.eq(top, top_want);
            vec![e.bin(BinaryArithOpKind::And, same, top_same)]
        });
        let what = format!("a sign extension of a known top limb, sign set: {set}");
        let compiled =
            Compiled::new(ssa).unwrap_or_else(|error| panic!("{what} does not compile: {error}"));
        let parameter = IntBits::all_ones(64);
        let one = IntBits::from_u128(1, 1);
        let verdict = compiled.run(&input_block(&[&parameter, &one]));
        assert!(verdict.is_accepted(), "{what}: {verdict:?}");
        let verdict = compiled
            .run_wasm(&input_block(&[&parameter, &one]))
            .expect("the WASM lane builds");
        assert!(verdict.is_accepted(), "{what}, WASM: {verdict:?}");
    }
}

/// The mutation test for a sign extension into the representation, whose one new column is the
/// source's sign bit: every target limb above the source is that bit times all ones.
#[test]
fn every_witness_column_of_a_sign_extension_is_pinned() {
    let to = 320usize;
    let ssa = main_program(&[Type::int(64)], &[Type::int(64)], move |e, params| {
        let extended = e.fresh_value();
        e.emit(OpCode::SExt {
            result: extended,
            value: params[0],
            from_bits: 64,
            to_bits: to,
        });
        let amount = e.int_const(IntBits::from_u128(to, (to - 64) as u128));
        let top = e.bin(BinaryArithOpKind::UShr, extended, amount);
        vec![e.cast_to(CastTarget::Int(64), top)]
    });
    let compiled = Compiled::new(ssa)
        .unwrap_or_else(|error| panic!("a sign extension to int{to} does not compile: {error}"));
    let negative = IntBits::from_signed(64, &SignedValue::from(-3));
    assert_every_column_is_pinned(
        "a sign extension from int64 to int320",
        &compiled,
        &[&negative, &IntBits::all_ones(64)],
    );
}

// THE REST OF THE WITNESS LANE
// ================================================================================================

/// A 128-bit witness `<<` by a literal amount compiles and agrees, whatever the amount.
///
/// This is `noir-bignum`'s `get_double_modulus`, which is what the three `passport` tests in the
/// local corpus compile. The amount is known, so the shift is a relabelling of bits: the operand is
/// cut where the discarded bits begin, and the piece below moves up.
#[test]
fn a_witness_shift_left_by_a_literal_amount_is_lowered() {
    let bits = 128;
    let amount = 120;
    let value = IntBits::from_u128(bits, 0xab);

    let ssa = main_program(&[Type::int(bits)], &[Type::int(bits)], |e, params| {
        let literal = e.int_const(IntBits::from_u128(bits, amount as u128));
        vec![e.bin(BinaryArithOpKind::UShl, params[0], literal)]
    });
    let compiled = Compiled::new(ssa).expect("a shift by a literal compiles");

    let verdict = compiled.run(&input_block(&[&value, &value.shifted_left(amount)]));
    assert!(verdict.is_accepted(), "int{bits} << {amount}: {verdict:?}");
}

/// A 128-bit shift by a literal **wraps**, as Noir's `<<` does, at every amount below the width.
///
/// A single field product `lhs * 2^n` cannot hold this: past 125 it wraps modulo the field. The
/// shift is limb-wise instead, where the operand is cut where the discarded bits begin, and 126 and
/// 127 are the amounts no single product reaches.
#[test]
fn a_literal_shift_at_a_hundred_and_twenty_eight_bits_wraps_at_every_amount() {
    let bits = 128;
    let value = IntBits::all_ones(bits);
    for amount in [3usize, 120, 125, 126, 127] {
        let ssa = main_program(&[Type::int(bits)], &[Type::int(bits)], |e, params| {
            let literal = e.int_const(IntBits::from_u128(bits, amount as u128));
            vec![e.bin(BinaryArithOpKind::UShl, params[0], literal)]
        });
        let compiled = Compiled::new(ssa)
            .unwrap_or_else(|error| panic!("int{bits} << {amount} does not compile: {error}"));
        let verdict = compiled.run(&input_block(&[&value, &value.shifted_left(amount)]));
        assert!(
            verdict.is_accepted(),
            "int{bits} << {amount} wraps: {verdict:?}"
        );
    }
}

/// A **pure** wide element read out of a sequence and computed on, on both lanes.
///
/// A sequence of wide elements is transposed into one sequence per limb position, pure elements
/// included, so a pure read comes back as limbs. The operations below have no lowering on limbs,
/// being witness gadgets, and are put back together and run in the hint domain instead. The index
/// is the count a loop leaves behind, which no fold answers, so the read survives to the
/// representation. Each answer leaves as its low limb, beside the orderings, and a wrong one is
/// refused.
#[test]
fn a_pure_wide_element_read_out_of_a_sequence_is_computed_on() {
    let bits = 320usize;
    let elements = pure_wide_elements();
    let other = IntBits::from_biguint(bits, &(BigUint::from(5u8).pow(60u32)));
    let read = elements[2].clone();
    assert!(
        BigUint::from(&read).bits() > 254,
        "the element has no field element"
    );
    let low = |value: &IntBits| value.cast(64);
    let model = |op: IntOp| {
        mavros_int_semantics::eval(op, &read, &other)
            .value()
            .unwrap()
    };
    let ops = [
        BinaryArithOpKind::UAdd,
        BinaryArithOpKind::USub,
        BinaryArithOpKind::UDiv,
        BinaryArithOpKind::URem,
        BinaryArithOpKind::SDiv,
        BinaryArithOpKind::SRem,
    ];
    let mut expected: Vec<IntBits> = ops
        .iter()
        .map(|kind| low(&model(IntOp::from(*kind))))
        .collect();
    for op in [CmpOp::ULt, CmpOp::SLt] {
        expected.push(IntBits::from_u128(1, u128::from(read.compare(op, &other))));
        expected.push(IntBits::from_u128(1, u128::from(other.compare(op, &read))));
    }
    let returns: Vec<Type> = expected
        .iter()
        .map(|value| Type::int(value.bits()))
        .collect();

    let program_elements = elements.clone();
    let program_other = other.clone();
    let ssa = main_program(&[], &returns, move |e, _| {
        let constants: Vec<ValueId> = program_elements
            .iter()
            .map(|element| e.int_const(element.clone()))
            .collect();
        let array = e.mk_seq(
            constants,
            SequenceTargetType::Array(program_elements.len()),
            Type::int(bits),
        );
        let count = loop_count(e);

        let element = e.array_get(array, count);
        let other = e.int_const(program_other);
        let mut answers: Vec<ValueId> = ops
            .iter()
            .map(|kind| {
                let answer = e.bin(*kind, element, other);
                e.cast_to(CastTarget::Int(64), answer)
            })
            .collect();
        for kind in [CmpKind::ULt, CmpKind::SLt] {
            answers.push(e.cmp(element, other, kind));
            answers.push(e.cmp(other, element, kind));
        }
        answers
    });
    let compiled = Compiled::new(ssa).unwrap_or_else(|error| {
        panic!("a pure wide element computed on does not compile: {error}")
    });

    let answers: Vec<&IntBits> = expected.iter().collect();
    let verdict = compiled.run(&input_block(&answers));
    assert!(verdict.is_accepted(), "{verdict:?}");
    let verdict = compiled
        .run_wasm(&input_block(&answers))
        .expect("the WASM lane builds");
    assert!(verdict.is_accepted(), "WASM: {verdict:?}");
    for index in 0..expected.len() {
        let mut wrong = expected.clone();
        wrong[index] = wrong[index].xor(&IntBits::one(wrong[index].bits()));
        let wrong: Vec<&IntBits> = wrong.iter().collect();
        assert!(
            !compiled.run(&input_block(&wrong)).is_accepted(),
            "answer {index} accepted a wrong value"
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

/// The same for the signed five past the single cell, whose rejections are built by the carry
/// chain's sign relation and the sign-magnitude gadgets rather than in one element.
///
/// Each operand is a witnessed selection between the pair that fits and the pair that fails, so
/// both are witnessed with neither sign known at compile time. A sum and a product leave the top,
/// a difference the bottom, a division is `INT_MIN / -1`, and a remainder is by zero. 200 is the
/// gadgets on a decomposition of each operand, and 320 on limbs.
#[test]
fn a_dead_wide_witnessed_signed_operation_still_rejects_its_failure() {
    let one = IntBits::from_u128(1, 1);
    for bits in [200usize, 320] {
        let signed = |v: i128| IntBits::from_signed(bits, &SignedValue::from(v));
        let power = |n: usize| IntBits::from_biguint(bits, &(BigUint::from(1u8) << n));
        let min = IntBits::from_signed(bits, &IntBits::signed_min(bits));
        let cases = [
            (
                BinaryArithOpKind::SAdd,
                (signed(1), signed(1)),
                (power(bits - 2), power(bits - 2)),
            ),
            (
                BinaryArithOpKind::SSub,
                (signed(1), signed(2)),
                (min.clone(), signed(1)),
            ),
            (
                BinaryArithOpKind::SMul,
                (signed(3), signed(-5)),
                (power(bits / 2), power(bits / 2 - 1)),
            ),
            (
                BinaryArithOpKind::SDiv,
                (signed(-7), signed(2)),
                (min.clone(), signed(-1)),
            ),
            (
                BinaryArithOpKind::SRem,
                (signed(-7), signed(2)),
                (signed(-7), signed(0)),
            ),
        ];
        for (kind, fits, fails) in cases {
            let (program_fits, program_fails) = (fits.clone(), fails.clone());
            let ssa = main_program(&[Type::int(1)], &[Type::int(1)], move |e, params| {
                let operand = |e: &mut HLBlockEmitter<'_>, fits: &IntBits, fails: &IntBits| {
                    let fits = e.int_const(fits.clone());
                    let fails = e.int_const(fails.clone());
                    e.select(params[0], fits, fails)
                };
                let lhs = operand(e, &program_fits.0, &program_fails.0);
                let rhs = operand(e, &program_fits.1, &program_fails.1);
                e.bin(kind, lhs, rhs);
                vec![e.int_const(IntBits::from_u128(1, 1))]
            });
            let what = format!("a dead int{bits} {kind:?}");
            let compiled = Compiled::new(ssa)
                .unwrap_or_else(|error| panic!("{what} does not compile: {error}"));
            assert!(
                mavros_int_semantics::eval(IntOp::from(kind), &fits.0, &fits.1)
                    .value()
                    .is_some(),
                "{what}: the model refuses the pair meant to fit"
            );
            assert!(
                mavros_int_semantics::eval(IntOp::from(kind), &fails.0, &fails.1)
                    .value()
                    .is_none(),
                "{what}: the model accepts the pair meant to fail"
            );
            for wasm in [false, true] {
                // A refusal is a trap on the VM, and a witness the constraints refuse on WASM.
                let refused = |verdict: &Verdict| {
                    if wasm {
                        verdict.rejects()
                    } else {
                        verdict.is_refusal()
                    }
                };
                let run = |selected: u128| {
                    let inputs = input_block(&[&IntBits::from_u128(1, selected), &one]);
                    if wasm {
                        compiled.run_wasm(&inputs).expect("the WASM lane builds")
                    } else {
                        compiled.run(&inputs)
                    }
                };
                let lane = if wasm { "WASM" } else { "VM" };
                let verdict = run(1);
                assert!(verdict.is_accepted(), "{what}, {lane}, fits: {verdict:?}");
                let verdict = run(0);
                assert!(
                    refused(&verdict),
                    "{what}, {lane}, fails, and nothing reads it: {verdict:?}"
                );
            }
        }
    }
}

/// A pure `INT_MIN` divided by a witnessed `-1` is refused, as is its remainder.
///
/// The guard IR gives a division by a witness no separate check, as its lowering refuses a zero
/// divisor and `INT_MIN / -1` itself. A known dividend leaves the lowering the least to read: its
/// sign and magnitude are known at compile time. 64 and 126 are the single cell, 127 the
/// sign-magnitude gadgets on a decomposition, and 320 limbs, whose divisor is a sign extension of
/// the parameter.
#[test]
fn a_pure_int_min_over_a_witnessed_minus_one_is_refused() {
    for (param, bits) in [(64usize, 64usize), (126, 126), (127, 127), (64, 320)] {
        for kind in [BinaryArithOpKind::SDiv, BinaryArithOpKind::SRem] {
            let ssa = main_program(&[Type::int(param)], &[Type::int(1)], move |e, params| {
                let divisor = if param == bits {
                    params[0]
                } else {
                    let extended = e.fresh_value();
                    e.emit(OpCode::SExt {
                        result: extended,
                        value: params[0],
                        from_bits: param,
                        to_bits: bits,
                    });
                    extended
                };
                let min = e.int_const(IntBits::one(bits).shifted_left(bits - 1));
                let answer = e.bin(kind, min, divisor);
                let zero = e.int_const(IntBits::zero(bits));
                vec![e.eq(answer, zero)]
            });
            let what = format!("int{bits} INT_MIN {kind:?} a witnessed divisor");
            let compiled = Compiled::new(ssa)
                .unwrap_or_else(|error| panic!("{what} does not compile: {error}"));

            // Refused whichever answer is declared, so it is the division that fails rather than
            // the return check.
            let minus_one = IntBits::all_ones(param);
            for declared in [0u128, 1] {
                let inputs = input_block(&[&minus_one, &IntBits::from_u128(1, declared)]);
                let verdict = compiled.run(&inputs);
                assert!(verdict.is_refusal(), "{what} of -1: {verdict:?}");
                let verdict = compiled.run_wasm(&inputs).expect("the WASM lane builds");
                assert!(verdict.rejects(), "{what} of -1, WASM: {verdict:?}");
            }

            // `INT_MIN / 1` is `INT_MIN`, which is not zero, and `INT_MIN % 1` is zero.
            let declared = IntBits::from_u128(1, u128::from(kind == BinaryArithOpKind::SRem));
            let inputs = input_block(&[&IntBits::one(param), &declared]);
            let verdict = compiled.run(&inputs);
            assert!(verdict.is_accepted(), "{what} of 1: {verdict:?}");
            let verdict = compiled.run_wasm(&inputs).expect("the WASM lane builds");
            assert!(verdict.is_accepted(), "{what} of 1, WASM: {verdict:?}");
        }
    }
}

/// A witnessed bitwise operation computes at **every** width, as the unsigned sum, difference and
/// ordering do, but by a different route: limb by limb with no carry between them.
///
/// **Each width is a different shape.** `40` is past the two limb cases and past the 32 bits the
/// narrow spread opcode takes. `96` spreads to 192 bits, past the widest value a host word holds.
/// `200` is past the narrow bound but not held as limbs. `254` is the first width the
/// representation splits **raggedly**, so its top limb is 62 bits.
///
/// What reaches all four is the same thing: the decomposition runs at the **half-limb**, which
/// keeps every spread within the narrow opcode, and the top half-limb carries only the bits that
/// are left.
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
/// `!x ^ y` is the shape for two reasons. It needs no constant wider than the return: the low 64
/// bits of a wide complement are the complement of the low 64 bits, which is what the declared
/// return reads. And the answer **depends on the complement**: the simplifier's `~~x -> x` rewrite
/// folds `!!x` back to `x` long before any pass this exercises, so a double complement would only
/// test that a widening and a truncating cast round trip.
///
/// **253 and 254 are the pair that matters.** At 253 the operand is one field element and the
/// complement is `2^bits - 1 - value`; at 254 it is limbs, complemented one by one at each limb's
/// width. The same program either side of the threshold.
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
/// The witness lowering does not rewrite a pure complement into `(2^bits - 1) - value` in the
/// field, just as it leaves a pure `and` alone. Nothing refuses it at any width either, because bit
/// `i` of the answer depends only on bit `i` of the operand. The `xor` against a witnessed operand
/// puts the answer where a constraint reads it: a pure value alone is a hint, and a wrong hint that
/// nothing checks does not fail a test.
///
/// **This does not reach the opcodes.** An entry point cannot build a pure non-constant value,
/// since witness-ness is inferred and taints every non-constant value here, so the operand is a
/// constant and the complement is folded. This tests that no width is refused, and a
/// `2^bits`-wide complement through the constant evaluators; the VM crate holds `NotInt`,
/// `NotInt128` and `NotIntn` to the model, in `the_complement_opcode_agrees_with_the_model` and
/// `the_intn_complement_agrees_with_the_model`.
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

// THE PURE SIGNED LANE
// ================================================================================================

/// What a pure signed sweep computes: a signed arithmetic operation, or the signed ordering.
#[derive(Clone, Copy, Debug)]
enum PureSigned {
    Arith(BinaryArithOpKind),
    Ordering,
}

impl PureSigned {
    /// Every signed operation there is, the ordering included.
    const ALL: [PureSigned; 8] = [
        PureSigned::Arith(BinaryArithOpKind::SAdd),
        PureSigned::Arith(BinaryArithOpKind::SSub),
        PureSigned::Arith(BinaryArithOpKind::SMul),
        PureSigned::Arith(BinaryArithOpKind::SDiv),
        PureSigned::Arith(BinaryArithOpKind::SRem),
        PureSigned::Arith(BinaryArithOpKind::SShl),
        PureSigned::Arith(BinaryArithOpKind::SShr),
        PureSigned::Ordering,
    ];

    fn emit(self, e: &mut HLBlockEmitter<'_>, lhs: ValueId, rhs: ValueId) -> ValueId {
        match self {
            PureSigned::Arith(kind) => e.bin(kind, lhs, rhs),
            PureSigned::Ordering => e.cmp(lhs, rhs, CmpKind::SLt),
        }
    }

    /// The model's answer, or [`None`] where it rejects the operands.
    fn model(self, lhs: &IntBits, rhs: &IntBits) -> Option<IntBits> {
        match self {
            PureSigned::Arith(kind) => {
                mavros_int_semantics::eval(IntOp::from(kind), lhs, rhs).value()
            }
            PureSigned::Ordering => Some(IntBits::from_u128(
                1,
                u128::from(lhs.compare(CmpOp::SLt, rhs)),
            )),
        }
    }

    /// The right operand a triple takes on a run that does not name it: zero, except for a divisor.
    fn idle_rhs(self, bits: usize) -> IntBits {
        match self {
            PureSigned::Arith(BinaryArithOpKind::SDiv | BinaryArithOpKind::SRem) => {
                IntBits::from_u128(bits, 1)
            }
            PureSigned::Arith(_) | PureSigned::Ordering => IntBits::zero(bits),
        }
    }
}

/// The signed corners of a `bits`-wide value: both boundaries and one step inside the lower, zero
/// and one either side of it, and a positive value with its low half set.
fn signed_corners(bits: usize) -> Vec<IntBits> {
    let signed = |v: SignedValue| IntBits::from_signed(bits, &v);
    let (min, max) = (IntBits::signed_min(bits), IntBits::signed_max(bits));
    dedup(vec![
        signed(SignedValue::from(0)),
        signed(SignedValue::from(1)),
        signed(SignedValue::from(-1)),
        signed(min.clone()),
        signed(min + 1),
        signed(max),
        IntBits::all_ones(bits / 2).cast(bits),
    ])
}

/// How many operations one pure program holds: compiling one grows faster than linearly in its
/// length, so a sweep is many small programs rather than one large one.
const PURE_TRIPLES_PER_PROGRAM: usize = 25;

/// A few operand pairs per signed operation at `bits`: each combination of signs, the boundaries,
/// and every reason the model has to reject the operation, each at least once.
fn signed_spot_triples(bits: usize) -> Vec<(PureSigned, IntBits, IntBits)> {
    use BinaryArithOpKind::{SAdd, SDiv, SMul, SRem, SShl, SShr, SSub};
    let v = |v: i64| IntBits::from_signed(bits, &SignedValue::from(v));
    let min = IntBits::from_signed(bits, &IntBits::signed_min(bits));
    let max = IntBits::from_signed(bits, &IntBits::signed_max(bits));
    let half = IntBits::all_ones(bits / 2).cast(bits);
    let amount = |a: usize| IntBits::from_u128(bits, a as u128);
    let arith = PureSigned::Arith;
    vec![
        (arith(SAdd), max.clone(), v(1)),
        (arith(SAdd), min.clone(), v(-1)),
        (arith(SAdd), min.clone(), max.clone()),
        (arith(SAdd), half.clone(), v(-5)),
        (arith(SSub), min.clone(), v(1)),
        (arith(SSub), max.clone(), v(-1)),
        (arith(SSub), v(-1), min.clone()),
        (arith(SSub), v(0), max.clone()),
        (arith(SMul), min.clone(), v(-1)),
        (arith(SMul), min.clone(), v(1)),
        (arith(SMul), max.clone(), v(-1)),
        (arith(SMul), half.clone(), half.clone()),
        (arith(SMul), half.clone(), v(-3)),
        (arith(SMul), v(-1), v(-1)),
        (arith(SDiv), min.clone(), v(-1)),
        (arith(SDiv), v(7), v(0)),
        (arith(SDiv), min.clone(), v(2)),
        (arith(SDiv), v(-7), v(2)),
        (arith(SDiv), v(7), v(-2)),
        (arith(SDiv), min.clone(), min.clone()),
        (arith(SRem), min.clone(), v(-1)),
        (arith(SRem), v(7), v(0)),
        (arith(SRem), v(-7), v(2)),
        (arith(SRem), v(7), v(-2)),
        (arith(SRem), min.clone(), max.clone()),
        (arith(SShl), v(-3), amount(bits - 1)),
        (arith(SShl), half.clone(), amount(HOST_LIMB_BITS)),
        (arith(SShl), v(1), amount(bits)),
        (arith(SShl), v(1), v(-1)),
        (arith(SShr), min.clone(), amount(bits - 1)),
        (arith(SShr), min.clone(), amount(HOST_LIMB_BITS)),
        (arith(SShr), max.clone(), amount(1)),
        (arith(SShr), v(-1), amount(bits)),
        (arith(SShr), v(-1), v(-1)),
        (PureSigned::Ordering, min.clone(), max.clone()),
        (PureSigned::Ordering, max, min),
        (PureSigned::Ordering, v(-1), v(0)),
        (PureSigned::Ordering, v(0), v(-1)),
        (PureSigned::Ordering, v(-1), v(-1)),
    ]
}

/// `cond ? if_t : if_f` at `bits`, computed rather than selected: the AD half of a program keeps a
/// copy of each unconstrained function, and nothing there lowers a `Select`. The condition is
/// broadcast into a mask by moving it to the sign bit and back arithmetically.
fn pure_choice(
    e: &mut HLBlockEmitter<'_>,
    cond: ValueId,
    if_t: ValueId,
    if_f: ValueId,
    bits: usize,
) -> ValueId {
    let top = e.int_const(IntBits::from_u128(bits, (bits - 1) as u128));
    let cond = e.cast_to(CastTarget::Int(bits), cond);
    let raised = e.bin(BinaryArithOpKind::UShl, cond, top);
    let mask = e.bin(BinaryArithOpKind::SShr, raised, top);
    let unmask = e.not(mask);
    let kept = e.bin(BinaryArithOpKind::And, if_t, mask);
    let other = e.bin(BinaryArithOpKind::And, if_f, unmask);
    e.bin(BinaryArithOpKind::Or, kept, other)
}

/// `main(chosen: u1, name: u32, also_chosen: u1) -> u1`, which passes its parameters to an
/// **unconstrained** function whose body `body` builds, and returns its answer.
///
/// Every value in the body is pure and the parameters are not literals, so the answer comes from
/// the arithmetic of the interpreter and of the compiled WASM, behind the guard IR, with no
/// constraint anywhere and no constant folder able to answer first.
fn unconstrained_program(
    body: impl FnOnce(&mut HLBlockEmitter<'_>, &[ValueId]) -> ValueId,
) -> HLSSA {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let body_id = ssa.add_function("pure_body".to_string());
    let mut sb = HLSSABuilder::new(&mut ssa);
    let parameters = [Type::int(1), Type::int(32), Type::int(1)];
    sb.modify_function(body_id, |b| {
        b.function.add_return_type(Type::int(1));
        let entry = b.function.get_entry_id();
        let mut e = b
            .block(entry)
            .with_source_location(SourceLocation::synthetic("pure_body"));
        let params: Vec<ValueId> = parameters
            .iter()
            .map(|t| e.add_parameter(t.clone()))
            .collect();
        let answer = body(&mut e, &params);
        e.terminate_return(vec![answer]);
    });
    sb.modify_function(main_id, |b| {
        b.function.add_return_type(Type::int(1));
        let entry = b.function.get_entry_id();
        let mut e = b
            .block(entry)
            .with_source_location(SourceLocation::synthetic("oracle_main"));
        let params: Vec<ValueId> = parameters
            .iter()
            .map(|t| e.add_parameter(t.clone()))
            .collect();
        let results = e.call_unconstrained(body_id, params, 1);
        e.terminate_return(results);
    });
    ssa
}

/// The inputs an [`unconstrained_program`] runs on: both choices made, `name` named, and the answer
/// `1` expected.
fn unconstrained_inputs(name: u128) -> Vec<InputValueOrdered> {
    let one = IntBits::from_u128(1, 1);
    input_block(&[&one, &IntBits::from_u128(32, name), &one, &one])
}

/// `triples`, computed outside the witness domain, against the model, on both lanes.
///
/// The operations live in an [`unconstrained_program`]. Each operand is a corner constant chosen by
/// a parameter, and the pairs the model rejects are chosen by a second parameter naming them, as
/// [`check_limbed_pairs`] does.
fn check_pure_signed_triples(bits: usize, triples: Vec<(PureSigned, IntBits, IntBits)>) {
    let answers: Vec<Option<IntBits>> = triples.iter().map(|(op, a, b)| op.model(a, b)).collect();
    let zero = IntBits::zero(bits);
    let unnamed: Vec<IntBits> = triples
        .iter()
        .map(|(op, _, _)| {
            op.model(&zero, &op.idle_rhs(bits))
                .expect("the operation accepts zero against its idle right operand")
        })
        .collect();

    let (program_triples, program_answers) = (triples.clone(), answers.clone());
    let ssa = unconstrained_program(move |e, params| {
        let zero = e.int_const(IntBits::zero(bits));
        let mut all = None;
        for (index, (((op, lhs, rhs), answer), unnamed)) in program_triples
            .iter()
            .zip(&program_answers)
            .zip(&unnamed)
            .enumerate()
        {
            let (left_chosen, right_chosen, want) = match answer {
                Some(want) => (params[0], params[2], want.clone()),
                None => {
                    let name = e.int_const(IntBits::from_u128(32, index as u128));
                    let named = e.eq(params[1], name);
                    (named, named, unnamed.clone())
                }
            };
            let lhs = e.int_const(lhs.clone());
            let lhs = pure_choice(e, left_chosen, lhs, zero, bits);
            let rhs = e.int_const(rhs.clone());
            let idle = e.int_const(op.idle_rhs(bits));
            let rhs = pure_choice(e, right_chosen, rhs, idle, bits);
            let got = op.emit(e, lhs, rhs);
            let want = e.int_const(want);
            let same = e.eq(got, want);
            all = Some(match all {
                None => same,
                Some(all) => e.bin(BinaryArithOpKind::And, all, same),
            });
        }
        all.expect("there is at least one triple")
    });
    let compiled = Compiled::new(ssa)
        .unwrap_or_else(|error| panic!("pure signed int{bits} does not compile: {error}"));

    let inputs = unconstrained_inputs;
    let none = u128::from(u32::MAX);
    let verdict = compiled.run(&inputs(none));
    assert!(
        verdict.is_accepted(),
        "pure signed int{bits} disagrees with the model on some triple it accepts: {verdict:?}"
    );
    let verdict = compiled
        .run_wasm(&inputs(none))
        .expect("the WASM lane builds");
    assert!(
        verdict.is_accepted(),
        "pure signed int{bits} disagrees with the model on the WASM lane: {verdict:?}"
    );

    for (index, ((op, lhs, rhs), answer)) in triples.iter().zip(&answers).enumerate() {
        if answer.is_some() {
            continue;
        }
        let verdict = compiled.run(&inputs(index as u128));
        assert!(
            verdict.is_refusal(),
            "pure {op:?}({lhs:?}, {rhs:?}) at {bits} bits is rejected by the model, but the \
             pipeline answered {verdict:?}"
        );
        let verdict = compiled
            .run_wasm(&inputs(index as u128))
            .expect("the WASM lane builds");
        assert!(
            verdict.is_refusal(),
            "pure {op:?}({lhs:?}, {rhs:?}) at {bits} bits is rejected by the model, but the \
             WASM lane answered {verdict:?}"
        );
    }
}

/// Every signed operation outside the witness domain agrees with the model past the host limb:
/// at 100 bits in the double lane, whose sign bit is not the lane's top bit, and at 200 in the
/// wide one, whose top limb is partial.
#[test]
fn a_pure_signed_operation_agrees_with_the_model_past_the_host_limb() {
    for bits in [100usize, 200] {
        for chunk in signed_spot_triples(bits).chunks(PURE_TRIPLES_PER_PROGRAM) {
            check_pure_signed_triples(bits, chunk.to_vec());
        }
    }
}

/// A pure sign extension agrees with the model on both lanes: within the narrow range, where it is
/// a place value added in one field element, from a source past the host limb, and past the narrow
/// range, where it is two shifts at the target width.
#[test]
fn a_pure_sign_extension_agrees_with_the_model() {
    let extensions = [(100usize, 128usize), (100, 1000), (128, 129), (200, 16384)];
    let ssa = unconstrained_program(move |e, params| {
        let mut all = None;
        for (from, to) in extensions {
            let zero = e.int_const(IntBits::zero(from));
            for value in signed_corners(from) {
                let want = e.int_const(value.sign_extend(to));
                let value = e.int_const(value);
                let value = pure_choice(e, params[0], value, zero, from);
                let extended = e.sext(value, from, to);
                let same = e.eq(extended, want);
                all = Some(match all {
                    None => same,
                    Some(all) => e.bin(BinaryArithOpKind::And, all, same),
                });
            }
        }
        all.expect("there is at least one extension")
    });
    let compiled = Compiled::new(ssa)
        .unwrap_or_else(|error| panic!("a pure sign extension does not compile: {error}"));
    let inputs = unconstrained_inputs(0);
    let verdict = compiled.run(&inputs);
    assert!(verdict.is_accepted(), "{verdict:?}");
    let verdict = compiled.run_wasm(&inputs).expect("the WASM lane builds");
    assert!(verdict.is_accepted(), "WASM: {verdict:?}");
}

/// A pure spread and unspread agree with the model on both lanes, either side of every boundary
/// where an opcode or a backend's ladder hands over to the wide path, including with a set bit
/// above the read, which must be discarded.
#[test]
fn a_pure_spread_and_unspread_agree_with_the_model() {
    // `(container, read)`: the narrow opcode and the LLVM ladder, the VM's wide opcode under the
    // ladder, both wide paths, and the widest container each op takes.
    let spreads = [
        (8usize, 5usize),
        (32, 32),
        (33, 20),
        (64, 64),
        (65, 65),
        (100, 77),
        (1000, 999),
        (8192, 8192),
    ];
    let unspreads = [
        (8usize, 7usize),
        (64, 64),
        (65, 65),
        (128, 100),
        (129, 129),
        (1000, 501),
        (16383, 16383),
        (16384, 10000),
    ];
    let values = |bits: usize| {
        let every_other = ((BigUint::from(1u8) << bits) - 1u8) / 3u8;
        [
            IntBits::all_ones(bits),
            IntBits::from_biguint(bits, &every_other),
            IntBits::one(bits).shifted_left(bits - 1),
        ]
    };

    let ssa = unconstrained_program(move |e, params| {
        let mut checks = Vec::new();
        for (bits, read) in spreads {
            let zero = e.int_const(IntBits::zero(bits));
            for value in values(bits) {
                let want = e.int_const(value.spread(read));
                let value = e.int_const(value);
                let value = pure_choice(e, params[0], value, zero, bits);
                let spread = e.spread(value, read);
                checks.push(e.eq(spread, want));
            }
        }
        for (bits, read) in unspreads {
            let zero = e.int_const(IntBits::zero(bits));
            for value in values(bits) {
                let (want_odd, want_even) = value.unspread(read);
                let (want_odd, want_even) = (e.int_const(want_odd), e.int_const(want_even));
                let value = e.int_const(value);
                let value = pure_choice(e, params[0], value, zero, bits);
                let (odd, even) = e.unspread(value, read);
                checks.push(e.eq(odd, want_odd));
                checks.push(e.eq(even, want_even));
            }
        }
        checks
            .into_iter()
            .reduce(|all, same| e.bin(BinaryArithOpKind::And, all, same))
            .expect("there is at least one check")
    });
    let compiled = Compiled::new(ssa)
        .unwrap_or_else(|error| panic!("a pure spread does not compile: {error}"));
    let inputs = unconstrained_inputs(0);
    let verdict = compiled.run(&inputs);
    assert!(verdict.is_accepted(), "{verdict:?}");
    let verdict = compiled.run_wasm(&inputs).expect("the WASM lane builds");
    assert!(verdict.is_accepted(), "WASM: {verdict:?}");
}

/// A pure signed operation under a witnessed condition is checked only where the condition holds.
///
/// Inside a branch on a witness a pure operation is guarded, so its rejection is an assertion under
/// the guard, built by the guarded arms of `LowerPureGuards` from the same sign bits and magnitudes
/// as the unguarded ones. Each operand is a pure value a loop computes, so it is never a literal
/// and no folder answers first, and each triple has a separate condition: with every condition off
/// the program is accepted whatever the operands, and with one on it agrees with the model or is
/// refused where the model rejects.
#[test]
fn a_guarded_pure_signed_operation_is_checked_only_where_its_guard_holds() {
    for bits in [100usize, 200] {
        let signed = |v: i64| IntBits::from_signed(bits, &SignedValue::from(v));
        let min = IntBits::from_signed(bits, &IntBits::signed_min(bits));
        let max = IntBits::from_signed(bits, &IntBits::signed_max(bits));
        let amount = |v: usize| IntBits::from_u128(bits, v as u128);
        let arith = PureSigned::Arith;
        use BinaryArithOpKind::{SAdd, SDiv, SMul, SRem, SShl, SShr, SSub};
        let triples = vec![
            (arith(SAdd), max.clone(), signed(1)),
            (arith(SAdd), max.clone(), signed(-1)),
            (arith(SSub), min.clone(), signed(1)),
            (arith(SSub), min.clone(), signed(-1)),
            (arith(SMul), min.clone(), signed(-1)),
            (arith(SMul), min.clone(), signed(1)),
            (arith(SMul), max.clone(), signed(-1)),
            (arith(SDiv), min.clone(), signed(-1)),
            (arith(SDiv), min.clone(), signed(2)),
            (arith(SDiv), signed(7), signed(0)),
            (arith(SRem), min.clone(), signed(-1)),
            (arith(SRem), signed(-7), signed(2)),
            (arith(SShl), signed(-3), amount(bits)),
            (arith(SShl), signed(-3), amount(bits - 1)),
            (arith(SShr), min.clone(), signed(-1)),
            (arith(SShr), min.clone(), amount(bits - 1)),
            (PureSigned::Ordering, min.clone(), max.clone()),
        ];
        check_guarded_pure_signed(bits, &triples);
    }
}

/// `triples` as pure operations on operands a loop computes, each under a separate witnessed
/// condition, checked against the model; see
/// [`a_guarded_pure_signed_operation_is_checked_only_where_its_guard_holds`].
fn check_guarded_pure_signed(bits: usize, triples: &[(PureSigned, IntBits, IntBits)]) {
    let answers: Vec<Option<IntBits>> = triples.iter().map(|(op, a, b)| op.model(a, b)).collect();
    let (program_triples, program_answers) = (triples.to_vec(), answers.clone());
    let mut parameters = vec![Type::int(1); triples.len()];
    parameters.push(Type::int(8));
    let ssa = main_program(&parameters, &[Type::int(1)], move |e, params| {
        let yes = e.int_const(IntBits::from_u128(1, 1));
        let start = e.int_const(IntBits::zero(8));
        let (head, _) = e.add_block();
        let (body, _) = e.add_block();
        let (done, _) = e.add_block();
        e.seal_and_switch(Terminator::Jmp(head, vec![start, yes]), head);
        let count = e.add_parameter(Type::int(8));
        let last = e.add_parameter(Type::int(1));
        let limit = e.int_const(IntBits::from_u128(8, 2));
        let more = e.cmp(count, limit, CmpKind::ULt);
        e.seal_and_switch(Terminator::JmpIf(more, body, done), body);

        // Zero, but only once the loop has run, so every operand below is never a literal.
        let hundred = e.int_const(IntBits::from_u128(8, 100));
        let nothing = e.bin(BinaryArithOpKind::UDiv, count, hundred);
        let nothing = e.cast_to(CastTarget::Int(bits), nothing);

        let mut all = yes;
        for (index, ((op, lhs, rhs), answer)) in
            program_triples.iter().zip(&program_answers).enumerate()
        {
            let lhs = e.int_const(lhs.clone());
            let lhs = e.bin(BinaryArithOpKind::Xor, nothing, lhs);
            let rhs = e.int_const(rhs.clone());
            let rhs = e.bin(BinaryArithOpKind::Xor, nothing, rhs);
            let (taken, _) = e.add_block();
            let (skipped, _) = e.add_block();
            let (merge, _) = e.add_block();
            e.seal_and_switch(Terminator::JmpIf(params[index], taken, skipped), taken);
            let got = op.emit(e, lhs, rhs);
            let agrees = match answer {
                Some(want) => {
                    let want = e.int_const(want.clone());
                    e.cmp(got, want, CmpKind::Eq)
                }
                None => yes,
            };
            e.seal_and_switch(Terminator::Jmp(merge, vec![agrees]), skipped);
            e.seal_and_switch(Terminator::Jmp(merge, vec![yes]), merge);
            let merged = e.add_parameter(Type::int(1));
            all = e.bin(BinaryArithOpKind::And, all, merged);
        }
        let one = e.int_const(IntBits::from_u128(8, 1));
        let next = e.bin(BinaryArithOpKind::UAdd, count, one);
        e.seal_and_switch(Terminator::Jmp(head, vec![next, all]), done);
        vec![last]
    });
    let compiled = Compiled::new(ssa)
        .unwrap_or_else(|error| panic!("guarded pure signed int{bits} does not compile: {error}"));

    let one = IntBits::from_u128(1, 1);
    let off = IntBits::zero(1);
    let run = |taken: Option<usize>, wasm: bool| {
        let conditions: Vec<IntBits> = (0..triples.len())
            .map(|i| {
                if Some(i) == taken {
                    one.clone()
                } else {
                    off.clone()
                }
            })
            .collect();
        let mut inputs: Vec<&IntBits> = conditions.iter().collect();
        let zero = IntBits::zero(8);
        inputs.push(&zero);
        inputs.push(&one);
        let inputs = input_block(&inputs);
        if wasm {
            compiled.run_wasm(&inputs).expect("the WASM lane builds")
        } else {
            compiled.run(&inputs)
        }
    };

    for wasm in [false, true] {
        let verdict = run(None, wasm);
        assert!(
            verdict.is_accepted(),
            "int{bits}, nothing taken, WASM {wasm}: {verdict:?}"
        );
        for (index, ((op, lhs, rhs), answer)) in triples.iter().zip(&answers).enumerate() {
            let verdict = run(Some(index), wasm);
            let what = format!("int{bits} {op:?}({lhs:?}, {rhs:?}) taken, WASM {wasm}");
            if answer.is_some() {
                assert!(verdict.is_accepted(), "{what}: {verdict:?}");
            } else if wasm {
                // The check under a witnessed condition is a constraint, which the WASM witness
                // generator does not evaluate, so there the witness it writes is what fails.
                assert!(verdict.rejects(), "{what}: {verdict:?}");
            } else {
                assert!(verdict.is_refusal(), "{what}: {verdict:?}");
            }
        }
    }
}

/// The same over every corner pair, at every width the double lane holds and every width the wide
/// corner set names except 16383, which gets only the spot triples: it differs from 16384 only in
/// how full its top limb is, which 1000 covers.
///
/// The cost is the WASM engine compiling the module, superlinear in a function's length (about
/// half a minute for 25 triples at 16384 bits, a second for 3), so past 1000 bits a program holds
/// [`WIDEST_TRIPLES_PER_PROGRAM`] triples rather than [`PURE_TRIPLES_PER_PROGRAM`].
#[test]
#[ignore = "every corner pair at every wide width but 16383; around twenty-five minutes"]
fn the_pure_signed_corner_matrix_agrees_with_the_model() {
    for bits in [65usize, 96, 127, 128]
        .into_iter()
        .chain(corners::WIDE_WIDTHS)
    {
        let triples: Vec<(PureSigned, IntBits, IntBits)> = if bits == 16383 {
            signed_spot_triples(bits)
        } else {
            PureSigned::ALL
                .into_iter()
                .flat_map(|op| {
                    let kind = match op {
                        PureSigned::Arith(kind) => IntOp::from(kind),
                        PureSigned::Ordering => IntOp::SSub,
                    };
                    let (values, rhs) = corners::wide_operands(kind, bits);
                    values
                        .into_iter()
                        .flat_map(move |a| rhs.clone().into_iter().map(move |b| (op, a.clone(), b)))
                        .collect::<Vec<_>>()
                })
                .collect()
        };
        let per_program = if bits > 1000 {
            WIDEST_TRIPLES_PER_PROGRAM
        } else {
            PURE_TRIPLES_PER_PROGRAM
        };
        for chunk in triples.chunks(per_program) {
            check_pure_signed_triples(bits, chunk.to_vec());
        }
    }
}

/// How many operations one pure program holds past 1000 bits, where the WASM engine's compile
/// time is what bounds a sweep; see `the_pure_signed_corner_matrix_agrees_with_the_model`.
const WIDEST_TRIPLES_PER_PROGRAM: usize = 3;

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

        // The answer is constrained, not just computed: declaring a different one is refused by
        // the wrapper's return check.
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
/// The target's top limb is narrower than the source limb it comes from. Cutting it with a
/// `BitRange` gives the right bits at the wrong type, as a window keeps its source's width, so it
/// cannot pair with an operand split at its true width. Limb-aligned targets (256, 320) never cut a
/// limb, so the widths here are not multiples of the limb.
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

/// And a value whose range already fits the target costs no window.
///
/// Noir's frontend emits `bit_range(v, 0, n)` followed by the cast, the shape of every compiled
/// program, and the window's range transfer bounds its result by `2^n`. So the rewrite must decline
/// there, or every narrowing in the corpus would pay for a second window. Stated as a comparison
/// rather than a count so it survives a change that moves both.
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
/// The `vm` conformance sweep shows the wide bodies compute the right answers, and the
/// interpreter's round-trip test shows the encoding survives dispatch. Neither checks _which_
/// opcode a width picks, which is all of the codegen half: a width that fell through to the double
/// lane would compute modulo the wrong power of two and all of those tests would still pass.
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

/// The three lanes are chosen by width, and each keeps a separate set of opcodes.
///
/// One test rather than three because it checks the partition: the failure worth catching is a
/// boundary that moved, which shows up as two widths agreeing that should not.
#[test]
fn each_width_picks_its_own_lane() {
    for (bits, expected, forbidden) in [
        (64usize, "add_int ", "add_int128"),
        (128, "add_int128", "add_intn"),
        (129, "add_intn", "add_int128"),
        // Two cells, and so the double lane: its six width-dependent opcodes carry a width, so it
        // spans all of `65..=128`, not just 128.
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
/// Only the listing shows which opcode a width picks: the interpreter's tests exercise the bodies
/// directly, so a width that fell through to the double lane would read two cells of a four-cell
/// value and they would all still pass.
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

/// An operand narrower than the result is an **ICE** in codegen.
///
/// Bytecode reads both operands at the result's cell count, so a narrower one is read past its
/// end above the cell lane. The type analysis refuses a shift whose amount is not its value's type
/// (see [`an_operand_of_a_type_the_ir_does_not_admit_is_a_compiler_error`]) but not a sum, so a sum
/// is what reaches the backstop. This drives codegen directly, `bytecode_listing` running no
/// passes.
#[test]
#[should_panic(expected = "does not cover the 4 frame cell(s)")]
fn a_wide_sum_with_a_narrower_operand_is_an_ice_in_codegen() {
    let ssa = main_program(
        &[Type::int(256), Type::int(32)],
        &[Type::int(256)],
        |e, params| vec![e.bin(BinaryArithOpKind::UAdd, params[0], params[1])],
    );
    let _ = bytecode_listing(&ssa);
}

/// The same backstop one lane down, where the stray cell belongs to the result.
#[test]
#[should_panic(expected = "does not cover the 2 frame cell(s)")]
fn a_double_lane_sum_with_a_narrower_operand_is_an_ice_in_codegen() {
    let ssa = main_program(
        &[Type::int(96), Type::int(32)],
        &[Type::int(96)],
        |e, params| vec![e.bin(BinaryArithOpKind::UAdd, params[0], params[1])],
    );
    let _ = bytecode_listing(&ssa);
}

/// Each of these is a compiler error, as nothing produces one and no backend reads one:
///
/// - a shift whose amount is not its value's type;
/// - a sequence index that is not an `int32`;
/// - a cast into `Field` of an integer wider than the field carries injectively;
/// - a sign extension that does not widen;
/// - a `Map` cast that converts its elements rather than adding or stripping `WitnessOf`.
///
/// The cast is refused on what the program says, so a small constant that folding could answer is
/// refused too, as is one in a function only a function pointer reaches.
#[test]
fn an_operand_of_a_type_the_ir_does_not_admit_is_a_compiler_error() {
    let shifting = |bits: usize, amount: Type| {
        main_program(
            &[Type::int(bits), amount],
            &[Type::int(bits)],
            |e, params| vec![e.bin(BinaryArithOpKind::UShl, params[0], params[1])],
        )
    };
    let indexing = |index: Type| {
        main_program(&[index], &[Type::field()], |e, params| {
            let elements: Vec<ValueId> = (0..3).map(|k| e.field_const(k)).collect();
            let array = e.mk_seq(elements, SequenceTargetType::Array(3), Type::field());
            vec![e.array_get(array, params[0])]
        })
    };
    let wide = first_width_past_the_field();
    let to_field = "cannot be cast to Field";
    let shift = "a shift amount must have the shifted value's type";
    let index = "a sequence index must be an int32";
    let sign_extending = |to_bits: usize| {
        main_program(&[Type::int(32)], &[Type::int(to_bits)], move |e, params| {
            vec![e.sext(params[0], 32, to_bits)]
        })
    };
    let cases = [
        (
            "a wide parameter cast to Field",
            to_field,
            main_program(&[Type::int(wide)], &[Type::field()], |e, params| {
                vec![e.cast_to_field(params[0])]
            }),
        ),
        (
            "a small wide constant cast to Field",
            to_field,
            main_program(&[], &[Type::field()], |e, _| {
                let value = e.int_const(IntBits::from_u128(wide, 5));
                vec![e.cast_to_field(value)]
            }),
        ),
        (
            "a wide cast behind a function pointer",
            to_field,
            program_calling_through_a_function_pointer(wide),
        ),
        (
            "an int8 shifted by an int64",
            shift,
            shifting(8, Type::int(64)),
        ),
        (
            "an int256 shifted by an int32",
            shift,
            shifting(256, Type::int(32)),
        ),
        (
            "an int64 shifted by a field",
            shift,
            shifting(64, Type::field()),
        ),
        ("an int64 index", index, indexing(Type::int(64))),
        ("an int8 index", index, indexing(Type::int(8))),
        ("a field index", index, indexing(Type::field())),
        (
            "a sign extension to the same width",
            "SExt must widen",
            sign_extending(32),
        ),
        (
            "a sign extension that narrows",
            "SExt must widen",
            sign_extending(16),
        ),
        (
            "an array of integers mapped to Field",
            "a Map cast only adds or strips WitnessOf",
            main_program(
                &[Type::int(32).array_of(2)],
                &[Type::field().array_of(2)],
                |e, params| {
                    vec![e.cast_to(CastTarget::Map(Box::new(CastTarget::Field)), params[0])]
                },
            ),
        ),
        (
            "an array of wide integers mapped to a wider integer",
            "a Map cast only adds or strips WitnessOf",
            main_program(
                &[Type::int(256).array_of(2)],
                &[Type::int(320).array_of(2)],
                |e, params| {
                    vec![e.cast_to(CastTarget::Map(Box::new(CastTarget::Int(320))), params[0])]
                },
            ),
        ),
    ];
    for (what, reason, ssa) in cases {
        let Err(refusal) =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Compiled::new(ssa)))
        else {
            panic!("{what} compiled");
        };
        let message = refusal
            .downcast_ref::<String>()
            .map_or("", String::as_str)
            .to_owned();
        assert!(
            message.starts_with("ICE: ") && message.contains(reason),
            "{what}: {message}"
        );
    }
}

/// In a signed `int1` the pattern `1` is `-1`, so `x * 1`, `1 * x` and `x / 1` are not `x`: at
/// `x = -1` each is `1`, which the width cannot hold, and the program is refused.
#[test]
fn a_one_bit_signed_product_or_quotient_by_the_pattern_one_is_not_folded() {
    use BinaryArithOpKind::{SDiv, SMul};
    let minus_one = IntBits::from_u128(1, 1);
    let zero = IntBits::zero(1);
    for (kind, constant_first) in [(SMul, false), (SMul, true), (SDiv, false)] {
        let ssa = main_program(&[Type::int(1)], &[Type::int(1)], move |e, params| {
            let one = e.int_const(IntBits::from_u128(1, 1));
            let (lhs, rhs) = if constant_first {
                (one, params[0])
            } else {
                (params[0], one)
            };
            vec![e.bin(kind, lhs, rhs)]
        });
        let what = format!("{kind:?}, constant first: {constant_first}");
        let compiled = Compiled::new(ssa).unwrap_or_else(|error| panic!("{what}: {error}"));
        let verdict = compiled.run(&input_block(&[&minus_one, &minus_one]));
        assert!(verdict.is_refusal(), "{what} at -1: {verdict:?}");
        let verdict = compiled.run(&input_block(&[&zero, &zero]));
        assert!(verdict.is_accepted(), "{what} at 0: {verdict:?}");
    }
}

/// `main(a: int(bits)) -> int(2 * bits) { spread(a) }`.
fn program_spreading(bits: usize) -> HLSSA {
    main_program(
        &[Type::int(bits)],
        &[Type::int(2 * bits)],
        move |e, params| vec![e.spread(params[0], bits)],
    )
}

/// `main(a: int(bits)) -> int(bits / 2) { unspread(a).odd }`, the inverse of a spread.
fn program_unspreading(bits: usize) -> HLSSA {
    main_program(
        &[Type::int(bits)],
        &[Type::int(bits / 2)],
        move |e, params| {
            let (odd, _even) = e.unspread(params[0], bits);
            vec![odd]
        },
    )
}

/// Each op takes its narrow opcode up to the widest container that opcode holds, and the wide one
/// from the next width on, which is what keeps the narrow lane's encoding for every spread the
/// bitwise lowering and the Noir surface make.
#[test]
fn a_spread_or_unspread_takes_the_wide_opcode_past_the_narrow_one() {
    for (program, narrow, wide) in [
        (
            program_spreading as fn(usize) -> HLSSA,
            (32, "spread_u32_to_u64"),
            (33, "spread_intn"),
        ),
        (
            program_unspreading,
            (64, "unspread_u64_to_u32"),
            (65, "unspread_intn"),
        ),
    ] {
        for ((bits, opcode), (_, other)) in [(narrow, wide), (wide, narrow)] {
            let listing = bytecode_listing(&program(bits));
            // Whole mnemonics, since `spread_intn` is a suffix of `unspread_intn`.
            let has = |mnemonic: &str| listing.split_whitespace().any(|word| word == mnemonic);
            assert!(has(opcode), "no {opcode} at {bits} in:\n{listing}");
            assert!(!has(other), "{other} at {bits} in:\n{listing}");
        }
    }
}

/// A witnessed unspread at an **odd** width agrees with the model on both lanes, the odd stream one
/// bit narrower than the even one.
///
/// The bitwise lowering never makes one, since it unspreads a sum it has cast to twice an operand's
/// width; the multi-cell representation does, on the ragged top limb of an operand held as limbs,
/// and this program makes one at the narrow lowering directly. The two streams are bounded by
/// lookups at two different widths, and what holds the lowering to them is that no column of the
/// witness can move without breaking a constraint.
#[test]
fn an_odd_width_unspread_agrees_with_the_model() {
    for bits in [3usize, 7, 33, 63] {
        let ssa = main_program(
            &[Type::int(bits)],
            &[Type::int(bits / 2), Type::int(bits.div_ceil(2))],
            move |e, params| {
                let (odd, even) = e.unspread(params[0], bits);
                vec![odd, even]
            },
        );
        let compiled = Compiled::new(ssa)
            .unwrap_or_else(|error| panic!("an int{bits} unspread is refused: {error}"));

        for value in corners::values(bits) {
            let value = IntBits::from_u128(bits, value);
            let (odd, even) = value.unspread(bits);
            let inputs = input_block(&[&value, &odd, &even]);
            let verdict = compiled.run(&inputs);
            assert!(verdict.is_accepted(), "int{bits} {value:?}: {verdict:?}");
            let verdict = compiled.run_wasm(&inputs).expect("the WASM lane builds");
            assert!(
                verdict.is_accepted(),
                "int{bits} {value:?} on WASM: {verdict:?}"
            );
        }

        let alternating = IntBits::from_u128(bits, 0x5555_5555_5555_5555 ^ 0x3);
        let (odd, even) = alternating.unspread(bits);
        assert_every_column_is_pinned(
            &format!("an int{bits} unspread"),
            &compiled,
            &[&alternating, &odd, &even],
        );
    }
}

/// A witnessed spread reads only the bits it names, and **rejects** a value with any above them.
///
/// The container is wider than the read, which is the shape the Noir surface gives every
/// `spread::<N>` of a `u32`, and the SHA-256 replacement leans on the rejection to range-check its
/// chunks. A pure spread discards those bits instead, which no witnessed one may do.
#[test]
fn a_witnessed_spread_rejects_the_bits_above_its_read() {
    let ssa = main_program(&[Type::int(16)], &[Type::int(32)], |e, params| {
        vec![e.spread(params[0], 8)]
    });
    let compiled = Compiled::new(ssa).expect("a spread of 8 bits of an int16 compiles");

    for value in [0u128, 1, 0x5A, 0xFF] {
        let value = IntBits::from_u128(16, value);
        let spread = value.spread(8);
        let inputs = input_block(&[&value, &spread]);
        let verdict = compiled.run(&inputs);
        assert!(verdict.is_accepted(), "{value:?}: {verdict:?}");
        let verdict = compiled.run_wasm(&inputs).expect("the WASM lane builds");
        assert!(verdict.is_accepted(), "{value:?} on WASM: {verdict:?}");
    }
    let value = IntBits::from_u128(16, 0x5A);
    assert_every_column_is_pinned(
        "a spread of 8 bits of an int16",
        &compiled,
        &[&value, &value.spread(8)],
    );

    // A value past the read has no witness, whatever answer is declared: not the masked spread the
    // model gives (which the VM computes, so only the constraints can refuse it), nor the spread of
    // the whole container.
    for value in [0x100u128, 0x1FF, 0xFFFF] {
        let value = IntBits::from_u128(16, value);
        for (declared, computed) in [(value.spread(8), true), (value.spread(16), false)] {
            let inputs = input_block(&[&value, &declared]);
            let refused = |verdict: &Verdict| {
                if computed {
                    refused_by_the_constraints(verdict)
                } else {
                    verdict.rejects()
                }
            };
            let verdict = compiled.run(&inputs);
            assert!(refused(&verdict), "{value:?} as {declared:?}: {verdict:?}");
            let verdict = compiled.run_wasm(&inputs).expect("the WASM lane builds");
            assert!(
                verdict.rejects(),
                "{value:?} as {declared:?} on WASM: {verdict:?}"
            );
        }
    }
}

/// A witnessed unspread rejects the bits above its read as the spread does, at an odd read inside a
/// wider container.
///
/// `unspread_64` in the SHA-256 replacement leans on this to range-check its limbs: each is an
/// unconstrained hint read through `unspread::<N>`, and only the lookups say it is a spread.
#[test]
fn a_witnessed_unspread_rejects_the_bits_above_its_read() {
    let ssa = main_program(
        &[Type::int(16)],
        &[Type::int(8), Type::int(8)],
        |e, params| {
            let (odd, even) = e.unspread(params[0], 7);
            vec![odd, even]
        },
    );
    let compiled = Compiled::new(ssa).expect("an unspread of 7 bits of an int16 compiles");

    for value in [0u128, 1, 0x2A, 0x55, 0x7F] {
        let value = IntBits::from_u128(16, value);
        let (odd, even) = value.unspread(7);
        let inputs = input_block(&[&value, &odd, &even]);
        let verdict = compiled.run(&inputs);
        assert!(verdict.is_accepted(), "{value:?}: {verdict:?}");
        let verdict = compiled.run_wasm(&inputs).expect("the WASM lane builds");
        assert!(verdict.is_accepted(), "{value:?} on WASM: {verdict:?}");
    }
    let value = IntBits::from_u128(16, 0x55);
    let (odd, even) = value.unspread(7);
    assert_every_column_is_pinned(
        "an unspread of 7 bits of an int16",
        &compiled,
        &[&value, &odd, &even],
    );

    // As with the spread: neither the masked streams, which are what the VM computes, nor the whole
    // container's has a witness.
    for value in [0x80u128, 0xAA, 0x100, 0xFFFF] {
        let value = IntBits::from_u128(16, value);
        for ((odd, even), computed) in [(value.unspread(7), true), (value.unspread(16), false)] {
            let inputs = input_block(&[&value, &odd, &even]);
            let refused = |verdict: &Verdict| {
                if computed {
                    refused_by_the_constraints(verdict)
                } else {
                    verdict.rejects()
                }
            };
            let verdict = compiled.run(&inputs);
            assert!(
                refused(&verdict),
                "{value:?} as {odd:?}, {even:?}: {verdict:?}"
            );
            let verdict = compiled.run_wasm(&inputs).expect("the WASM lane builds");
            assert!(
                verdict.rejects(),
                "{value:?} as {odd:?}, {even:?} on WASM: {verdict:?}"
            );
        }
    }
}

/// The widths of the 64-bit windows a `width`-wide value is read out of an entry point through.
fn window_widths(width: usize) -> Vec<usize> {
    (0..width.div_ceil(64))
        .map(|index| (width - 64 * index).min(64))
        .collect()
}

/// The 64-bit windows of a `width`-wide value, low first, as the values an entry point returns.
fn windows(e: &mut HLBlockEmitter<'_>, value: ValueId, width: usize) -> Vec<ValueId> {
    window_widths(width)
        .into_iter()
        .enumerate()
        .map(|(index, window)| {
            let shifted = if index == 0 {
                value
            } else {
                let amount = e.int_const(IntBits::from_u128(width, 64 * index as u128));
                e.bin(BinaryArithOpKind::UShr, value, amount)
            };
            e.cast_to(CastTarget::Int(window), shifted)
        })
        .collect()
}

/// The 64-bit windows of `value`'s pattern, as the input block declares them.
fn pattern_windows(value: &IntBits) -> Vec<IntBits> {
    window_widths(value.bits())
        .into_iter()
        .enumerate()
        .map(|(index, window)| value.bit_range(64 * index, window))
        .collect()
}

/// An `int(bits)` put together out of the 64-bit windows `windows` carries it in.
fn assembled(e: &mut HLBlockEmitter<'_>, windows: &[ValueId], bits: usize) -> ValueId {
    let mut x = e.cast_to(CastTarget::Int(bits), windows[0]);
    for (index, window) in windows.iter().enumerate().skip(1) {
        let widened = e.cast_to(CastTarget::Int(bits), *window);
        let amount = e.int_const(IntBits::from_u128(bits, 64 * index as u128));
        let placed = e.bin(BinaryArithOpKind::UShl, widened, amount);
        x = e.bin(BinaryArithOpKind::Or, x, placed);
    }
    x
}

/// `main` taking an `int(bits)` as 64-bit windows, assembling it, and returning the windows of
/// `spread(x, spread_read)` and of both streams of `unspread(x, unspread_read)`.
///
/// An entry point takes and returns only what one field element holds, so a value past the field
/// is assembled and taken apart inside the program, with the representation's `Or`, shifts and
/// narrowing casts.
fn program_spreading_witnessed(bits: usize, spread_read: usize, unspread_read: usize) -> HLSSA {
    let params: Vec<Type> = window_widths(bits).into_iter().map(Type::int).collect();
    let returns: Vec<Type> = [2 * bits, bits / 2, bits.div_ceil(2)]
        .into_iter()
        .flat_map(window_widths)
        .map(Type::int)
        .collect();
    main_program(&params, &returns, move |e, params| {
        let x = assembled(e, params, bits);
        let spread = e.spread(x, spread_read);
        let (odd, even) = e.unspread(x, unspread_read);
        let mut answers = windows(e, spread, 2 * bits);
        answers.extend(windows(e, odd, bits / 2));
        answers.extend(windows(e, even, bits.div_ceil(2)));
        answers
    })
}

/// The input block of a [`program_spreading_witnessed`]: `value`'s windows, then the windows of
/// the model's answers, which mask to each read.
fn spreading_inputs(value: &IntBits, spread_read: usize, unspread_read: usize) -> Vec<IntBits> {
    let (odd, even) = value.unspread(unspread_read);
    [value.clone(), value.spread(spread_read), odd, even]
        .iter()
        .flat_map(pattern_windows)
        .collect()
}

/// A witnessed spread and unspread agree with the model where the spread, the operand or a stream
/// is too wide for one field element, and every column of the witness is pinned.
///
/// Each width is a different shape:
/// - `127` spreads into the first width past the field from an operand that is one element;
/// - `254` is the first operand held as limbs, whose streams are one element each again;
/// - `601` has streams held as limbs too;
/// - the odd widths leave a ragged top limb and an even stream one bit wider than the odd one;
/// - `577`'s top limb is a single bit, which has no odd half.
///
/// Reads short of the width put a straddling half-limb (or, for an unspread, a straddling limb)
/// under a narrower lookup, a limb an unspread reads one bit of under a one-bit range check, and
/// limbs wholly above the read under a zero constraint. Short reads are covered on an operand that
/// is one element (`200`, read at 150 by each op) and on ones held as limbs: `300` spread at 250,
/// `600` unspread at 333, and `601` unspread at 513, which reads one bit of its top-but-one limb.
///
/// The widest is 601 because this program's scaffolding (the wide shifts that assemble the operand
/// and cut the answers into windows) stops the WASM engine loading the module from about a thousand
/// bits; the tests in `wide_witness_ints` take the lowering to the widest widths.
#[test]
fn a_witnessed_spread_and_unspread_agree_with_the_model_past_the_field() {
    for (bits, spread_read, unspread_read) in [
        (127usize, 127usize, 127usize),
        (200, 150, 200),
        (200, 200, 150),
        (253, 253, 253),
        (254, 254, 254),
        (300, 299, 300),
        (300, 250, 300),
        (577, 577, 577),
        (600, 600, 333),
        (601, 601, 513),
    ] {
        let compiled = Compiled::new(program_spreading_witnessed(
            bits,
            spread_read,
            unspread_read,
        ))
        .unwrap_or_else(|error| panic!("an int{bits} spread and unspread are refused: {error}"));
        let read = spread_read.min(unspread_read);
        let every_other = ((BigUint::from(1u8) << read) - 1u8) / 3u8;
        let values = [
            IntBits::all_ones(read),
            IntBits::from_biguint(read, &every_other),
            IntBits::one(read).shifted_left(read - 1),
            IntBits::from_limbs(read, &[0x0123_4567_89AB_CDEF; 16]),
        ];
        for value in values {
            let value = value.cast(bits);
            let block = spreading_inputs(&value, spread_read, unspread_read);
            let inputs = input_block(&block.iter().collect::<Vec<_>>());
            let verdict = compiled.run(&inputs);
            assert!(verdict.is_accepted(), "int{bits} {value:?}: {verdict:?}");
            let verdict = compiled.run_wasm(&inputs).expect("the WASM lane builds");
            assert!(
                verdict.is_accepted(),
                "int{bits} {value:?} on WASM: {verdict:?}"
            );
        }

        {
            let value = IntBits::from_biguint(read, &every_other).cast(bits);
            let block = spreading_inputs(&value, spread_read, unspread_read);
            assert_every_column_is_pinned(
                &format!("an int{bits} spread and unspread"),
                &compiled,
                &block.iter().collect::<Vec<_>>(),
            );
        }
    }
}

/// A witnessed spread or unspread past the field rejects a bit above its read, as the narrow ones
/// do, whatever answer is declared: the model's, which discards the bit and is what the VM
/// computes, so only the constraints can refuse it, or the whole container's.
#[test]
fn a_witnessed_spread_or_unspread_past_the_field_rejects_the_bits_above_its_read() {
    // `(width, spread read, unspread read, the bit set past the shorter read)`: a bit in the
    // half-limb the read straddles, and one in a limb wholly above it, for each op, and two in a limb
    // an unspread reads a single bit of, the second past the half-limb that limb's even stream
    // keeps, where reading the limb at that half's width would drop it.
    for (bits, spread_read, unspread_read, bit) in [
        (300usize, 250usize, 300usize, 251usize),
        (300, 250, 300, 290),
        (600, 600, 333, 335),
        (600, 600, 333, 500),
        (600, 600, 513, 514),
        (600, 600, 513, 550),
    ] {
        let compiled = Compiled::new(program_spreading_witnessed(
            bits,
            spread_read,
            unspread_read,
        ))
        .unwrap_or_else(|error| panic!("an int{bits} spread and unspread are refused: {error}"));
        let value = IntBits::from_u128(bits, 0xF0F0).or(&IntBits::one(bits).shifted_left(bit));

        // The model's masked answers are the ones the VM computes, so it is the constraints that
        // have to refuse them; the whole container's answers have no witness either.
        let computed = spreading_inputs(&value, spread_read, unspread_read);
        let whole = spreading_inputs(&value, bits, bits);
        for (block, by_constraints) in [(computed, true), (whole, false)] {
            let refused = |verdict: &Verdict| {
                if by_constraints {
                    refused_by_the_constraints(verdict)
                } else {
                    verdict.rejects()
                }
            };
            let inputs = input_block(&block.iter().collect::<Vec<_>>());
            let verdict = compiled.run(&inputs);
            assert!(
                refused(&verdict),
                "int{bits} with bit {bit} set: {verdict:?}"
            );
            let verdict = compiled.run_wasm(&inputs).expect("the WASM lane builds");
            assert!(
                verdict.rejects(),
                "int{bits} with bit {bit} set on WASM: {verdict:?}"
            );
        }
    }
}

/// A limb an unspread reads one bit of is held to that bit, not taken whole as the even stream's
/// half-limb.
///
/// The declared answer is what an unbounded limb computes, as the VM does, so the constraints must
/// refuse it. A read stopping at bit 512 holds that bit and nothing above; this one also carries
/// the limb's bit 513, as bit 257 of the even stream.
#[test]
fn a_limb_an_unspread_reads_one_bit_of_is_held_to_that_bit() {
    let (bits, read) = (600usize, 513usize);
    let compiled = Compiled::new(program_spreading_witnessed(bits, bits, read))
        .unwrap_or_else(|error| panic!("an int{bits} spread and unspread are refused: {error}"));
    let value = IntBits::from_u128(bits, 0b11).shifted_left(512);
    let (odd, even) = value.unspread(read);
    let forged = even.or(&IntBits::one(even.bits()).shifted_left(257));
    let block: Vec<IntBits> = [value.clone(), value.spread(bits), odd, forged]
        .iter()
        .flat_map(pattern_windows)
        .collect();
    let inputs = input_block(&block.iter().collect::<Vec<_>>());
    let verdict = compiled.run(&inputs);
    assert!(refused_by_the_constraints(&verdict), "{verdict:?}");
    let verdict = compiled.run_wasm(&inputs).expect("the WASM lane builds");
    assert!(verdict.rejects(), "on WASM: {verdict:?}");
}

/// `main(taken: u1, x)` answering the windows of `spread(x, read)`, or of both streams of
/// `unspread(x, read)`, computed in a branch on `taken`, and of zeros where it is not taken.
///
/// Only the spread or unspread is in the branch; the answers are cut into windows after the merge,
/// so that nothing else in the program is under the guard.
fn program_spreading_under_a_branch(bits: usize, read: usize, unspread: bool) -> HLSSA {
    let answer_widths: Vec<usize> = if unspread {
        vec![bits / 2, bits.div_ceil(2)]
    } else {
        vec![2 * bits]
    };
    let mut params = vec![Type::int(1)];
    params.extend(window_widths(bits).into_iter().map(Type::int));
    let returns: Vec<Type> = answer_widths
        .iter()
        .flat_map(|width| window_widths(*width))
        .map(Type::int)
        .collect();
    main_program(&params, &returns, move |e, params| {
        let x = assembled(e, &params[1..], bits);
        let (then_id, _) = e.add_block();
        let (else_id, _) = e.add_block();
        let (merge_id, _) = e.add_block();
        e.seal_and_switch(Terminator::JmpIf(params[0], then_id, else_id), then_id);
        let answers = if unspread {
            let (odd, even) = e.unspread(x, read);
            vec![odd, even]
        } else {
            vec![e.spread(x, read)]
        };
        e.seal_and_switch(Terminator::Jmp(merge_id, answers), else_id);
        let zeros = answer_widths
            .iter()
            .map(|width| e.int_const(IntBits::zero(*width)))
            .collect();
        e.seal_and_switch(Terminator::Jmp(merge_id, zeros), merge_id);
        let merged: Vec<ValueId> = answer_widths
            .iter()
            .map(|width| e.add_parameter(Type::int(*width)))
            .collect();
        merged
            .into_iter()
            .zip(&answer_widths)
            .flat_map(|(answer, width)| windows(e, answer, *width))
            .collect()
    })
}

/// A witnessed spread or unspread in a branch rejects a bit above its read only where the branch is
/// taken, on the narrow lowering and on the representation's: its lookups, zero checks and one-bit
/// checks hold only under its guard, as a guarded range check's do.
#[test]
fn a_guarded_witnessed_spread_rejects_the_bits_above_its_read_only_where_taken() {
    // `(width, read, unspread)`: each op on each lowering, at the narrow one both with a read a
    // single lookup table takes and with one it splits into chunks, and the unspread of a limb the
    // read takes one bit of, whose top bit is past the half-limb its even stream keeps at `560`.
    for (bits, read, unspread) in [
        (16usize, 8usize, false),
        (16, 7, true),
        (64, 30, false),
        (64, 61, true),
        (300, 250, false),
        (600, 333, true),
        (600, 513, true),
        (560, 513, true),
    ] {
        let what = format!(
            "a guarded int{bits} {} of {read} bits",
            if unspread { "unspread" } else { "spread" }
        );
        let compiled = Compiled::new(program_spreading_under_a_branch(bits, read, unspread))
            .unwrap_or_else(|error| panic!("{what} does not compile: {error}"));
        let answers = |value: &IntBits| -> Vec<IntBits> {
            if unspread {
                let (odd, even) = value.unspread(read);
                [odd, even].iter().flat_map(pattern_windows).collect()
            } else {
                pattern_windows(&value.spread(read))
            }
        };
        let run = |taken: u128, value: &IntBits, answers: Vec<IntBits>| {
            let mut block = vec![IntBits::from_u128(1, taken)];
            block.extend(pattern_windows(value));
            block.extend(answers);
            let inputs = input_block(&block.iter().collect::<Vec<_>>());
            let wasm = compiled.run_wasm(&inputs).expect("the WASM lane builds");
            (compiled.run(&inputs), wasm)
        };

        let clean = IntBits::all_ones(read).cast(bits);
        // A bit just past the read, which a straddled piece holds, and the top bit, which a piece
        // or limb wholly above the read holds.
        let dirty = clean
            .or(&IntBits::one(bits).shifted_left(read))
            .or(&IntBits::one(bits).shifted_left(bits - 1));
        let zeros = |value: &IntBits| {
            answers(value)
                .into_iter()
                .map(|window| IntBits::zero(window.bits()))
                .collect::<Vec<_>>()
        };
        for (taken, value, declared, accepted) in [
            (1, &clean, answers(&clean), true),
            (0, &clean, zeros(&clean), true),
            (0, &dirty, zeros(&dirty), true),
            (1, &dirty, answers(&dirty), false),
        ] {
            let (vm, wasm) = run(taken, value, declared);
            for (lane, verdict) in [("VM", vm), ("WASM", wasm)] {
                let holds = if accepted {
                    verdict.is_accepted()
                } else if lane == "VM" {
                    refused_by_the_constraints(&verdict)
                } else {
                    verdict.rejects()
                };
                assert!(
                    holds,
                    "{what}, taken {taken}, {value:?} on {lane}: {verdict:?}"
                );
            }
        }
    }
}

// THE FIELD BOUNDARY
// ================================================================================================

/// The first width whose values are not all distinct field elements.
///
/// `2^bits <= p` makes an `Int -> Field` cast injective, and writing the modulus takes exactly this
/// many bits, so this is the first width at which two integers share an element.
///
/// Derived from the field's bit size rather than the compiler's `widest_injective_int_bits`, so
/// that the test checks they agree.
fn first_width_past_the_field() -> usize {
    FieldConfig::bn254().field_bit_size() as usize
}

/// A cast at the widest width the field _does_ carry is admitted.
#[test]
fn the_widest_injective_width_casts_to_field() {
    let ssa = main_program(
        &[Type::int(first_width_past_the_field() - 1)],
        &[Type::field()],
        |e, params| vec![e.cast_to_field(params[0])],
    );

    assert!(
        first_phase_only(ssa).is_ok(),
        "the widest injective width was refused"
    );
}

/// Every integer parameter of `main` is range-checked to its width where it is written to a
/// witness column, so a wide entry point meets that decomposition first.
///
/// The check is on a **field element**, which is why it reaches this far: the widest integer the
/// modulus carries injectively is a parameter like any other, and `main` needs no wide calling
/// convention to take one.
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

/// An integer wider than one input element carries cannot cross this ABI directly. The frontend
/// rejects such parameters; a hand-built SSA that bypasses it must also be refused.
#[test]
#[should_panic(expected = "crosses the entry point")]
fn a_main_parameter_wider_than_the_field_is_a_compiler_error() {
    let ssa = main_program(
        &[Type::int(first_width_past_the_field())],
        &[Type::int(64)],
        |e, params| vec![e.cast_to(CastTarget::Int(64), params[0])],
    );
    let _ = Compiled::new(ssa);
}

/// The entry point's range check bites: a value above the declared width is refused when proving.
///
/// An honest witness satisfies a decomposition that drops its top chunks as well as one that does
/// not, so only a **dishonest** input shows the chunks are all there, which
/// `the_widest_injective_width_is_an_entry_point_parameter` cannot.
#[test]
fn a_wide_entry_point_parameter_is_held_to_its_width() {
    for width in [129usize, 200] {
        let ssa = main_program(&[Type::int(width)], &[Type::field()], |e, params| {
            vec![e.cast_to_field(params[0])]
        });
        let compiled = Compiled::new(ssa).expect("a wide parameter compiles");

        // One bit above the declared width: only the entry point's range check stands between it
        // and the rest of the program.
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

/// A range check wider than the field carries injectively holds for every element, so it is
/// dropped rather than decomposed: the largest element passes it.
#[test]
fn a_range_check_past_what_the_field_carries_always_holds() {
    for bits in [first_width_past_the_field(), 300] {
        let ssa = main_program(&[Type::field()], &[Type::field()], |e, params| {
            e.rangecheck(params[0], bits);
            vec![params[0]]
        });
        let compiled = Compiled::new(ssa)
            .unwrap_or_else(|error| panic!("a {bits}-bit range check compiles: {error}"));
        let largest = -Field::from(1u64);
        let verdict = compiled.run(&[
            InputValueOrdered::Field(largest),
            InputValueOrdered::Field(largest),
        ]);
        assert!(verdict.is_accepted(), "{bits} bits: {verdict:?}");
    }
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
/// A mixed pure/witness pair arrives as one value beside a five-limb one. Zipping the two lists
/// would compare only the low limbs, which is right whenever they decide the answer, as in every
/// honest sweep.
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
/// This is what the representation gives that a single element cannot: `2^299` has no field
/// element (two integers that far apart share one), but each of its limbs does, so a constant at
/// such a width splits into limb constants. `hlssa_to_r1cs::of_int` refuses a magnitude at or above
/// the modulus, which a single-element constant of this width would hit.
///
/// 254 is the first width the modulus does not carry, and 300 is well past it; both answer.
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
/// An honest witness cannot stand in for this test. A limb the reconstruction constraint does not
/// reach is free, and no run with correct inputs shows it, since the prover never uses the freedom.
/// Adding one to each column in turn finds it: `2^h` is invertible mod `p`, so an unpinned limb
/// lets a prover solve for any value.
///
/// **The source is wider than one limb and the result narrower**, which is what makes the test
/// bite. An `int64` source has one limb, and a result that reads every limb pins them through its
/// assertion whether or not the decomposition constrains anything; either shape passes with the
/// reconstruction removed.
///
/// It contains no equality. `lower_eq` witnesses the inverse of the difference as a hint, which is
/// free whenever the difference is zero, so an honest `a == a` has an unconstrained column at every
/// width, with or without this representation.
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
/// The scalar test above shows the representation constrains its limbs; this shows the same for a
/// sequence. Entering one mints `k` witness columns **per element**, each with a separate range
/// check, and reading one back pins `k` lookups rather than a decomposition.
///
/// The element comes from a **parameter**, so the columns exist, and the sequence is read at a
/// **witness** index, so the lookups are real. The declared return is narrower than the element, so
/// the program does not pin the high limbs through its assertion.
///
/// **What this cannot show:** a column nothing pins is also a column nothing reads, so it is
/// eliminated rather than left free. Delete one of the `k` lookups and the witness comes back
/// shorter, with every remaining column pinned. A pinning test only finds a free column that
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
/// The index is a parameter, so the lookup cannot be folded away, and a witness, so the tape has to
/// address the lookup.
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
/// Above the injective width an entry fits in no single field element, so the sequence is stored
/// as one sequence per limb, and each limb is an entry the tape can already read. 254 is the first
/// such width; 320 differs from it in both limb layout and limb count, so the two exercise
/// different shapes.
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
/// One cell is `ELEM_WORD`, two are `ELEM_U128`, and from 129 bits up to the widest injective width
/// an element is one field element spanning more cells, `ELEM_CELLS`; wider sequences are
/// transposed. So only `129..=253` selects it.
///
/// The element is compared **whole**, not narrowed to its low limb, since a reader that drops every
/// high cell answers a narrowing return correctly: without this, `read_cells_as_field` reading one
/// cell instead of `cells` leaves the whole workspace green.
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
/// `Cast{ArrayToSlice}` is the one slice-family opcode that reaches this pass without a separate
/// sequence test: `SlicePop`, `SliceInsert` and `SliceRemove` are lowered before it runs, and
/// `SlicePush`/`SliceLen` are covered above.
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
    // `MkRepeated`: every slot is the same wide element, derived from a parameter so that it is
    // witnessed.
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

    // A slice, which reaches its elements through the slice opcodes rather than the tape. Its
    // elements are **constants**, the shape that folds if a read reassembles a whole wide value
    // instead of keeping its limbs.
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
/// The slice family does not involve the tape: `SlicePush` and `SliceLen` move limbs themselves,
/// and `SliceLen` reads the length from **one** limb sequence. That is correct only because every
/// length-changing operation is applied identically to all of them; a push that reached some limb
/// sequences and not others would leave them ragged, and the length would depend on which one was
/// asked.
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
/// The corpus covers source programs at standard and extended integer widths. This hand-built
/// SSA test checks the wide field boundary and the multi-cell representation against the interpreter
/// and the constraint system.
///
/// Skipped only where the linker reported success and wrote no module; see
/// `harness::compile_wasm`.
#[test]
fn the_wasm_lane_agrees_at_a_wide_width() {
    let injective = first_width_past_the_field() - 1;
    let mut ran = 0usize;

    // The field boundary, up to the widest width one element carries, with the value in the top
    // limb. Past 128 bits the element spans more cells than the narrow lanes hold.
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

/// A spread whose answer nothing reads is still a range check on its operand, in `main` and in a
/// function `main` calls.
///
/// A witnessed spread rejects an operand with bits above its read, and that lookup is the only
/// bound the operand has. Before witness typing only `main`'s parameters are witnesses, so a
/// callee's spread has a pure operand when the first dead-code runs see it, and deleting it there
/// would leave the operand unchecked once the callee is typed. `width` is the operand, read at 8
/// bits: 255 fits and 256 does not, and 600 is a wide operand held as limbs.
#[test]
fn a_dead_spread_still_rejects_the_bits_above_its_read() {
    for width in [16usize, 600] {
        for in_callee in [false, true] {
            let mut ssa = HLSSA::with_main("main".to_string());
            let main_id = ssa.get_unique_entrypoint_id();
            let mut builder = HLSSABuilder::new(&mut ssa);
            let (callee_id, ()) = builder.add_function("checks".to_string(), |b| {
                let entry = b.function.get_entry_id();
                let mut e = b
                    .block(entry)
                    .with_source_location(SourceLocation::synthetic("checks"));
                let value = e.add_parameter(Type::int(width));
                let _ = e.spread(value, 8);
                e.terminate_return(vec![]);
            });
            builder.modify_function(main_id, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b
                    .block(entry)
                    .with_source_location(SourceLocation::synthetic("main"));
                let param = e.add_parameter(Type::int(16));
                let operand = e.cast_to(CastTarget::Int(width), param);
                if in_callee {
                    let _ = e.call(callee_id, vec![operand], 0);
                } else {
                    let _ = e.spread(operand, 8);
                }
                e.terminate_return(vec![]);
            });
            let compiled = Compiled::new(ssa).unwrap_or_else(|error| {
                panic!("int{width}, in a callee: {in_callee}, does not compile: {error}")
            });
            let input = |value: u128| input_block(&[&IntBits::from_u128(16, value)]);
            let verdict = compiled.run(&input(255));
            assert!(
                verdict.is_accepted(),
                "int{width}, in a callee: {in_callee}, refused an operand that fits: {verdict:?}"
            );
            let verdict = compiled.run(&input(256));
            assert!(
                refused_by_the_constraints(&verdict),
                "int{width}, in a callee: {in_callee}, accepted a bit above the read: {verdict:?}"
            );
        }
    }
}

// SHAPES ONLY THE RUST API STATES
// ================================================================================================

/// Emits a loop that counts to 2 and switches to the block after it, returning the count: an
/// `int32` no fold answers.
fn loop_count(e: &mut HLBlockEmitter<'_>) -> ValueId {
    let zero = e.int_const(IntBits::zero(32));
    let (head, _) = e.add_block();
    let (back, _) = e.add_block();
    let (done, _) = e.add_block();
    e.seal_and_switch(Terminator::Jmp(head, vec![zero]), head);
    let count = e.add_parameter(Type::int(32));
    let step = e.int_const(IntBits::from_u128(32, 1));
    let count = e.bin(BinaryArithOpKind::UAdd, count, step);
    let limit = e.int_const(IntBits::from_u128(32, 2));
    let more = e.cmp(count, limit, CmpKind::ULt);
    e.seal_and_switch(Terminator::JmpIf(more, back, done), back);
    e.seal_and_switch(Terminator::Jmp(head, vec![count]), done);
    count
}

/// Three pure `int320` elements, the last with its top bit set so that it is negative under a
/// signed reading.
fn pure_wide_elements() -> Vec<IntBits> {
    let bits = 320;
    let low = |seed: u32| BigUint::from(seed).pow(150u32) % (BigUint::from(1u8) << (bits - 2));
    vec![
        IntBits::from_biguint(bits, &low(3)),
        IntBits::from_biguint(bits, &low(7)),
        IntBits::from_biguint(bits, &(low(11) | (BigUint::from(1u8) << (bits - 1)))),
    ]
}

/// `main(w: int64) -> returns { body(elements[n], w) }` for a pure `[int320; 3]`, where `n` is the
/// count a loop leaves behind, 2, which no fold answers.
fn reading_a_pure_wide_element(
    returns: Vec<Type>,
    body: impl FnOnce(&mut HLBlockEmitter<'_>, ValueId, ValueId) -> Vec<ValueId>,
) -> HLSSA {
    let elements = pure_wide_elements();
    main_program(&[Type::int(64)], &returns, move |e, params| {
        let constants: Vec<ValueId> = elements.iter().map(|v| e.int_const(v.clone())).collect();
        let array = e.mk_seq(constants, SequenceTargetType::Array(3), Type::int(320));
        let count = loop_count(e);
        let element = e.array_get(array, count);
        body(e, element, params[0])
    })
}

/// A pure wide element read out of a sequence reaches every kind of consumer, on both lanes.
///
/// A sequence of wide elements is held one sequence per limb position, pure elements included, and
/// a pure element is put back together as it is read. So every consumer meets it whole: a pure
/// one, a witnessed gadget beside a witness operand, a select or a sequence that mixes it with a
/// witness, a merge of two branches, a loop's accumulator and a call. The element is negative under
/// a signed reading, so a sign extension fills ones. Every answer is an `int320` and leaves as its
/// top limb, past every limb a bind could have kept by accident, and a wrong one is refused.
#[test]
fn a_pure_wide_element_read_out_of_a_sequence_reaches_every_consumer() {
    use BinaryArithOpKind::{SDiv, SMul, SRem, UDiv, UMul, URem, UShl, UShr, Xor};
    let element = pure_wide_elements()[2].clone();
    let other = IntBits::from_biguint(320, &BigUint::from(5u8).pow(60u32));
    let input = IntBits::from_u128(64, 12345);
    let witnessed = input.cast(320);
    let model = |op: IntOp, lhs: &IntBits, rhs: &IntBits| {
        mavros_int_semantics::eval(op, lhs, rhs)
            .value()
            .unwrap_or_else(|| panic!("{op:?} fits"))
    };
    type Body = Box<dyn FnOnce(&mut HLBlockEmitter<'_>, ValueId, ValueId, ValueId) -> ValueId>;
    let mut shapes: Vec<(String, IntBits, Body)> = vec![
        (
            "xor".into(),
            element.xor(&other),
            Box::new(|e, x, o, _| e.bin(Xor, x, o)),
        ),
        (
            "not".into(),
            element.complement(),
            Box::new(|e, x, _, _| e.not(x)),
        ),
        (
            "a sign extension".into(),
            element.sign_extend(400).shifted_right(80).cast(320),
            Box::new(|e, x, _, _| {
                let extended = e.sext(x, 320, 400);
                let amount = e.int_const(IntBits::from_u128(400, 80));
                let shifted = e.bin(UShr, extended, amount);
                e.cast_to(CastTarget::Int(320), shifted)
            }),
        ),
        (
            "a widening".into(),
            element.cast(400).shifted_right(80).cast(320),
            Box::new(|e, x, _, _| {
                let widened = e.cast_to(CastTarget::Int(400), x);
                let amount = e.int_const(IntBits::from_u128(400, 80));
                let shifted = e.bin(UShr, widened, amount);
                e.cast_to(CastTarget::Int(320), shifted)
            }),
        ),
        (
            "a select on a witness, against a witness".into(),
            element.clone(),
            Box::new(|e, x, _, w| {
                let zero = e.int_const(IntBits::zero(64));
                let chosen = e.cmp(zero, w, CmpKind::ULt);
                let wide = e.cast_to(CastTarget::Int(320), w);
                e.select(chosen, x, wide)
            }),
        ),
        (
            "a sequence beside a witness".into(),
            element.clone(),
            Box::new(|e, x, _, w| {
                let wide = e.cast_to(CastTarget::Int(320), w);
                let both = e.mk_seq(vec![x, wide], SequenceTargetType::Array(2), Type::int(320));
                let zero = e.int_const(IntBits::zero(64));
                let past = e.cmp(w, zero, CmpKind::ULt);
                let index = e.cast_to(CastTarget::Int(32), past);
                e.array_get(both, index)
            }),
        ),
        (
            "a merge of two branches".into(),
            element.clone(),
            Box::new(|e, x, o, _| {
                let pick = e.cmp(o, x, CmpKind::ULt);
                let (then, _) = e.add_block();
                let (otherwise, _) = e.add_block();
                let (merge, _) = e.add_block();
                e.seal_and_switch(Terminator::JmpIf(pick, then, otherwise), then);
                e.seal_and_switch(Terminator::Jmp(merge, vec![x]), otherwise);
                e.seal_and_switch(Terminator::Jmp(merge, vec![o]), merge);
                e.add_parameter(Type::int(320))
            }),
        ),
        (
            "a loop's accumulator".into(),
            element.complement(),
            Box::new(|e, x, _, _| {
                let zero = e.int_const(IntBits::zero(32));
                let (head, _) = e.add_block();
                let (back, _) = e.add_block();
                let (done, _) = e.add_block();
                e.seal_and_switch(Terminator::Jmp(head, vec![zero, x]), head);
                let count = e.add_parameter(Type::int(32));
                let accumulated = e.add_parameter(Type::int(320));
                let step = e.int_const(IntBits::from_u128(32, 1));
                let count = e.bin(BinaryArithOpKind::UAdd, count, step);
                let limit = e.int_const(IntBits::from_u128(32, 2));
                let more = e.cmp(count, limit, CmpKind::ULt);
                e.seal_and_switch(Terminator::JmpIf(more, back, done), back);
                let flipped = e.not(accumulated);
                e.seal_and_switch(Terminator::Jmp(head, vec![count, flipped]), done);
                accumulated
            }),
        ),
        (
            "a shift of a witness by it".into(),
            witnessed.shifted_left(
                usize::try_from(&element.shifted_right(312)).expect("eight bits fit"),
            ),
            // By the element's top eight bits, at least 128 as its top bit is set and below the
            // width.
            Box::new(|e, x, _, w| {
                let wide = e.cast_to(CastTarget::Int(320), w);
                let top = e.int_const(IntBits::from_u128(320, 312));
                let amount = e.bin(UShr, x, top);
                e.bin(UShl, wide, amount)
            }),
        ),
    ];
    for kind in [UDiv, URem, SDiv, SRem, UMul, SMul] {
        let (lhs, rhs) = if matches!(kind, UMul | SMul) {
            (element.clone(), IntBits::one(320))
        } else {
            (element.clone(), witnessed.clone())
        };
        let want = model(IntOp::from(kind), &lhs, &rhs);
        let one = matches!(kind, UMul | SMul);
        shapes.push((
            format!("{kind:?} by a witness"),
            want,
            Box::new(move |e, x, _, w| {
                let wide = e.cast_to(CastTarget::Int(320), w);
                let rhs = if one {
                    let zero = e.int_const(IntBits::zero(320));
                    let flag = e.cmp(zero, wide, CmpKind::ULt);
                    e.cast_to(CastTarget::Int(320), flag)
                } else {
                    wide
                };
                e.bin(kind, x, rhs)
            }),
        ));
    }
    for (what, want, body) in shapes {
        let other = other.clone();
        let ssa = reading_a_pure_wide_element(vec![Type::int(64)], move |e, x, w| {
            let other = e.int_const(other);
            let answer = body(e, x, other, w);
            let amount = e.int_const(IntBits::from_u128(320, 256));
            let top = e.bin(UShr, answer, amount);
            vec![e.cast_to(CastTarget::Int(64), top)]
        });
        let want = want.shifted_right(256).cast(64);
        let compiled =
            Compiled::new(ssa).unwrap_or_else(|error| panic!("{what} does not compile: {error}"));
        let verdict = compiled.run(&input_block(&[&input, &want]));
        assert!(verdict.is_accepted(), "{what}: {verdict:?}");
        let verdict = compiled
            .run_wasm(&input_block(&[&input, &want]))
            .expect("the WASM lane builds");
        assert!(verdict.is_accepted(), "{what} on WASM: {verdict:?}");
        let wrong = want.xor(&IntBits::one(64));
        assert!(
            !compiled.run(&input_block(&[&input, &wrong])).is_accepted(),
            "{what} accepted a wrong answer"
        );
    }
}

/// A pure wide equality asserted inside a witness branch holds only where the branch is taken.
///
/// The operands have no field image, so they are compared as integers and the answer is asserted
/// under the guard.
#[test]
fn a_pure_wide_equality_asserted_in_a_witness_branch_binds_only_where_taken() {
    for (index, holds) in [(2usize, true), (1, false)] {
        let rhs = pure_wide_elements()[index].clone();
        let ssa = reading_a_pure_wide_element(vec![], move |e, x, w| {
            let rhs = e.int_const(rhs);
            let zero = e.int_const(IntBits::zero(64));
            let taken = e.cmp(zero, w, CmpKind::ULt);
            let (then, _) = e.add_block();
            let (merge, _) = e.add_block();
            e.seal_and_switch(Terminator::JmpIf(taken, then, merge), then);
            e.assert_cmp(CmpKind::Eq, x, rhs);
            e.seal_and_switch(Terminator::Jmp(merge, vec![]), merge);
            vec![]
        });
        let compiled = Compiled::new(ssa).unwrap_or_else(|error| panic!("{index}: {error}"));
        let not_taken = input_block(&[&IntBits::zero(64)]);
        let taken = input_block(&[&IntBits::from_u128(64, 1)]);
        assert!(compiled.run(&not_taken).is_accepted(), "{index} not taken");
        let verdict = compiled.run(&taken);
        assert_eq!(verdict.is_accepted(), holds, "{index} taken: {verdict:?}");
        let verdict = compiled.run_wasm(&taken).expect("the WASM lane builds");
        assert_eq!(
            verdict.is_accepted(),
            holds,
            "{index} taken on WASM: {verdict:?}"
        );
    }
}

/// `main(w: int64) -> int64 { callee(elements[n], w) }`, where `callee(x: int320, w: int64)` answers
/// the top limb of `x ^ (w as int320)`, called as a constrained or an unconstrained function.
fn calling_with_a_pure_wide_element(unconstrained: bool) -> HLSSA {
    let elements = pure_wide_elements();
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let mut builder = HLSSABuilder::new(&mut ssa);
    let (callee, ()) = builder.add_function("callee".to_string(), |b| {
        b.function.add_return_type(Type::int(64));
        let entry = b.function.get_entry_id();
        let mut e = b
            .block(entry)
            .with_source_location(SourceLocation::synthetic("callee"));
        let x = e.add_parameter(Type::int(320));
        let w = e.add_parameter(Type::int(64));
        let wide = e.cast_to(CastTarget::Int(320), w);
        let mixed = e.bin(BinaryArithOpKind::Xor, x, wide);
        let amount = e.int_const(IntBits::from_u128(320, 256));
        let top = e.bin(BinaryArithOpKind::UShr, mixed, amount);
        let top = e.cast_to(CastTarget::Int(64), top);
        e.terminate_return(vec![top]);
    });
    builder.modify_function(main_id, |b| {
        b.function.add_return_type(Type::int(64));
        let entry = b.function.get_entry_id();
        let mut e = b
            .block(entry)
            .with_source_location(SourceLocation::synthetic("main"));
        let w = e.add_parameter(Type::int(64));
        let constants: Vec<ValueId> = elements.iter().map(|v| e.int_const(v.clone())).collect();
        let array = e.mk_seq(constants, SequenceTargetType::Array(3), Type::int(320));
        let count = loop_count(&mut e);
        let element = e.array_get(array, count);
        let answer = if unconstrained {
            e.call_unconstrained(callee, vec![element, w], 1)
        } else {
            e.call(callee, vec![element, w], 1)
        };
        e.terminate_return(answer);
    });
    ssa
}

/// A pure wide element is handed to a function whole, beside a witness or as an unconstrained
/// call's argument, on both lanes, and a wrong answer is refused.
#[test]
fn a_pure_wide_element_is_handed_to_a_call_whole() {
    let element = pure_wide_elements()[2].clone();
    let input = IntBits::from_u128(64, 12345);
    let want = element.xor(&input.cast(320)).shifted_right(256).cast(64);
    for unconstrained in [false, true] {
        let compiled = Compiled::new(calling_with_a_pure_wide_element(unconstrained))
            .unwrap_or_else(|error| panic!("unconstrained: {unconstrained}: {error}"));
        let verdict = compiled.run(&input_block(&[&input, &want]));
        assert!(
            verdict.is_accepted(),
            "unconstrained: {unconstrained}: {verdict:?}"
        );
        let verdict = compiled
            .run_wasm(&input_block(&[&input, &want]))
            .expect("the WASM lane builds");
        assert!(
            verdict.is_accepted(),
            "unconstrained: {unconstrained} on WASM: {verdict:?}"
        );
        let wrong = want.xor(&IntBits::one(64));
        assert!(
            !compiled.run(&input_block(&[&input, &wrong])).is_accepted(),
            "unconstrained: {unconstrained} accepted a wrong answer"
        );
    }
}

/// A witness-indexed read of a wide sequence inside a branch on a witness reads the element where
/// the branch is taken, and an index past the end is refused there and nowhere else.
#[test]
fn a_witness_indexed_wide_read_in_a_witness_branch_binds_only_where_taken() {
    let high = IntBits::from_biguint(320, &((BigUint::from(1u8) << 312u32) + BigUint::from(7u8)));
    let ssa = main_program(
        &[Type::int(64), Type::int(32)],
        &[Type::int(64)],
        move |e, params| {
            let wide = e.cast_to(CastTarget::Int(320), params[0]);
            let high = e.int_const(high);
            let array = e.mk_seq(
                vec![wide, high],
                SequenceTargetType::Array(2),
                Type::int(320),
            );
            let zero = e.int_const(IntBits::zero(64));
            let taken = e.cmp(zero, params[0], CmpKind::ULt);
            let (then, _) = e.add_block();
            let (skip, _) = e.add_block();
            let (merge, _) = e.add_block();
            e.seal_and_switch(Terminator::JmpIf(taken, then, skip), then);
            let element = e.array_get(array, params[1]);
            let amount = e.int_const(IntBits::from_u128(320, 256));
            let top = e.bin(BinaryArithOpKind::UShr, element, amount);
            let top = e.cast_to(CastTarget::Int(64), top);
            e.seal_and_switch(Terminator::Jmp(merge, vec![top]), skip);
            e.seal_and_switch(Terminator::Jmp(merge, vec![zero]), merge);
            vec![e.add_parameter(Type::int(64))]
        },
    );
    let compiled = Compiled::new(ssa).expect("a guarded wide read compiles");
    let run = |w: u128, index: u128, answer: u128| {
        compiled.run(&input_block(&[
            &IntBits::from_u128(64, w),
            &IntBits::from_u128(32, index),
            &IntBits::from_u128(64, answer),
        ]))
    };
    for (w, index, answer) in [(5, 1, 1 << 56), (5, 0, 0), (0, 9, 0)] {
        let verdict = run(w, index, answer);
        assert!(verdict.is_accepted(), "w {w}, index {index}: {verdict:?}");
    }
    let verdict = run(5, 9, 0);
    assert!(
        refused_by_the_constraints(&verdict),
        "an index past the end where taken: {verdict:?}"
    );
    assert!(!run(5, 1, 1).is_accepted(), "a wrong element was accepted");
}

/// An unconstrained function returns a wide integer, alone and in a sequence, to constrained code.
///
/// The answer is a hint, so each limb becomes a new column bounded at its width, as a narrower
/// answer's one element is range-checked. The program ties the answer back to its input, so a wrong
/// one is refused.
#[test]
fn an_unconstrained_function_answers_a_wide_integer() {
    for sequence in [false, true] {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let mut builder = HLSSABuilder::new(&mut ssa);
        let answer_type = if sequence {
            Type::int(320).array_of(2)
        } else {
            Type::int(320)
        };
        let hint_type = answer_type.clone();
        let (hint, ()) = builder.add_function("hint".to_string(), move |b| {
            b.function.add_return_type(hint_type);
            let entry = b.function.get_entry_id();
            let mut e = b
                .block(entry)
                .with_source_location(SourceLocation::synthetic("hint"));
            let input = e.add_parameter(Type::int(64));
            let wide = e.cast_to(CastTarget::Int(320), input);
            let amount = e.int_const(IntBits::from_u128(320, 250));
            let shifted = e.bin(BinaryArithOpKind::UShl, wide, amount);
            let answer = if sequence {
                e.mk_seq(
                    vec![wide, shifted],
                    SequenceTargetType::Array(2),
                    Type::int(320),
                )
            } else {
                shifted
            };
            e.terminate_return(vec![answer]);
        });
        builder.modify_function(main_id, |b| {
            b.function.add_return_type(Type::int(64));
            let entry = b.function.get_entry_id();
            let mut e = b
                .block(entry)
                .with_source_location(SourceLocation::synthetic("main"));
            let input = e.add_parameter(Type::int(64));
            let answer = e.call_unconstrained(hint, vec![input], 1)[0];
            let shifted = if sequence {
                let one = e.int_const(IntBits::one(32));
                e.array_get(answer, one)
            } else {
                answer
            };
            let amount = e.int_const(IntBits::from_u128(320, 250));
            let back = e.bin(BinaryArithOpKind::UShr, shifted, amount);
            let back = e.cast_to(CastTarget::Int(64), back);
            e.terminate_return(vec![back]);
        });
        let compiled =
            Compiled::new(ssa).unwrap_or_else(|error| panic!("in a sequence: {sequence}: {error}"));
        let input = IntBits::from_u128(64, 12345);
        let verdict = compiled.run(&input_block(&[&input, &input]));
        assert!(
            verdict.is_accepted(),
            "in a sequence: {sequence}: {verdict:?}"
        );
        let verdict = compiled
            .run_wasm(&input_block(&[&input, &input]))
            .expect("the WASM lane builds");
        assert!(
            verdict.is_accepted(),
            "in a sequence: {sequence} on WASM: {verdict:?}"
        );
        let wrong = input.xor(&IntBits::one(64));
        assert!(
            !compiled.run(&input_block(&[&input, &wrong])).is_accepted(),
            "in a sequence: {sequence} accepted a wrong answer"
        );
    }
}

/// A pure wide ordering asserted inside a branch on a witness holds only where the branch is
/// taken: the assertion is false, so taking the branch is refused and not taking it is not.
#[test]
fn a_pure_wide_ordering_asserted_in_a_witness_branch_binds_only_where_taken() {
    let other = IntBits::from_biguint(320, &BigUint::from(5u8).pow(60u32));
    let ssa = reading_a_pure_wide_element(vec![], move |e, x, w| {
        let other = e.int_const(other);
        let zero = e.int_const(IntBits::zero(64));
        let taken = e.cmp(zero, w, CmpKind::ULt);
        let (then, _) = e.add_block();
        let (merge, _) = e.add_block();
        e.seal_and_switch(Terminator::JmpIf(taken, then, merge), then);
        // The element is positive and far larger than `other`, so this is false.
        e.assert_cmp(CmpKind::ULt, x, other);
        e.seal_and_switch(Terminator::Jmp(merge, vec![]), merge);
        vec![]
    });
    let compiled = Compiled::new(ssa).expect("a guarded pure ordering compiles");
    let not_taken = compiled.run(&input_block(&[&IntBits::zero(64)]));
    assert!(not_taken.is_accepted(), "{not_taken:?}");
    let taken = compiled.run(&input_block(&[&IntBits::from_u128(64, 1)]));
    assert!(refused_by_the_constraints(&taken), "{taken:?}");
}

/// A witnessed field element reads back as a wide integer: its canonical value, zero-filled.
///
/// The widths are the field's bit size, a ragged one past it and a far wider one; the values reach
/// the field's top limb, up to the largest element there is. Each run returns two windows of the
/// answer, one of which straddles the field's top limb, and a wrong one is refused.
#[test]
fn a_witnessed_field_reads_back_as_a_wide_integer() {
    let modulus = BigUint::parse_bytes(
        b"21888242871839275222246405745257275088548364400416034343698204186575808495617",
        10,
    )
    .expect("the bn254 modulus");
    let values = [
        BigUint::from(5u8),
        (BigUint::from(1u8) << 200u32) + 3u8,
        &modulus - 1u8,
    ];
    for bits in [254usize, 300, 1000] {
        let ssa = main_program(
            &[Type::field()],
            &[Type::int(64), Type::int(64)],
            move |e, params| {
                let wide = e.cast_to(CastTarget::Int(bits), params[0]);
                let amount = e.int_const(IntBits::from_u128(bits, 190));
                let high = e.bin(BinaryArithOpKind::UShr, wide, amount);
                vec![
                    e.cast_to(CastTarget::Int(64), wide),
                    e.cast_to(CastTarget::Int(64), high),
                ]
            },
        );
        let compiled = Compiled::new(ssa)
            .unwrap_or_else(|error| panic!("int{bits} does not compile: {error}"));
        for value in &values {
            let window = |offset: u32| {
                IntBits::from_biguint(64, &((value >> offset) % (BigUint::from(1u8) << 64u32)))
            };
            let input = IntBits::from_biguint(254, value);
            let (low, high) = (window(0), window(190));
            let verdict = compiled.run(&input_block(&[&input, &low, &high]));
            assert!(verdict.is_accepted(), "int{bits} of {value}: {verdict:?}");
            let verdict = compiled
                .run_wasm(&input_block(&[&input, &low, &high]))
                .expect("the WASM lane builds");
            assert!(
                verdict.is_accepted(),
                "int{bits} of {value} on WASM: {verdict:?}"
            );
            let wrong = high.xor(&IntBits::one(64));
            assert!(
                !compiled
                    .run(&input_block(&[&input, &low, &wrong]))
                    .is_accepted(),
                "int{bits} of {value} accepted a wrong high window"
            );
        }
    }
}

/// An integer decomposes into bits and bytes, either way round, at the widths one field element
/// holds and past them, and a value with more digits than are asked for is refused.
///
/// The wide value is `2^299 + 2^250 + 5`: bit 250 is the first past a 248-bit chunk, and bit 299
/// the top one. A decomposition into more digits than the value has reads zero past it.
#[test]
fn an_integer_decomposes_into_bits_and_bytes_at_every_width() {
    let wide = (BigUint::from(1u8) << 299u32) + (BigUint::from(1u8) << 250u32) + 5u8;
    // (operand width, bytes rather than bits, big-endian, digits, digit read, want)
    let cases = [
        (300usize, false, false, 300usize, 0usize, 1u128),
        (300, false, false, 300, 250, 1),
        (300, false, false, 300, 299, 1),
        (300, false, false, 300, 298, 0),
        (300, false, true, 300, 0, 1),
        (300, false, false, 400, 350, 0),
        (300, true, false, 38, 31, 1 << 2),
        (300, true, false, 38, 37, 1 << 3),
        (300, true, true, 38, 0, 1 << 3),
        (64, false, false, 64, 2, 1),
        (64, true, true, 8, 7, 5),
        // More digits than a field element has, from a narrow integer.
        (64, false, false, 300, 2, 1),
        (64, false, false, 300, 299, 0),
        (64, true, false, 40, 0, 5),
        // Either side of the most digits whose sum the field holds below its modulus.
        (64, false, false, 253, 2, 1),
        (64, false, false, 254, 2, 1),
        (64, false, false, 254, 253, 0),
        (64, true, false, 31, 0, 5),
        (64, true, true, 32, 31, 5),
    ];
    for (bits, bytes, big, digits, read, want) in cases {
        let value = wide.clone();
        let digit_bits = if bytes { 8 } else { 1 };
        let ssa = main_program(
            &[Type::int(64)],
            &[Type::int(digit_bits)],
            move |e, params| {
                let operand = e.cast_to(CastTarget::Int(bits), params[0]);
                let operand = if bits > 64 {
                    let high = e.int_const(IntBits::from_biguint(bits, &(&value - 5u8)));
                    e.bin(BinaryArithOpKind::Or, operand, high)
                } else {
                    operand
                };
                let endianness = if big {
                    Endianness::Big
                } else {
                    Endianness::Little
                };
                let decomposed = if bytes {
                    e.to_radix(operand, Radix::Bytes, endianness, digits)
                } else {
                    e.to_bits(operand, endianness, digits)
                };
                let index = e.int_const(IntBits::from_u128(32, read as u128));
                vec![e.array_get(decomposed, index)]
            },
        );
        let what = format!(
            "int{bits} into {digits} {}, digit {read}",
            if bytes { "bytes" } else { "bits" }
        );
        let compiled = Compiled::new(ssa).unwrap_or_else(|error| panic!("{what}: {error}"));
        let inputs = |answer: u128| {
            input_block(&[
                &IntBits::from_u128(64, 5),
                &IntBits::from_u128(digit_bits, answer),
            ])
        };
        let verdict = compiled.run(&inputs(want));
        assert!(verdict.is_accepted(), "{what}: {verdict:?}");
        let verdict = compiled
            .run_wasm(&inputs(want))
            .expect("the WASM lane builds");
        assert!(verdict.is_accepted(), "{what} on WASM: {verdict:?}");
        assert!(
            !compiled.run(&inputs(want ^ 1)).is_accepted(),
            "{what} accepted a wrong digit"
        );
    }

    // Bit 299 is past 280 bits.
    let ssa = main_program(&[Type::int(64)], &[Type::int(1)], move |e, params| {
        let operand = e.cast_to(CastTarget::Int(300), params[0]);
        let high = e.int_const(IntBits::from_biguint(300, &(BigUint::from(1u8) << 299u32)));
        let operand = e.bin(BinaryArithOpKind::Or, operand, high);
        let decomposed = e.to_bits(operand, Endianness::Little, 280);
        let index = e.int_const(IntBits::zero(32));
        vec![e.array_get(decomposed, index)]
    });
    let compiled = Compiled::new(ssa).expect("a short decomposition compiles");
    let verdict = compiled.run(&input_block(&[
        &IntBits::from_u128(64, 5),
        &IntBits::one(1),
    ]));
    assert!(refused_by_the_constraints(&verdict), "{verdict:?}");
}

/// A bit window of an integer too wide for one field element, witnessed or pure, on both lanes:
/// one inside the low limb, one straddling two limbs, and one reaching the top bit.
#[test]
fn a_bit_window_of_a_wide_integer_is_cut_from_it() {
    let value = (BigUint::from(0x1234_5678_9ABC_DEF0u64) << 236u32) | BigUint::from(0xFEDC_BA98u32);
    for (offset, width) in [(0usize, 64usize), (40, 100), (236, 64)] {
        for witnessed in [true, false] {
            let pattern = IntBits::from_biguint(300, &value);
            let ssa = main_program(&[Type::int(64)], &[Type::int(64)], move |e, params| {
                let constant = e.int_const(pattern);
                let operand = if witnessed {
                    let wide = e.cast_to(CastTarget::Int(300), params[0]);
                    e.bin(BinaryArithOpKind::Or, wide, constant)
                } else {
                    constant
                };
                let window = e.bit_range(operand, offset, width);
                vec![e.cast_to(CastTarget::Int(64), window)]
            });
            let what = format!(
                "bits {offset}..{} of an int300, witnessed: {witnessed}",
                offset + width
            );
            let compiled = Compiled::new(ssa).unwrap_or_else(|error| panic!("{what}: {error}"));
            let want = IntBits::from_biguint(
                64,
                &((&value >> offset as u32) % (BigUint::from(1u8) << width.min(64) as u32)),
            );
            let inputs = |answer: &IntBits| input_block(&[&IntBits::zero(64), answer]);
            let verdict = compiled.run(&inputs(&want));
            assert!(verdict.is_accepted(), "{what}: {verdict:?}");
            let verdict = compiled
                .run_wasm(&inputs(&want))
                .expect("the WASM lane builds");
            assert!(verdict.is_accepted(), "{what} on WASM: {verdict:?}");
            assert!(
                !compiled
                    .run(&inputs(&want.xor(&IntBits::one(64))))
                    .is_accepted(),
                "{what} accepted a wrong window"
            );
        }
    }
}

/// `main(i: int8) -> (int64, int64) { (G[i] as int64, H[0]) }` for a global `G: [int320; 3]` and
/// a global `H: [int64; 1]` after it, whose slot moves when `G` takes one per limb.
fn reading_a_wide_global() -> HLSSA {
    let elements = pure_wide_elements();
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let init = ssa.add_function("globals_init".to_string());
    let deinit = ssa.add_function("globals_deinit".to_string());
    let mut builder = HLSSABuilder::new(&mut ssa);
    builder.modify_function(init, |b| {
        let entry = b.function.get_entry_id();
        let mut e = b
            .block(entry)
            .with_source_location(SourceLocation::synthetic("globals_init"));
        let constants: Vec<ValueId> = elements.iter().map(|v| e.int_const(v.clone())).collect();
        let wide = e.mk_seq(constants, SequenceTargetType::Array(3), Type::int(320));
        e.init_global(0, wide);
        let narrow = e.int_const(IntBits::from_u128(64, 77));
        let narrow = e.mk_seq(vec![narrow], SequenceTargetType::Array(1), Type::int(64));
        e.init_global(1, narrow);
        e.terminate_return(vec![]);
    });
    builder.modify_function(deinit, |b| {
        let entry = b.function.get_entry_id();
        let mut e = b
            .block(entry)
            .with_source_location(SourceLocation::synthetic("globals_deinit"));
        e.drop_global(0);
        e.drop_global(1);
        e.terminate_return(vec![]);
    });
    builder.modify_function(main_id, |b| {
        b.function.add_return_type(Type::int(64));
        b.function.add_return_type(Type::int(64));
        let entry = b.function.get_entry_id();
        let mut e = b
            .block(entry)
            .with_source_location(SourceLocation::synthetic("main"));
        let index = e.add_parameter(Type::int(8));
        let wide = e.read_global(0, Type::int(320).array_of(3));
        let index = e.cast_to(CastTarget::Int(32), index);
        let element = e.array_get(wide, index);
        let narrow = e.read_global(1, Type::int(64).array_of(1));
        let zero = e.int_const(IntBits::zero(32));
        let other = e.array_get(narrow, zero);
        let element = e.cast_to(CastTarget::Int(64), element);
        e.terminate_return(vec![element, other]);
    });
    ssa.set_global_types(vec![Type::int(320).array_of(3), Type::int(64).array_of(1)]);
    ssa.set_globals_init_fn(init);
    ssa.set_globals_deinit_fn(deinit);
    ssa
}

/// A global holding wide integers takes a slot per limb, and is read at a witness index on both
/// lanes; the global after it is read from where its slot moved to.
#[test]
fn a_global_holding_wide_integers_is_read_at_a_witness_index() {
    let compiled = Compiled::new(reading_a_wide_global()).expect("a wide global compiles");
    let elements = pure_wide_elements();
    for index in 0..3u128 {
        let want = elements[index as usize].cast(64);
        let inputs = [
            IntBits::from_u128(8, index),
            want.clone(),
            IntBits::from_u128(64, 77),
        ];
        let inputs: Vec<&IntBits> = inputs.iter().collect();
        let verdict = compiled.run(&input_block(&inputs));
        assert!(verdict.is_accepted(), "element {index}: {verdict:?}");
        let verdict = compiled
            .run_wasm(&input_block(&inputs))
            .expect("the WASM lane builds");
        assert!(
            verdict.is_accepted(),
            "element {index} on WASM: {verdict:?}"
        );
    }
}

/// A read or a write at a pure index past the end, known only once a loop has run, is refused when
/// the circuit is built: every execution fails there, as a constant assertion that cannot hold does.
#[test]
fn an_access_past_the_end_known_when_the_circuit_is_built_is_refused() {
    for write in [false, true] {
        let ssa = main_program(&[Type::int(8)], &[Type::int(8)], move |e, params| {
            let zero = e.int_const(IntBits::zero(32));
            let count = loop_count(e);
            let five = e.int_const(IntBits::from_u128(32, 5));
            let past = e.bin(BinaryArithOpKind::UAdd, count, five);
            let elements: Vec<ValueId> = (0..3)
                .map(|k| e.int_const(IntBits::from_u128(8, k)))
                .collect();
            let array = e.mk_seq(elements, SequenceTargetType::Array(3), Type::int(8));
            let array = if write {
                e.array_set(array, past, params[0])
            } else {
                array
            };
            let index = if write { zero } else { past };
            vec![e.array_get(array, index)]
        });
        let Err(error) = Compiled::new(ssa) else {
            panic!("write: {write}: a program that always reads past the end compiled");
        };
        assert!(
            matches!(error, DriverError::UnsatisfiableProgram(_)),
            "write: {write}: {error}"
        );
        assert!(
            error.to_string().contains("past the end"),
            "write: {write}: {error}"
        );
    }
}

/// Run the pipeline through its first phase, [`Driver::make_struct_access_static`], and no further.
fn first_phase_only(ssa: HLSSA) -> Result<(), DriverError> {
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

/// A program that calls through a function pointer gets through the first phase.
///
/// The first lowering pass types the untransformed SSA, so `TypeInfo` must type a
/// `CallTarget::Dynamic`, from the callee's `TypeExpr::Function`. Every `map`, `fold` or
/// `fn(..) -> ..` parameter in the corpus lowers to this shape.
#[test]
fn a_program_with_an_indirect_call_is_typed_before_defunctionalization() {
    assert!(
        first_phase_only(program_calling_through_a_function_pointer(8)).is_ok(),
        "an indirect call is not a width problem"
    );
}

// ADVERSARIAL COVERAGE
// ================================================================================================

/// `main(p: int64, b: int1) -> (int64, int64) { (low, top) of (b ? big : 0) op (p as int(bits)) }`.
///
/// Selecting `big` by a witnessed bit makes it a witnessed value held as limbs. The answer leaves
/// as its low and top 64 bits.
fn limbed_operation(kind: BinaryArithOpKind, bits: usize, big: &IntBits) -> HLSSA {
    let big = big.clone();
    main_program(
        &[Type::int(64), Type::int(1)],
        &[Type::int(64), Type::int(64)],
        move |e, params| {
            let constant = e.int_const(big);
            let zero = e.int_const(IntBits::zero(bits));
            let lhs = e.select(params[1], constant, zero);
            let rhs = e.cast_to(CastTarget::Int(bits), params[0]);
            let answer = e.bin(kind, lhs, rhs);
            let top = e.int_const(IntBits::from_u128(bits, (bits - 64) as u128));
            let high = e.bin(BinaryArithOpKind::UShr, answer, top);
            vec![
                e.cast_to(CastTarget::Int(64), answer),
                e.cast_to(CastTarget::Int(64), high),
            ]
        },
    )
}

/// The inputs [`limbed_operation`] runs on and the answer it declares.
fn limbed_inputs(kind: BinaryArithOpKind, bits: usize, big: &IntBits, p: u128) -> Vec<IntBits> {
    let rhs = IntBits::from_u128(bits, p);
    let answer = mavros_int_semantics::eval(IntOp::from(kind), big, &rhs)
        .value()
        .unwrap_or_else(|| panic!("{kind:?} at {bits} bits rejects its operands"));
    let low = answer.cast(64);
    let high = IntBits::from_biguint(64, &(BigUint::from(&answer) >> (bits - 64)));
    vec![IntBits::from_u128(64, p), IntBits::one(1), low, high]
}

/// Every limbed operation pins every column at a ragged width (300) and a far one (1000), and at
/// the ragged one no two columns can trade a limb's place value.
///
/// The operand reaching the top limb is the first the operation accepts of: top bit set, a small
/// negative, a small positive. So each signed form cuts its sign on the ragged limb.
#[test]
fn every_column_of_a_limbed_operation_is_pinned_at_ragged_and_far_widths() {
    use BinaryArithOpKind::{
        SAdd, SDiv, SMul, SRem, SShr, UAdd, UDiv, UMul, URem, UShl, UShr, USub,
    };
    for bits in [300usize, 1000] {
        let small = (BigUint::from(1u8) << (bits - 70)) | BigUint::from(12345u32);
        let candidates = [
            // The top bit set, so negative under a signed reading.
            IntBits::from_biguint(bits, &((BigUint::from(1u8) << (bits - 1)) | &small)),
            // Every high bit set: a small negative, for a signed product that must not overflow.
            IntBits::from_biguint(bits, &((BigUint::from(1u8) << bits) - &small)),
            IntBits::from_biguint(bits, &small),
        ];
        let three = IntBits::from_u128(bits, 3);
        for kind in [
            UAdd, USub, UMul, UDiv, URem, UShl, UShr, SAdd, SMul, SDiv, SRem, SShr,
        ] {
            let what = format!("{kind:?} at {bits} bits");
            let big = candidates
                .iter()
                .find(|big| {
                    mavros_int_semantics::eval(IntOp::from(kind), big, &three)
                        .value()
                        .is_some()
                })
                .unwrap_or_else(|| panic!("{what}: the model rejects every candidate"))
                .clone();
            let compiled = Compiled::new(limbed_operation(kind, bits, &big))
                .unwrap_or_else(|error| panic!("{what}: {error}"));
            let inputs = limbed_inputs(kind, bits, &big, 3);
            let inputs: Vec<&IntBits> = inputs.iter().collect();
            assert_every_column_is_pinned(&what, &compiled, &inputs);
            if bits == 300 {
                assert_no_two_columns_trade_a_limb(&what, &compiled, &inputs);
            }
        }
    }
}

/// The newer wide lowerings pin every column, and no two of their columns can trade a limb's place
/// value: a bit window of a wide integer, a field element read as a wide integer, a wide
/// unconstrained answer, and integer digits as one field image and in chunks.
#[test]
fn every_column_of_the_wide_shape_lowerings_is_pinned() {
    let wide_value = (BigUint::from(0x1234_5678u32) << 260u32) | BigUint::from(0xFEDC_BA98u32);
    let parameter = IntBits::from_u128(64, 5);

    // A bit window straddling two limbs of a witnessed int300.
    let window_value = wide_value.clone();
    let window = main_program(&[Type::int(64)], &[Type::int(64)], move |e, params| {
        let wide = e.cast_to(CastTarget::Int(300), params[0]);
        let constant = e.int_const(IntBits::from_biguint(300, &window_value));
        let operand = e.bin(BinaryArithOpKind::Or, wide, constant);
        let window = e.bit_range(operand, 40, 100);
        vec![e.cast_to(CastTarget::Int(64), window)]
    });
    let window_answer = IntBits::from_biguint(
        64,
        &(((&wide_value | BigUint::from(5u8)) >> 40u32) % (BigUint::from(1u8) << 64u32)),
    );

    // A field element read as an int300 and an int1000.
    let field_reads: Vec<(usize, HLSSA)> = [300usize, 1000]
        .into_iter()
        .map(|bits| {
            let ssa = main_program(
                &[Type::field()],
                &[Type::int(64), Type::int(64)],
                move |e, params| {
                    let wide = e.cast_to(CastTarget::Int(bits), params[0]);
                    let amount = e.int_const(IntBits::from_u128(bits, 190));
                    let high = e.bin(BinaryArithOpKind::UShr, wide, amount);
                    vec![
                        e.cast_to(CastTarget::Int(64), wide),
                        e.cast_to(CastTarget::Int(64), high),
                    ]
                },
            );
            (bits, ssa)
        })
        .collect();
    let element = (BigUint::from(1u8) << 200u32) + 3u8;
    let element_input = IntBits::from_biguint(254, &element);
    let element_low = IntBits::from_u128(64, 3);
    let element_high = IntBits::from_u128(64, 1 << 10);

    // A wide answer from an unconstrained function, tied back to its input.
    let mut unconstrained = HLSSA::with_main("main".to_string());
    let main_id = unconstrained.get_unique_entrypoint_id();
    let mut builder = HLSSABuilder::new(&mut unconstrained);
    let (hint, ()) = builder.add_function("hint".to_string(), |b| {
        b.function.add_return_type(Type::int(300));
        let entry = b.function.get_entry_id();
        let mut e = b
            .block(entry)
            .with_source_location(SourceLocation::synthetic("hint"));
        let input = e.add_parameter(Type::int(64));
        let wide = e.cast_to(CastTarget::Int(300), input);
        let amount = e.int_const(IntBits::from_u128(300, 230));
        let shifted = e.bin(BinaryArithOpKind::UShl, wide, amount);
        e.terminate_return(vec![shifted]);
    });
    builder.modify_function(main_id, |b| {
        b.function.add_return_type(Type::int(64));
        let entry = b.function.get_entry_id();
        let mut e = b
            .block(entry)
            .with_source_location(SourceLocation::synthetic("main"));
        let input = e.add_parameter(Type::int(64));
        let answer = e.call_unconstrained(hint, vec![input], 1)[0];
        let amount = e.int_const(IntBits::from_u128(300, 230));
        let back = e.bin(BinaryArithOpKind::UShr, answer, amount);
        let back = e.cast_to(CastTarget::Int(64), back);
        e.terminate_return(vec![back]);
    });

    // An integer's digits: an int64 as one field image, and an int300 in chunks.
    let digits = |bits: usize, count: usize, read: usize| {
        let high = wide_value.clone();
        main_program(&[Type::int(64)], &[Type::int(1)], move |e, params| {
            let operand = e.cast_to(CastTarget::Int(bits), params[0]);
            let operand = if bits > 64 {
                let constant = e.int_const(IntBits::from_biguint(bits, &high));
                e.bin(BinaryArithOpKind::Or, operand, constant)
            } else {
                operand
            };
            let decomposed = e.to_bits(operand, Endianness::Little, count);
            let index = e.int_const(IntBits::from_u128(32, read as u128));
            vec![e.array_get(decomposed, index)]
        })
    };

    let mut cases: Vec<(String, HLSSA, Vec<IntBits>)> = vec![
        (
            "a bit window of an int300".to_string(),
            window,
            vec![parameter.clone(), window_answer],
        ),
        (
            "an unconstrained int300 answer".to_string(),
            unconstrained,
            vec![parameter.clone(), parameter.clone()],
        ),
        (
            "an int64's 64 bits".to_string(),
            digits(64, 64, 2),
            vec![parameter.clone(), IntBits::one(1)],
        ),
        (
            "an int300's 300 bits".to_string(),
            digits(300, 300, 260),
            vec![parameter.clone(), IntBits::zero(1)],
        ),
    ];
    for (bits, ssa) in field_reads {
        cases.push((
            format!("a field element read as an int{bits}"),
            ssa,
            vec![
                element_input.clone(),
                element_low.clone(),
                element_high.clone(),
            ],
        ));
    }
    for (what, ssa, inputs) in cases {
        let compiled = Compiled::new(ssa).unwrap_or_else(|error| panic!("{what}: {error}"));
        let inputs: Vec<&IntBits> = inputs.iter().collect();
        assert_every_column_is_pinned(&what, &compiled, &inputs);
        assert_no_two_columns_trade_a_limb(&what, &compiled, &inputs);
    }
}

// THE WIDEST VALUES ON WASM
// ================================================================================================

/// A 16384-bit answer from an unconstrained function enters the representation and is read back,
/// on both lanes.
///
/// Its limbs are cut by shifts by a constant, which wasm emits inline. Divisions would each call a
/// runtime helper through value-wide buffers, and the function would be too large to compile.
#[test]
#[ignore = "a 16384-bit value entering the representation; several minutes in a debug build"]
fn the_widest_unconstrained_answer_runs_on_both_lanes() {
    let bits = 16384;
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let mut builder = HLSSABuilder::new(&mut ssa);
    let (hint, ()) = builder.add_function("hint".to_string(), move |b| {
        b.function.add_return_type(Type::int(bits));
        let entry = b.function.get_entry_id();
        let mut e = b
            .block(entry)
            .with_source_location(SourceLocation::synthetic("hint"));
        let input = e.add_parameter(Type::int(64));
        let wide = e.cast_to(CastTarget::Int(bits), input);
        let amount = e.int_const(IntBits::from_u128(bits, 16000));
        let shifted = e.bin(BinaryArithOpKind::UShl, wide, amount);
        e.terminate_return(vec![shifted]);
    });
    builder.modify_function(main_id, |b| {
        b.function.add_return_type(Type::int(64));
        let entry = b.function.get_entry_id();
        let mut e = b
            .block(entry)
            .with_source_location(SourceLocation::synthetic("main"));
        let input = e.add_parameter(Type::int(64));
        let answer = e.call_unconstrained(hint, vec![input], 1)[0];
        let amount = e.int_const(IntBits::from_u128(bits, 16000));
        let back = e.bin(BinaryArithOpKind::UShr, answer, amount);
        let back = e.cast_to(CastTarget::Int(64), back);
        e.terminate_return(vec![back]);
    });
    let compiled = Compiled::new(ssa).expect("compiles");
    let input = IntBits::from_u128(64, 9);
    let inputs = input_block(&[&input, &input]);
    let verdict = compiled.run(&inputs);
    assert!(verdict.is_accepted(), "{verdict:?}");
    let verdict = compiled.run_wasm(&inputs).expect("the WASM lane builds");
    assert!(verdict.is_accepted(), "on WASM: {verdict:?}");
    let wrong = input_block(&[&input, &IntBits::from_u128(64, 8)]);
    assert!(
        !compiled.run(&wrong).is_accepted(),
        "a wrong answer was accepted"
    );
}

/// The shape of a call's results in [`program_calling_with_many_arguments`].
#[derive(Clone, Copy)]
enum CallResults {
    /// One `int32`.
    One,
    /// Two `int32`s, the same value twice.
    Two,
    /// One `int128`, which lowers to two wasm values.
    Wide,
}

/// `main(input, first, second) -> int32` calling a callee on `count` arguments `input + i` and the
/// two positions, which answers `args[first] + 2 * args[second]` in the shape `results` names.
///
/// Every argument is read through a sequence at witnessed positions, so none is dropped as unused,
/// and the callee builds no value per argument, which would overflow the unoptimized wasm stack.
fn program_calling_with_many_arguments(count: usize, results: CallResults) -> HLSSA {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let mut builder = HLSSABuilder::new(&mut ssa);
    let (callee, ()) = builder.add_function("callee".to_string(), move |b| {
        let result_types = match results {
            CallResults::One => vec![Type::int(32)],
            CallResults::Two => vec![Type::int(32), Type::int(32)],
            CallResults::Wide => vec![Type::int(128)],
        };
        for result_type in result_types {
            b.function.add_return_type(result_type);
        }
        let entry = b.function.get_entry_id();
        let mut e = b
            .block(entry)
            .with_source_location(SourceLocation::synthetic("callee"));
        let params: Vec<ValueId> = (0..count).map(|_| e.add_parameter(Type::int(32))).collect();
        let (first, second) = (
            e.add_parameter(Type::int(32)),
            e.add_parameter(Type::int(32)),
        );
        let sequence = e.mk_seq(params, SequenceTargetType::Array(count), Type::int(32));
        let first = e.array_get(sequence, first);
        let second = e.array_get(sequence, second);
        let two = e.int_const(IntBits::from_u128(32, 2));
        let second = e.bin(BinaryArithOpKind::UMul, second, two);
        let sum = e.bin(BinaryArithOpKind::UAdd, first, second);
        let answers = match results {
            CallResults::One => vec![sum],
            CallResults::Two => vec![sum, sum],
            CallResults::Wide => vec![e.cast_to(CastTarget::Int(128), sum)],
        };
        e.terminate_return(answers);
    });
    builder.modify_function(main_id, |b| {
        b.function.add_return_type(Type::int(32));
        let entry = b.function.get_entry_id();
        let mut e = b
            .block(entry)
            .with_source_location(SourceLocation::synthetic("main"));
        let input = e.add_parameter(Type::int(32));
        let first = e.add_parameter(Type::int(32));
        let second = e.add_parameter(Type::int(32));
        let mut args: Vec<ValueId> = (0..count)
            .map(|position| {
                let offset = e.int_const(IntBits::from_u128(32, position as u128));
                e.bin(BinaryArithOpKind::UAdd, input, offset)
            })
            .collect();
        args.extend([first, second]);
        let arity = match results {
            CallResults::One | CallResults::Wide => 1,
            CallResults::Two => 2,
        };
        let answers = e.call(callee, args, arity);
        let answer = match results {
            CallResults::One => answers[0],
            CallResults::Two => e.bin(BinaryArithOpKind::UAdd, answers[0], answers[1]),
            CallResults::Wide => e.cast_to(CastTarget::Int(32), answers[0]),
        };
        e.terminate_return(vec![answer]);
    });
    ssa
}

/// Runs [`program_calling_with_many_arguments`] on both lanes at a few positions, and requires a
/// wrong answer to be refused.
fn assert_a_call_with_many_arguments_runs(count: usize, results: CallResults) {
    let compiled =
        Compiled::new(program_calling_with_many_arguments(count, results)).expect("compiles");
    let int32 = |value: u128| IntBits::from_u128(32, value);
    let input = 3u128;
    let last = count as u128 - 1;
    for (first, second) in [(last, 1u128), (0, last - 1), (last / 2, last / 2)] {
        let sum = (input + first) + 2 * (input + second);
        let want = match results {
            CallResults::One | CallResults::Wide => sum,
            CallResults::Two => 2 * sum,
        };
        let inputs = input_block(&[&int32(input), &int32(first), &int32(second), &int32(want)]);
        let what = format!("{count} arguments, positions {first} and {second}");
        let verdict = compiled.run(&inputs);
        assert!(verdict.is_accepted(), "{what}: {verdict:?}");
        let verdict = compiled.run_wasm(&inputs).expect("the WASM lane builds");
        assert!(verdict.is_accepted(), "{what} on WASM: {verdict:?}");
        let wrong = input_block(&[
            &int32(input),
            &int32(first),
            &int32(second),
            &int32(want + 1),
        ]);
        assert!(
            !compiled.run(&wrong).is_accepted(),
            "{what}: a wrong sum was accepted"
        );
    }
}

/// A function with more arguments than a wasm function may declare takes them through memory, on
/// both lanes.
///
/// 1000 `int32` arguments, two positions and the VM pointer are 1003 parameters, past the cap of
/// 1000, as four witnessed 16384-bit values (256 limbs each) are.
#[test]
fn a_call_with_more_arguments_than_wasm_allows_runs_on_both_lanes() {
    assert_a_call_with_many_arguments_runs(1000, CallResults::One);
}

/// A result that lowers to more than one wasm value is returned through a pointer LLVM adds as a
/// parameter, which counts against the cap.
///
/// 997 `int32` arguments, two positions and the VM pointer are exactly 1000 parameters, so an
/// `int128` result takes the function past the cap. Two `int32` results are run at the same count.
#[test]
fn a_call_whose_result_takes_a_parameter_counts_it_against_the_wasm_cap() {
    for results in [CallResults::Wide, CallResults::Two] {
        assert_a_call_with_many_arguments_runs(997, results);
    }
}
