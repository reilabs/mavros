//! Goldilocks programs built directly as SSA, compiled to a Goldilocks R1CS and VM bytecode, run on
//! the VM, and checked against the field and integer model.
//!
//! Every program here also passes the harness' AD check. None has a WASM lane: the runtime has no
//! Goldilocks instance.

use mavros_artifacts::{
    Field, FieldConfig, Goldilocks, InputValueOrdered, field::goldilocks::MODULUS,
};
use mavros_compiler::compiler::{
    passes::shared::limbs::witness_limb_bits,
    ssa::{
        ValueId,
        hlssa::{
            BinaryArithOpKind, CastTarget, CmpKind, Endianness, Type,
            builder::{HLBlockEmitter, HLEmitter as _},
        },
    },
};
use mavros_int_semantics::{CmpOp, IntBits, IntOp, SignedValue, corners};
use num_bigint::BigUint;

use crate::{
    Limbed, Rhs, assert_every_column_is_pinned, assert_no_two_columns_trade_a_limb,
    check_limbed_grid_in, corner_pairs, dedup, few_shift_amounts, half_width_corners,
    harness::{BinaryOpOracle, Compiled, Verdict, input_block_in, main_program_in},
    limb_reaching_values, limb_shift_amounts, signed_limb_corners_at,
};

// HELPERS
// ================================================================================================

fn element(value: u64) -> Goldilocks {
    Goldilocks::from_canonical(value).expect("a value below the modulus")
}

fn input(value: Goldilocks) -> InputValueOrdered<Goldilocks> {
    InputValueOrdered::Field(value)
}

/// Values at the boundaries the field's reductions branch on.
fn operands() -> Vec<Goldilocks> {
    [
        0,
        1,
        2,
        3,
        (1 << 32) - 1,
        1 << 32,
        (MODULUS - 1) / 2,
        (MODULUS + 1) / 2,
        MODULUS - 2,
        MODULUS - 1,
        0x1234_5678_9abc_def0,
    ]
    .into_iter()
    .map(element)
    .collect()
}

/// Compiles `main(params...) -> returns...` over Goldilocks.
fn compile(
    params: &[Type],
    returns: &[Type],
    body: impl FnOnce(&mut HLBlockEmitter<'_>, &[ValueId]) -> Vec<ValueId>,
) -> Compiled<Goldilocks> {
    let ssa = main_program_in(FieldConfig::goldilocks(), params, returns, body);
    Compiled::compile(ssa).unwrap_or_else(|error| panic!("a Goldilocks program failed: {error}"))
}

/// Asserts `inputs` with `expected` declared as the return is accepted, and with the return off
/// by one is refused at the return check, so the return is constrained.
fn assert_returns(
    what: &str,
    compiled: &Compiled<Goldilocks>,
    inputs: &[InputValueOrdered<Goldilocks>],
    expected: InputValueOrdered<Goldilocks>,
    wrong: InputValueOrdered<Goldilocks>,
) {
    let mut block = inputs.to_vec();
    block.push(expected);
    let verdict = compiled.run(&block);
    assert!(verdict.is_accepted(), "{what}: {verdict:?}");

    *block.last_mut().unwrap() = wrong;
    let verdict = compiled.run(&block);
    assert!(
        matches!(
            verdict,
            Verdict::Trapped {
                in_return_check: true,
                ..
            }
        ),
        "{what}: a wrong return gave {verdict:?}"
    );
}

// FIELD ARITHMETIC
// ================================================================================================

/// `a op b` on witnessed field operands agrees with the field at every boundary pair, and a zero
/// divisor is refused.
#[test]
fn field_arithmetic_agrees_with_the_field() {
    type Model = fn(Goldilocks, Goldilocks) -> Option<Goldilocks>;
    let cases: [(BinaryArithOpKind, Model); 4] = [
        (BinaryArithOpKind::UAdd, |a, b| Some(a + b)),
        (BinaryArithOpKind::USub, |a, b| Some(a - b)),
        (BinaryArithOpKind::UMul, |a, b| Some(a * b)),
        (BinaryArithOpKind::UDiv, |a, b| {
            b.inverse().map(|inverse| a * inverse)
        }),
    ];
    for (kind, model) in cases {
        let compiled = compile(&[Type::field(), Type::field()], &[Type::field()], |e, p| {
            vec![e.bin(kind, p[0], p[1])]
        });
        for &a in &operands() {
            for &b in &operands() {
                match model(a, b) {
                    Some(expected) => assert_returns(
                        &format!("{kind:?}({a}, {b}) = {expected}"),
                        &compiled,
                        &[input(a), input(b)],
                        input(expected),
                        input(expected + Goldilocks::ONE),
                    ),
                    None => {
                        let verdict = compiled.run(&[input(a), input(b), input(Goldilocks::ZERO)]);
                        assert!(verdict.is_refusal(), "{kind:?}({a}, {b}): {verdict:?}");
                    }
                }
            }
        }
    }
}

/// Field equality on witnessed operands.
#[test]
fn field_equality_agrees_with_the_field() {
    let compiled = compile(&[Type::field(), Type::field()], &[Type::bool()], |e, p| {
        vec![e.cmp(p[0], p[1], CmpKind::Eq)]
    });
    let bit = |holds: bool| input(Goldilocks::from(holds));
    for &a in &operands() {
        for &b in &operands() {
            assert_returns(
                &format!("{a} == {b}"),
                &compiled,
                &[input(a), input(b)],
                bit(a == b),
                bit(a != b),
            );
        }
    }
}

/// A constant expression evaluates modulo the Goldilocks prime: `(p - 1)·(p - 1) + 2^32·2^32`
/// is `1 + (2^32 - 1)`, the latter being `2^64 mod p`, and dividing it by 3 multiplies by `3^-1`.
#[test]
fn field_constants_fold_modulo_the_goldilocks_prime() {
    let compiled = compile(&[], &[Type::field()], |e, _| {
        let top = e.field_const(MODULUS - 1);
        let square = e.umul(top, top);
        let low = e.field_const(1u64 << 32);
        let wrap = e.umul(low, low);
        let sum = e.uadd(square, wrap);
        let three = e.field_const(3u64);
        vec![e.udiv(sum, three)]
    });
    let expected = element(1 << 32) * element(3).inverse().unwrap();
    assert_returns(
        "((p - 1)^2 + 2^64) / 3",
        &compiled,
        &[],
        input(expected),
        input(expected + Goldilocks::ONE),
    );
}

/// A witnessed field element narrowed to an integer, in the shape the Noir frontend gives
/// `x as uN` (a bit window, then the cast), decomposes the element at bn254's 128-bit halves, so
/// the lowering refuses it on Goldilocks (Layer 6). This pins the refusal until a Goldilocks
/// decomposition exists.
#[test]
#[should_panic(expected = "decomposes the element into two 128-bit halves")]
fn a_witnessed_field_bit_window_is_refused_on_goldilocks() {
    compile(&[Type::field()], &[Type::int(8)], |e, p| {
        let window = e.bit_range(p[0], 0, 8);
        vec![e.cast_to(CastTarget::Int(8), window)]
    });
}

// INTEGER ARITHMETIC
// ================================================================================================

/// Sweeps `kind` at `bits` over every corner pair on Goldilocks, and checks the declared return is
/// constrained.
fn sweep(kind: BinaryArithOpKind, bits: usize) {
    let op = IntOp::from(kind);
    let oracle = BinaryOpOracle::<Goldilocks>::compile(kind, bits).unwrap_or_else(|error| {
        panic!("{kind:?} at {bits} bits failed to compile on Goldilocks: {error}")
    });
    let pairs = corner_pairs(op, bits);
    for (lhs, rhs) in &pairs {
        if let Err(disagreement) = oracle.check(lhs, rhs) {
            panic!("Goldilocks: {disagreement}");
        }
    }
    let accepted = pairs
        .iter()
        .find(|(lhs, rhs)| mavros_int_semantics::eval(op, lhs, rhs).value().is_some())
        .expect("the model accepts a corner pair");
    if let Err(unconstrained) = oracle.check_a_wrong_answer_is_refused(&accepted.0, &accepted.1) {
        panic!("Goldilocks: {unconstrained}");
    }
}

/// Every integer operation agrees with the model at every corner pair, at widths below 32 bits.
#[test]
fn every_operation_agrees_with_the_model_on_goldilocks() {
    for bits in [1, 5, 8, 16, 31] {
        for kind in BinaryArithOpKind::ALL {
            sweep(kind, bits);
        }
    }
}

/// The additive and bitwise operations agree up to 63 bits, the widest integer one Goldilocks
/// element holds injectively. 62 bits is the widest sum one element holds, and 63 the first width
/// whose sum takes the carry chain. Wider integer _parameters_ do not fit one entry-blob element.
#[test]
fn additive_and_bitwise_operations_agree_up_to_63_bits() {
    use BinaryArithOpKind::{And, Or, SAdd, SSub, UAdd, USub, Xor};
    for bits in [32, 48, 62, 63] {
        for kind in [UAdd, SAdd, USub, SSub, And, Or, Xor] {
            sweep(kind, bits);
        }
    }
}

/// Products, quotients, remainders and shifts agree with the model on both sides of the single
/// cell's last width, 32 bits, where a `u32` division's `q·d + r` reaches `p - 1`. Past it, the
/// schoolbook, the division and the limb-wise shift run on 16-bit witness limbs: three full ones
/// at 48, and a narrow top limb at 33, 62 and 63.
#[test]
fn products_quotients_and_shifts_agree_from_32_to_63_bits() {
    use BinaryArithOpKind::{SDiv, SMul, SRem, SShl, SShr, UDiv, UMul, URem, UShl, UShr};
    for bits in [32, 33, 48, 62, 63] {
        for kind in [UMul, SMul, UDiv, SDiv, URem, SRem, UShl, SShl, UShr, SShr] {
            sweep(kind, bits);
        }
    }
}

// LIMBED OPERATIONS
// ================================================================================================

/// Every limbed operation at `bits` against the model, over the corners of Goldilocks' witness
/// limb, with each [`Rhs`]: witnessed, constant, and witnessed against a pure left operand. A
/// selected operand is a constant chosen by a witnessed bit, so it can be wider than one entry-blob
/// element.
///
/// Up to 64 bits a shift takes every whole number of limbs and one either side; past that, a few
/// amounts, which keeps the wider matrices to minutes.
fn check_limbed_matrix(bits: usize) {
    let limb = witness_limb_bits(FieldConfig::goldilocks());
    let mut values = vec![IntBits::zero(bits)];
    values.extend(limb_reaching_values(bits, limb));
    values.extend(half_width_corners(bits));
    let values = dedup(values);
    let signed_values = signed_limb_corners_at(bits, limb);
    let amounts = if bits <= 64 {
        limb_shift_amounts(bits, limb)
    } else {
        few_shift_amounts(bits, limb)
    };
    for rhs in [Rhs::Witnessed, Rhs::Constant, Rhs::AgainstPure] {
        for limbed in [
            Limbed::Sum,
            Limbed::Difference,
            Limbed::Ordering,
            Limbed::Product,
            Limbed::Quotient,
            Limbed::Remainder,
        ] {
            check_limbed_matrix_pairs(limbed, bits, &values, &values, rhs);
        }
        for limbed in [
            Limbed::SignedSum,
            Limbed::SignedDifference,
            Limbed::SignedOrdering,
            Limbed::SignedProduct,
            Limbed::SignedQuotient,
            Limbed::SignedRemainder,
        ] {
            check_limbed_matrix_pairs(limbed, bits, &signed_values, &signed_values, rhs);
        }
        for limbed in [Limbed::ShiftLeft, Limbed::ShiftRight] {
            check_limbed_matrix_pairs(limbed, bits, &values, &amounts, rhs);
        }
        for limbed in [Limbed::SignedShiftLeft, Limbed::SignedShiftRight] {
            check_limbed_matrix_pairs(limbed, bits, &signed_values, &amounts, rhs);
        }
    }
}

/// `limbed` over every left value against every right one, by [`check_limbed_grid_in`].
///
/// A constant right operand the model rejects against both idle left operands (zero and all ones),
/// as a zero divisor or an amount past the width is, is left out: [`Rhs::Constant`] keeps the left
/// operand idle on every run but the named one, so it needs an idle left operand the model accepts.
fn check_limbed_matrix_pairs(
    limbed: Limbed,
    bits: usize,
    lefts: &[IntBits],
    rights: &[IntBits],
    rhs: Rhs,
) {
    let idle_lefts = [IntBits::zero(bits), IntBits::all_ones(bits)];
    let rights: Vec<IntBits> = rights
        .iter()
        .filter(|right| {
            !matches!(rhs, Rhs::Constant)
                || idle_lefts
                    .iter()
                    .any(|left| limbed.model(left, right).is_some())
        })
        .cloned()
        .collect();
    check_limbed_grid_in::<Goldilocks>(limbed, bits, lefts, &rights, rhs);
}

/// [`crate::check_limbed_pairs_in`] reaches the WASM lane through [`Compiled::as_bn254`], so a
/// downcast that missed bn254 would skip that lane without a failure.
#[test]
fn only_a_bn254_program_has_a_wasm_lane() {
    let program =
        |field| main_program_in(field, &[Type::int(8)], &[Type::int(8)], |_, p| vec![p[0]]);
    let bn254 = Compiled::<Field>::compile(program(FieldConfig::bn254())).unwrap();
    assert!(bn254.as_bn254().is_some());
    let goldilocks = Compiled::<Goldilocks>::compile(program(FieldConfig::goldilocks())).unwrap();
    assert!(goldilocks.as_bn254().is_none());
}

/// 63 bits, the widest integer one Goldilocks element holds injectively: the limbed lowerings split
/// each operand into four limbs, the top one 15 bits wide.
#[test]
fn every_limbed_operation_agrees_with_the_model_at_63_bits() {
    check_limbed_matrix(63);
}

/// The integer comparisons agree with the model at every corner pair, on both sides of the switch
/// from a single-element comparison (62 bits) to the carry chain (63 bits).
#[test]
fn integer_comparisons_agree_with_the_model_on_goldilocks() {
    for (kind, op) in [
        (CmpKind::ULt, CmpOp::ULt),
        (CmpKind::SLt, CmpOp::SLt),
        (CmpKind::Eq, CmpOp::Eq),
    ] {
        for bits in [31, 62, 63] {
            let compiled = compile(
                &[Type::int(bits), Type::int(bits)],
                &[Type::int(1)],
                |e, p| vec![e.cmp(p[0], p[1], kind)],
            );
            let values: Vec<IntBits> = corners::values(bits)
                .into_iter()
                .map(|value| IntBits::from_u128(bits, value))
                .collect();
            for lhs in &values {
                for rhs in &values {
                    let holds = lhs.compare(op, rhs);
                    assert_returns(
                        &format!("{kind:?}({lhs:?}, {rhs:?}) at {bits} bits"),
                        &compiled,
                        &input_block_in::<Goldilocks>(&[lhs, rhs]),
                        input(Goldilocks::from(holds)),
                        input(Goldilocks::from(!holds)),
                    );
                }
            }
        }
    }
}

/// A witnessed field element decomposes into the 64 bits of its integer, in either order. The
/// decomposition is raw: its canonicity is the caller's to check.
#[test]
fn a_field_element_decomposes_into_its_64_bits() {
    for endianness in [Endianness::Little, Endianness::Big] {
        let compiled = compile(&[Type::field()], &[Type::int(1).array_of(64)], |e, p| {
            vec![e.to_bits(p[0], endianness, 64)]
        });
        for &a in &operands() {
            let bit = |index: usize| {
                let position = match endianness {
                    Endianness::Little => index,
                    Endianness::Big => 63 - index,
                };
                Goldilocks::from((a.value() >> position) & 1)
            };
            let bits: Vec<_> = (0..64).map(|index| input(bit(index))).collect();
            let mut block = vec![input(a)];
            block.extend(bits);
            let verdict = compiled.run(&block);
            assert!(
                verdict.is_accepted(),
                "{a} to {endianness:?} bits: {verdict:?}"
            );

            let mut wrong = block.clone();
            wrong[1] = input(Goldilocks::ONE - bit(0));
            let verdict = compiled.run(&wrong);
            assert!(
                !verdict.is_accepted(),
                "{a} to {endianness:?} bits accepted a flipped first bit: {verdict:?}"
            );
        }
    }
}

/// A Goldilocks element has 64 bits, so a witnessed decomposition into 65 is refused.
#[test]
#[should_panic(expected = "asks for more bits than a field element has")]
fn a_65_bit_decomposition_is_refused_on_goldilocks() {
    compile(&[Type::field()], &[Type::int(1).array_of(65)], |e, p| {
        vec![e.to_bits(p[0], Endianness::Little, 65)]
    });
}

/// No witness column of an accepted run can move alone, and no two can trade a limb's place value,
/// on both sides of each lowering switch. A sum or product that wraps modulo `p` into a second
/// satisfying witness shows here; an honest run cannot reveal it.
#[test]
fn every_witness_column_of_a_goldilocks_operation_is_pinned() {
    use BinaryArithOpKind::{And, SDiv, SMul, SShr, UAdd, UDiv, UMul, URem, UShl, UShr, USub, Xor};
    let int = |bits: usize, value: u128| IntBits::from_u128(bits, value);
    let negative =
        |bits: usize, magnitude: u128| IntBits::from_signed(bits, &-SignedValue::from(magnitude));
    let top = |bits: usize| IntBits::all_ones(bits);
    let every_other = |bits: usize| int(bits, ((1 << bits) - 1) / 3);
    for (kind, bits, lhs, rhs) in [
        (UAdd, 62, int(62, 1 << 61), int(62, (1 << 61) - 1)),
        (UAdd, 63, int(63, 1 << 62), int(63, (1 << 62) - 1)),
        (USub, 63, top(63), int(63, (1 << 62) + 3)),
        (UMul, 31, int(31, 46_340), int(31, 46_340)),
        (UMul, 32, int(32, 65_537), int(32, 65_535)),
        (UMul, 33, int(33, 65_539), int(33, 65_537)),
        (UMul, 63, int(63, (1 << 31) + 1), int(63, (1 << 31) - 1)),
        (UShl, 32, int(32, 0x1234_5678), int(32, 9)),
        (UShl, 33, int(33, 0x1234_5678), int(33, 9)),
        (UShr, 32, int(32, 0xfedc_ba98), int(32, 9)),
        (UShr, 33, int(33, 0x1_fedc_ba98), int(33, 9)),
        (UDiv, 32, int(32, 0xfedc_ba98), int(32, 1_003)),
        (URem, 32, top(32), int(32, 0xffff_fff0)),
        (UDiv, 33, int(33, 0x1_fedc_ba98), int(33, 1_003)),
        (SMul, 32, negative(32, 46_341), int(32, 46_340)),
        (SMul, 33, negative(33, 65_521), int(33, 65_537)),
        (SDiv, 32, negative(32, 1 << 31), int(32, 3)),
        (SMul, 63, negative(63, (1 << 40) + 7), int(63, 3_000_017)),
        (UDiv, 63, int(63, (1 << 62) + 12_345), int(63, 1_000_003)),
        (URem, 48, int(48, (1 << 47) + 99), int(48, 65_539)),
        (SDiv, 63, negative(63, (1 << 50) + 11), int(63, 70_001)),
        (UShl, 63, int(63, 0x1_2345_6789), int(63, 23)),
        (UShr, 63, top(63), int(63, 17)),
        (SShr, 63, negative(63, (1 << 45) + 3), int(63, 33)),
        (And, 63, top(63), every_other(63)),
        (Xor, 63, int(63, 0x0f0f_0f0f_0f0f), every_other(63)),
    ] {
        let oracle = BinaryOpOracle::<Goldilocks>::compile(kind, bits).unwrap();
        let result = mavros_int_semantics::eval(IntOp::from(kind), &lhs, &rhs)
            .value()
            .expect("an accepted pair");
        let what = format!("{kind:?}({lhs:?}, {rhs:?}) at {bits} bits on Goldilocks");
        assert_every_column_is_pinned(&what, oracle.compiled(), &[&lhs, &rhs, &result]);
        assert_no_two_columns_trade_a_limb(&what, oracle.compiled(), &[&lhs, &rhs, &result]);
    }
    let compiled = compile(&[Type::int(63), Type::int(63)], &[Type::int(1)], |e, p| {
        vec![e.cmp(p[0], p[1], CmpKind::ULt)]
    });
    let (lhs, rhs) = (int(63, 5), top(63));
    let less = IntBits::from_u128(1, 1);
    let what = "ULt at 63 bits on Goldilocks";
    assert_every_column_is_pinned(what, &compiled, &[&lhs, &rhs, &less]);
    assert_no_two_columns_trade_a_limb(what, &compiled, &[&lhs, &rhs, &less]);
}

// WIDE INTEGERS
// ================================================================================================

/// The limbed operations agree with the model past what one Goldilocks element carries, held as
/// four, seven (the top one four bits wide) and eight limbs.
#[test]
fn every_limbed_operation_agrees_with_the_model_past_one_element() {
    for bits in [64, 100, 128] {
        check_limbed_matrix(bits);
    }
}

/// [`every_limbed_operation_agrees_with_the_model_past_one_element`] at 256 bits, sixteen limbs.
#[test]
#[ignore = "a sixteen-limb matrix; about four minutes in a debug build"]
fn every_limbed_operation_agrees_with_the_model_at_256_bits() {
    check_limbed_matrix(256);
}

/// No witness column of an operation on wide integers can move alone, and at 64 bits no two can
/// trade a limb's place value. Each operand is a wide constant selected by a witnessed bit, the
/// left one negative for a signed operation, and the program asserts the answer is the model's, so
/// every limb of the answer is constrained. An equality bit would not do: its inverse hint is free
/// where the two sides are equal, which the one-column probe reports.
///
/// A missing range check on an answer limb or a carry is exploited by moving value between limbs
/// through a carry, which neither the one-column probe nor the `2^h` trade reaches; the structural
/// audits in `wide_witness_ints` run on Goldilocks for that.
#[test]
fn every_witness_column_of_a_wide_goldilocks_operation_is_pinned() {
    use BinaryArithOpKind::{
        And, SDiv, SMul, SRem, SShr, UAdd, UDiv, UMul, URem, UShl, UShr, USub,
    };
    let selected = IntBits::from_u128(1, 1);
    for bits in [64usize, 128] {
        let wide = |top: usize, low: u128| IntBits::from_u128(bits, (1u128 << top) + low);
        for kind in [
            UAdd, USub, UMul, UDiv, URem, UShl, UShr, And, SMul, SDiv, SRem, SShr,
        ] {
            // Both operands reach past the lowest limbs: a product's factors meet in its top
            // limbs, a left shift by 21 lands in its top limb, and every other operation's left
            // operand reaches the top limb.
            let (lhs, rhs) = match kind {
                UMul | SMul => (wide(bits / 2 - 2, 12_345), wide(bits / 2 - 3, 999)),
                UShl => (wide(bits - 24, 12_345), IntBits::from_u128(bits, 21)),
                UShr | SShr => (wide(bits - 6, 12_345), IntBits::from_u128(bits, 21)),
                _ => (
                    wide(bits - 6, (1 << 40) + 12_345),
                    wide(bits - 9, (1 << 30) + 999),
                ),
            };
            let lhs = if matches!(kind, SMul | SDiv | SRem | SShr) {
                IntBits::from_signed(bits, &-SignedValue::from(BigUint::from(&lhs)))
            } else {
                lhs
            };
            let answer = mavros_int_semantics::eval(IntOp::from(kind), &lhs, &rhs)
                .value()
                .expect("an accepted pair");

            let (left, right, expected) = (lhs.clone(), rhs.clone(), answer);
            let ssa = main_program_in(
                FieldConfig::goldilocks(),
                &[Type::int(1), Type::int(1)],
                &[Type::int(1)],
                move |e, p| {
                    let zero = e.int_const(IntBits::zero(bits));
                    let left = e.int_const(left);
                    let lhs = e.select(p[0], left, zero);
                    let right = e.int_const(right);
                    let rhs = e.select(p[1], right, zero);
                    let result = e.bin(kind, lhs, rhs);
                    let expected = e.int_const(expected);
                    e.assert_eq(result, expected);
                    vec![p[0]]
                },
            );
            let compiled = Compiled::<Goldilocks>::compile(ssa).unwrap_or_else(|error| {
                panic!("{kind:?} at {bits} bits failed on Goldilocks: {error}")
            });
            let what = format!("{kind:?}({lhs:?}, {rhs:?}) at {bits} bits on Goldilocks");
            let inputs = [&selected, &selected, &selected];
            assert_every_column_is_pinned(&what, &compiled, &inputs);
            if bits == 64 {
                assert_no_two_columns_trade_a_limb(&what, &compiled, &inputs);
            }
        }
    }
}
