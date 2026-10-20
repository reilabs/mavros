//! Goldilocks programs built directly as SSA, compiled to a Goldilocks R1CS and VM bytecode, run on
//! the VM, and checked against the field and against the integer model.
//!
//! Every program here also passes the harness' AD check. None has a WASM lane: the runtime has no
//! Goldilocks instance.

use mavros_artifacts::{FieldConfig, Goldilocks, InputValueOrdered, field::goldilocks::MODULUS};
use mavros_compiler::compiler::ssa::{
    ValueId,
    hlssa::{
        BinaryArithOpKind, CastTarget, CmpKind, Endianness, Type,
        builder::{HLBlockEmitter, HLEmitter as _},
    },
};
use mavros_int_semantics::{CmpOp, IntBits, IntOp, corners};

use crate::{
    assert_every_column_is_pinned, assert_no_two_columns_trade_a_limb, corner_pairs,
    harness::{BinaryOpOracle, Compiled, Verdict, input_block_in, main_program_in},
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

/// From 32 bits a product is a schoolbook of 32-bit witness limbs, whose partial product and carry
/// do not fit a Goldilocks element, so the lowering refuses it (Layer 6). This pins the refusal
/// until the lowering supports the product.
#[test]
#[should_panic(expected = "schoolbook product of witness limbs")]
fn a_32_bit_product_is_refused_on_goldilocks() {
    let _ = BinaryOpOracle::<Goldilocks>::compile(BinaryArithOpKind::UMul, 32);
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
    use BinaryArithOpKind::{UAdd, UMul, USub};
    let int = |bits: usize, value: u128| IntBits::from_u128(bits, value);
    let top = |bits: usize| IntBits::all_ones(bits);
    for (kind, bits, lhs, rhs) in [
        (UAdd, 62, int(62, 1 << 61), int(62, (1 << 61) - 1)),
        (UAdd, 63, int(63, 1 << 62), int(63, (1 << 62) - 1)),
        (USub, 63, top(63), int(63, (1 << 62) + 3)),
        (UMul, 31, int(31, 46_340), int(31, 46_340)),
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
