//! The oracle's test suite: what the pipeline must agree with the model about.

mod harness;

use harness::{BinaryOpOracle, Compiled, Verdict, bytecode_listing, input_block, main_program};
use mavros_artifacts::{Field, FieldConfig, InputValueOrdered};
use mavros_compiler::{
    compiler::ssa::{
        SourceLocation, SourcePosition,
        hlssa::{
            BinaryArithOpKind, Blob, CastTarget, CmpKind, Constant, HLSSA, OpCode,
            SequenceTargetType, Type,
            builder::{HLEmitter as _, HLSSABuilder},
        },
    },
    driver::{Driver, Error as DriverError},
};
use mavros_int_semantics::{IntBits, IntOp, corners};
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

/// Sweeps one operation at one width, reporting the first disagreement.
fn sweep(kind: BinaryArithOpKind, bits: usize) {
    let op = IntOp::from(kind);
    let oracle = BinaryOpOracle::new(kind, bits, bits)
        .unwrap_or_else(|e| panic!("{kind:?} at {bits} bits failed to compile: {e}"));
    let pairs = corner_pairs(op, bits);
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

/// The current cap, which is the two-cell lane: `128` is where the VM stops using one frame cell
/// per integer and `witness_bitwise` stops using one field element, so it is the only width at
/// which today's code already does anything limb-wise.
#[test]
fn arithmetic_agrees_at_the_current_cap() {
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
        (BinaryArithOpKind::UMul, 32),
        (BinaryArithOpKind::UDiv, 16),
        (BinaryArithOpKind::SShr, 8),
    ] {
        let oracle = BinaryOpOracle::new(kind, bits, bits).unwrap();
        let (lhs, rhs) = (IntBits::from_u128(bits, 7), IntBits::from_u128(bits, 3));
        let expected = mavros_int_semantics::eval(IntOp::from(kind), &lhs, &rhs)
            .value()
            .expect("7 op 3 is accepted for each operation swept here");
        let verdict = oracle
            .compiled()
            .run(&input_block(&[&lhs, &rhs, &expected]));
        let witness = verdict
            .witness()
            .unwrap_or_else(|| panic!("{kind:?} at {bits} bits: {verdict:?}"))
            .to_vec();

        for column in 0..witness.len() {
            let mut perturbed = witness.clone();
            perturbed[column] += Field::from(1u64);
            assert!(
                oracle
                    .compiled()
                    .first_unsatisfied_constraint(&perturbed)
                    .is_some(),
                "{kind:?} at {bits} bits: column {column} of {} is unconstrained — \
                 adding one to it left every constraint satisfied",
                witness.len()
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

/// A 127-bit witness `*` is the one width whose product the field cannot tell from a residue and
/// which has no fallback: `int128` splits into two limbs, and `int126` and below fit one product.
#[test]
fn a_witness_multiply_at_a_hundred_and_twenty_seven_bits_is_refused() {
    contains_all(
        &refusal_for(BinaryArithOpKind::UMul, 127, 127),
        &[
            "error: a witnessed int127 multiplication is not supported",
            "= note: a witnessed multiplication is a single field product",
        ],
    );
}

/// A 129-bit multiplication is refused for its witnessed operands and for nothing else.
///
/// The program the oracle builds carries the operation twice: witness taint inference splits the
/// entry point by context, so a **pure** copy of the multiply sits beside the witnessed one. Only
/// the witnessed copy has no lowering, and that is what makes this the closest thing to end-to-end
/// evidence that the pure multiply's overflow check is built at this width.
///
/// The count is the assertion. The witness refusal's own second note tells the reader the operation
/// is supported outside the witness domain, and a second refusal saying it is not would contradict
/// it in the same compile.
#[test]
fn a_wide_multiplication_is_refused_for_its_witness_operands_alone() {
    let refusal = refusal_for(BinaryArithOpKind::UMul, 129, 129);
    contains_all(
        &refusal,
        &[
            "error: a witnessed int129 multiplication is not supported",
            "= note: the same operation is supported at this width outside the witness domain",
        ],
    );
    assert_eq!(
        refusal.matches("multiplication is not supported").count(),
        1,
        "the pure copy of the same multiply is lowered, not refused:\n{refusal}"
    );
}

/// A bitwise operation falls off `lower_binary_bitwise`'s two limb cases — 64 and 128 — into the
/// fall-through that spreads at the operand's own width. The VM's spread stops at 32, so every
/// width in `33..=63` lands on this.
#[test]
fn a_bitwise_operation_between_the_word_sizes_is_refused() {
    contains_all(
        &refusal_for(BinaryArithOpKind::Xor, 40, 40),
        &[
            "error: a witnessed int40 bitwise xor is not supported",
            "= note: a witnessed bitwise operation is decomposed limb-wise at int64 and int128",
        ],
    );
}

/// The same fall-through above 64 bits reaches a _different_ ceiling: the spread of a
/// `65..=127`-bit operand is a `130..=254`-bit value, and `Unspread` refuses anything above
/// `int128`. Both bands are one gap in the same lowering.
#[test]
fn a_bitwise_operation_between_a_word_and_a_double_word_is_refused() {
    contains_all(
        &refusal_for(BinaryArithOpKind::Xor, 96, 96),
        &[
            "error: a witnessed int96 bitwise xor is not supported",
            "= note: a witnessed bitwise operation is decomposed limb-wise at int64 and int128",
        ],
    );
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

/// `main(a: int64, i: int32) -> int64 { [a as int(bits), 3][i] as int64 }`.
///
/// The index is a parameter, so the lookup cannot be folded away before validation sees the
/// sequence it reads, and it is a witness, so the lookup is one the tape has to address.
fn program_indexing_a_sequence_of(bits: usize) -> HLSSA {
    main_program(
        &[Type::int(64), Type::int(32)],
        &[Type::int(64)],
        move |e, params| {
            let wide = e.cast_to(CastTarget::Int(bits), params[0]);
            let other = e.int_const(IntBits::from_u128(bits, 3));
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

/// A wide element of a sequence is refused, and names the lane that cannot read it.
///
/// **Both bands, and the second is the one a bound stated as the field alone would miss.** An
/// element has to become one field element, which is the modulus' question; and the VM's lookup
/// tape addresses it with a tag naming a cell count, of which it has `ELEM_WORD` for a cell and
/// `ELEM_U128` for a pair and no third. So the tape stops at the double lane, far below the
/// modulus, and a width between the two reaches `lookup_elem_kind` with no tag to take.
#[test]
fn a_sequence_element_wider_than_the_tape_is_refused() {
    // Past the modulus, and between the tape and the modulus.
    for bits in [first_width_past_the_field(), 200] {
        let Err(DriverError::Refused(diagnostics)) =
            Compiled::new(program_indexing_a_sequence_of(bits))
        else {
            panic!("an int{bits} sequence element is refused rather than compiled");
        };
        assert!(
            diagnostics
                .iter()
                .any(|d| d.message() == format!("an int{bits} sequence element is not supported")),
            "int{bits}: {diagnostics:?}"
        );
    }
}

/// And the bound is exact rather than conservative: the widest element the tape names still
/// compiles, so what is refused above is a frontier and not a blanket.
#[test]
fn the_widest_element_the_tape_names_still_compiles() {
    let _ = Compiled::new(program_indexing_a_sequence_of(128))
        .expect("an int128 element is one the tape has a tag for");
}

/// The same refusal for a sequence built from a **blob**, which states its element type under a
/// name of its own.
///
/// Three opcodes make a sequence and each spells its element type differently, so a rule that
/// matched on two of them would let the third through to `lookup_elem_kind`, which has no tag to
/// take, or to the multi-cell representation, which does not reach inside a sequence.
#[test]
fn a_blob_sequence_of_wide_elements_is_refused() {
    for bits in [200usize, first_width_past_the_field()] {
        let ssa = main_program(&[Type::int(32)], &[Type::int(64)], move |e, params| {
            let blob = e.emit_constant(Constant::Blob(Blob::new(
                Type::int(bits),
                vec![
                    Constant::Int(IntBits::from_u128(bits, 3)),
                    Constant::Int(IntBits::from_u128(bits, 4)),
                ],
            )));
            let array = e.mk_seq_of_blob(Type::int(bits), blob);
            let element = e.array_get(array, params[0]);
            vec![e.cast_to(CastTarget::Int(64), element)]
        });

        let Err(DriverError::Refused(diagnostics)) = Compiled::new(ssa) else {
            panic!("an int{bits} blob element is refused rather than compiled");
        };
        assert!(
            diagnostics
                .iter()
                .any(|d| d.message() == format!("an int{bits} sequence element is not supported")),
            "int{bits}: {diagnostics:?}"
        );
    }
}

/// The third lane: the same programs compiled to WASM and run under wasmtime.
///
/// The corpus already checks this lane byte-identically at every width Noir can name, which is
/// strictly stronger than anything here — and it is silent above 128 bits, because no corpus
/// program has a width there. So this is the only thing that says the wide field boundary and the
/// multi-cell representation compute the same answers in the compiled module as in the interpreter
/// and the constraint system.
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

    assert!(
        ran == 6 || !compiled.wasm_is_available(),
        "the WASM lane was available and ran {ran} of 6"
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
