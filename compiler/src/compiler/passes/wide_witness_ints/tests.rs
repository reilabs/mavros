use super::*;
use mavros_artifacts::FieldId;
use mavros_int_semantics::int_bits::HOST_LIMB_BITS;

use crate::compiler::{
    analysis::types::Types,
    passes::shared::limbs::narrow_int_bits,
    ssa::hlssa::{
        MAX_SPREAD_INPUT_BITS, MAX_SUPPORTED_INT_BITS,
        builder::{HLEmitter, HLSSABuilder},
    },
};

fn bn254() -> FieldConfig {
    FieldConfig::bn254()
}

/// The fields the structural audits run on.
fn fields() -> [FieldConfig; 2] {
    [bn254(), FieldConfig::goldilocks()]
}

/// Two widths past what one element of `field` holds: one with a narrow top limb, then one with
/// a full one.
fn limbed_widths(field: FieldConfig) -> [usize; 2] {
    match field.id() {
        FieldId::Bn254 => [254, 320],
        FieldId::Goldilocks => [100, 64],
    }
}

/// The threshold is where a value stops having a field element, which is what forces limbs.
///
/// It is deliberately **not** the narrower width the single-cell lowerings stop at: between the
/// two a value is still one element, and an operation the single cell cannot take there is cut
/// into limbs by the operation that needs them.
#[test]
fn the_representation_threshold_is_where_an_element_stops_being_injective() {
    let field = bn254();

    assert_eq!(multi_cell_int_bits(field), widest_injective_int_bits(field));
    assert!(multi_cell_int_bits(field) > narrow_int_bits(field));
}

/// A limb's declared width is the width it is held to, and the top one carries the remainder.
#[test]
fn the_top_limb_carries_the_remainder() {
    assert_eq!(limb_widths(320, 64), vec![64, 64, 64, 64, 64]);
    assert_eq!(limb_widths(300, 64), vec![64, 64, 64, 64, 44]);
    assert_eq!(limb_widths(255, 64), vec![64, 64, 64, 63]);
    assert_eq!(limb_widths(64, 64), vec![64]);
}

/// Only a witnessed integer past the threshold becomes limbs.
#[test]
fn a_narrow_or_pure_integer_is_left_alone() {
    let field = bn254();
    let wide = multi_cell_int_bits(field) + 1;

    assert_eq!(limb_types(&Type::int(wide), field).len(), 1);
    assert_eq!(
        limb_types(
            &Type::witness_of(Type::int(multi_cell_int_bits(field))),
            field
        )
        .len(),
        1
    );
    assert_eq!(
        limb_types(&Type::witness_of(Type::int(wide)), field).len(),
        wide.div_ceil(witness_limb_bits(field))
    );
}

/// A reference and a sequence both expand, and a sequence expands whether or not its element
/// is witnessed.
///
/// A reference is a place, and every limb needs one. A sequence is transposed, so the count is
/// the element's limbs, not the sequence's length. A **pure** wide element counts too, the one
/// place [`element_limb_types`] differs from [`limb_types`]. A narrow element expands in none
/// of these, which keeps a `[u128; n]` on bn254 off this path entirely.
#[test]
fn a_reference_and_a_sequence_both_expand_by_the_element() {
    let field = bn254();
    let wide = multi_cell_int_bits(field) + 1;
    let expected = wide.div_ceil(witness_limb_bits(field));
    assert!(expected > 1, "the width has to span limbs to be a test");

    assert_eq!(
        limb_types(&Type::witness_of(Type::int(wide)).ref_of(), field).len(),
        expected
    );
    assert_eq!(
        limb_types(&Type::witness_of(Type::int(wide)).array_of(4), field).len(),
        expected
    );
    assert_eq!(
        limb_types(&Type::int(wide).array_of(4), field).len(),
        expected,
        "a pure wide element is transposed too, or a constant sequence could not be read"
    );
    assert_eq!(
        limb_types(&Type::slice_of(Type::int(wide)), field).len(),
        expected
    );

    // A narrow element, at the widest width the double lane holds, stays one sequence.
    assert_eq!(
        limb_types(
            &Type::witness_of(Type::int(2 * HOST_LIMB_BITS)).array_of(4),
            field
        )
        .len(),
        1
    );
}

/// `main(wide) { jmp block_1(wide) }` — a wide value crossing a block boundary.
fn program_passing_a_wide_value_across_a_block(bits: usize) -> HLSSA {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main = ssa.get_unique_entrypoint_id();
    let mut builder = HLSSABuilder::new(&mut ssa);
    let carried = builder.fresh_value();
    let start = builder.fresh_value();
    builder.modify_function(main, |fb| {
        let entry = fb.function.get_entry_id();
        let next = {
            let mut editor = fb.block(entry);
            let (next, block) = editor.add_block();
            block.push_parameter(carried, Type::witness_of(Type::int(bits)));
            block.set_terminator(Terminator::Return(vec![]));
            next
        };
        let entry_type = Type::witness_of(Type::int(bits));
        fb.function
            .get_block_mut(entry)
            .push_parameter(start, entry_type);
        fb.block(entry)
            .set_terminator(Terminator::Jmp(next, vec![start]));
    });
    ssa
}

/// Every block parameter carrying a wide value becomes one parameter per limb, and the jump
/// that feeds it carries one argument per limb.
///
/// A memoized table could get this silently wrong: limbs re-derived at the consumer from the
/// pure shadow, a hint, would leave producer and consumer holding unrelated witnesses.
#[test]
fn a_block_parameter_becomes_one_parameter_per_limb() {
    let bits = 320usize;
    let mut ssa = program_passing_a_wide_value_across_a_block(bits);

    let expected = limb_widths(bits, witness_limb_bits(bn254()));
    run_pass(&mut ssa);

    let main = ssa.get_unique_entrypoint_id();
    let function = ssa.get_function(main);
    let widths: Vec<usize> = function
        .get_block(BlockId(1))
        .get_parameters()
        .map(|(_, ty)| match &ty.strip_witness().expr {
            TypeExpr::Int(bits) => *bits,
            other => panic!("a limb is an integer, got {other:?}"),
        })
        .collect();

    assert_eq!(widths, expected);

    // And the jump that feeds it carries one argument per limb. A parameter list and argument
    // list that disagree would surface only in an arity check far downstream of the cause.
    let Some(Terminator::Jmp(_, args)) = function.get_block(BlockId(0)).get_terminator() else {
        panic!("the entry block jumps to the block carrying the value");
    };
    assert_eq!(args.len(), expected.len());
}

/// `main(a: WitnessOf(int(from))) -> WitnessOf(int(to)) { a as int(to) }`.
fn program_widening(from: usize, to: usize) -> HLSSA {
    program_widening_in(bn254(), from, to)
}

/// [`program_widening`] over `field`.
fn program_widening_in(field: FieldConfig, from: usize, to: usize) -> HLSSA {
    let mut ssa = HLSSA::with_main_in(field, "main".to_string());
    let main = ssa.get_unique_entrypoint_id();
    let value = ssa.fresh_value();
    let result = ssa.fresh_value();
    let mut builder = HLSSABuilder::new(&mut ssa);
    builder.modify_function(main, |fb| {
        fb.function.add_return_type(Type::witness_of(Type::int(to)));
        let entry = fb.function.get_entry_id();
        fb.function
            .get_block_mut(entry)
            .push_parameter(value, Type::witness_of(Type::int(from)));
        let mut block = fb.test_block(entry);
        block.emit(OpCode::Cast {
            result,
            value,
            target: CastTarget::Int(to),
        });
        block.terminate_return(vec![result]);
    });
    ssa
}

/// `main(value: int(bits)) -> WitnessOf<int(bits)> { value as witness }` over `field`, a
/// **pure** wide value injected into the representation.
fn program_injecting(field: FieldConfig, bits: usize) -> HLSSA {
    let mut ssa = HLSSA::with_main_in(field, "main".to_string());
    let main = ssa.get_unique_entrypoint_id();
    let value = ssa.fresh_value();
    let result = ssa.fresh_value();
    let mut builder = HLSSABuilder::new(&mut ssa);
    builder.modify_function(main, |fb| {
        fb.function
            .add_return_type(Type::witness_of(Type::int(bits)));
        let entry = fb.function.get_entry_id();
        fb.function
            .get_block_mut(entry)
            .push_parameter(value, Type::int(bits));
        let mut block = fb.test_block(entry);
        block.emit(OpCode::Cast {
            result,
            value,
            target: CastTarget::WitnessOf,
        });
        block.terminate_return(vec![result]);
    });
    ssa
}

/// A **pure** value injected into the representation is bounded, and that bound is all there
/// is.
///
/// Unlike [`Rewriter::decompose`], [`Rewriter::inject`] has no prior value to tie its limbs to,
/// so the per-limb range check is the **only** bound: as `2^h` is invertible mod `p`, an
/// unbounded limb lets a prover choose any value. No honest witness can see this, so it is
/// checked structurally.
#[test]
fn a_pure_value_injected_into_the_representation_is_bounded_per_limb() {
    for field in fields() {
        let id = field.id();
        // Above the representation threshold, so the value is held as limbs, and not a whole
        // number of them, so the top limb is bounded at its width rather than a full limb's.
        let bits = match field.id() {
            FieldId::Bn254 => 328,
            FieldId::Goldilocks => 100,
        };
        let mut ssa = program_injecting(field, bits);
        run_pass(&mut ssa);

        let ops = emitted(&ssa);
        let bounded: Vec<usize> = ops
            .iter()
            .filter_map(|op| match op {
                OpCode::Rangecheck { max_bits, .. } => Some(*max_bits),
                _ => None,
            })
            .collect();

        assert_eq!(
            bounded,
            limb_widths(bits, witness_limb_bits(field)),
            "{id:?}: each injected limb is bounded at its declared width"
        );
        assert!(
            bounded.len() > 1,
            "{id:?}: the value has to span more than one limb to be a test"
        );
        // And there is deliberately no reconstruction: a hint has nothing to be tied back to.
        assert!(
            !ops.iter().any(|op| matches!(op, OpCode::Constrain { .. })),
            "{id:?}: an injection ties nothing back; if it does, this test is checking the wrong gadget"
        );
    }
}

/// `main(shadow, table, index, flag)`, the part of `witness_array_access`'s output that this
/// arm reads: a hint out of the pure shadow, across the field boundary, into a column, and a
/// lookup tying that column to the table.
fn program_reading_a_sequence_at_a_witness_index(bits: usize, slots: usize) -> HLSSA {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main = ssa.get_unique_entrypoint_id();
    let (shadow, table) = (ssa.fresh_value(), ssa.fresh_value());
    let (hint_index, index, flag) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let (hint, image, column) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let mut builder = HLSSABuilder::new(&mut ssa);
    builder.modify_function(main, |fb| {
        let entry = fb.function.get_entry_id();
        {
            let block = fb.function.get_block_mut(entry);
            // The hint comes off the pure shadow and the lookup pins it against the witnessed
            // table, which is the pair `witness_array_access` leaves behind.
            block.push_parameter(shadow, Type::int(bits).array_of(slots));
            block.push_parameter(hint_index, Type::int(32));
            block.push_parameter(table, Type::witness_of(Type::int(bits)).array_of(slots));
            block.push_parameter(index, Type::witness_of(Type::int(32)));
            block.push_parameter(flag, Type::witness_of(Type::field()));
        }
        let mut block = fb.test_block(entry);
        block.emit(OpCode::ArrayGet {
            result: hint,
            array: shadow,
            index: hint_index,
        });
        block.emit(OpCode::Cast {
            result: image,
            value: hint,
            target: CastTarget::Field,
        });
        block.emit(OpCode::WriteWitness {
            result: Some(column),
            value: image,
            pinned: false,
        });
        block.emit(OpCode::Lookup {
            target: LookupTarget::Array(table),
            args: vec![index, column],
            flag,
        });
        block.terminate_return(vec![]);
    });
    ssa
}

/// A transposed read is **one lookup per limb**, all at the same index.
///
/// No run can see the count: a limb the tape never pins is read by nothing else either, so it
/// is eliminated rather than left free, and `every_column_of_a_wide_sequence_is_pinned` still
/// passes. The shared index is the other half: `k` lookups at `k` **different** indices would
/// pin `k` limbs of no single element.
#[test]
fn a_transposed_read_is_one_lookup_per_limb_at_one_index() {
    let bits = 320usize;
    let mut ssa = program_reading_a_sequence_at_a_witness_index(bits, 2);
    run_pass(&mut ssa);

    let lookups: Vec<(ValueId, ValueId, ValueId)> = emitted(&ssa)
        .into_iter()
        .filter_map(|op| match op {
            OpCode::Lookup {
                target: LookupTarget::Array(array),
                args,
                ..
            } => Some((array, args[0], args[1])),
            _ => None,
        })
        .collect();

    let expected = limb_widths(bits, witness_limb_bits(bn254())).len();
    assert!(expected > 1, "the element has to span limbs to be a test");
    assert_eq!(
        lookups.len(),
        expected,
        "an int{bits} element is {expected} limbs and each one needs its own lookup"
    );

    let indices: Vec<ValueId> = lookups.iter().map(|(_, index, _)| *index).collect();
    assert!(
        indices.windows(2).all(|pair| pair[0] == pair[1]),
        "every limb is read at the same index, or the limbs are not one element's"
    );

    let mut tables: Vec<ValueId> = lookups.iter().map(|(array, ..)| *array).collect();
    let mut results: Vec<ValueId> = lookups.iter().map(|(.., result)| *result).collect();
    tables.sort();
    tables.dedup();
    results.sort();
    results.dedup();
    assert_eq!(tables.len(), expected, "one table per limb sequence");
    assert_eq!(results.len(), expected, "one column per limb");
}

/// The field image of a pure read is cast from the limbs the read produced.
///
/// A pure read is recombined for the users that want the whole value, and the image a
/// witness-indexed lookup moves through wants the limbs again. Splitting the recombined value
/// would use logical right shifts and narrowings at the full width, so the arm hands back the
/// limbs it has: every `Field` cast takes one of the read's own results, and nothing in between
/// shifts right or divides at the value's width.
#[test]
fn the_field_image_of_a_pure_read_is_cast_from_the_limbs_it_was_read_as() {
    let bits = 320usize;
    let mut ssa = program_reading_a_sequence_at_a_witness_index(bits, 2);
    run_pass(&mut ssa);
    let ops = emitted(&ssa);

    let read: Vec<ValueId> = ops
        .iter()
        .filter_map(|op| match op {
            OpCode::ArrayGet { result, .. } => Some(*result),
            _ => None,
        })
        .collect();
    assert_eq!(
        read.len(),
        limb_widths(bits, witness_limb_bits(bn254())).len(),
        "one read per limb sequence"
    );

    let imaged: Vec<ValueId> = ops
        .iter()
        .filter_map(|op| match op {
            OpCode::Cast {
                value,
                target: CastTarget::Field,
                ..
            } => Some(*value),
            _ => None,
        })
        .collect();
    assert_eq!(
        imaged, read,
        "the field image is cast from the read's limbs, not split out of the whole value"
    );
    assert!(
        !ops.iter().any(|op| matches!(
            op,
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::UShr | BinaryArithOpKind::UDiv,
                ..
            }
        )),
        "no logical right shift or division re-splits the recombined value into limbs"
    );
}

/// Every emitted opcode of one function, in order.
fn emitted(ssa: &HLSSA) -> Vec<OpCode> {
    let main = ssa.get_unique_entrypoint_id();
    let function = ssa.get_function(main);
    function
        .get_block(function.get_entry_id())
        .get_instructions()
        .cloned()
        .collect()
}

/// A value entering the representation is witnessed and bounded once per limb, and tied back to
/// its source once.
///
/// A structural audit, not a behavioral one, because neither half is visible to an honest
/// witness. Without the range check a prover can move value between limbs while the sum still
/// holds; without the reconstruction the limbs are not this value's at all. In that second case
/// the limb loses its only consumer, so it is eliminated along with its witness column, and a
/// column-by-column perturbation has nothing left to find.
#[test]
fn a_value_entering_the_representation_is_witnessed_bounded_and_tied_back() {
    for field in fields() {
        let id = field.id();
        // A source that is not a whole number of limbs, so the top limb is bounded at its width
        // rather than a full limb's.
        let (from, to) = match field.id() {
            FieldId::Bn254 => (200, 320),
            FieldId::Goldilocks => (50, 100),
        };
        let mut ssa = program_widening_in(field, from, to);
        run_pass(&mut ssa);

        let ops = emitted(&ssa);
        let limbs = from.div_ceil(witness_limb_bits(field));
        assert!(
            limbs > 1,
            "{id:?}: the source has to span more than one limb to be a test"
        );

        let written = ops
            .iter()
            .filter(|op| matches!(op, OpCode::WriteWitness { .. }))
            .count();
        let bounded: Vec<usize> = ops
            .iter()
            .filter_map(|op| match op {
                OpCode::Rangecheck { max_bits, .. } => Some(*max_bits),
                _ => None,
            })
            .collect();
        let tied = ops
            .iter()
            .filter(|op| matches!(op, OpCode::Constrain { .. }))
            .count();

        assert_eq!(
            written, limbs,
            "{id:?}: one witness column per limb of the source"
        );
        assert_eq!(
            bounded,
            limb_widths(from, witness_limb_bits(field)),
            "{id:?}: each limb is bounded at its declared width"
        );
        assert_eq!(
            tied, 1,
            "{id:?}: one reconstruction, tying the limbs to their source"
        );
    }
}

/// A witnessed field element read as a wide integer is decomposed at the field's width and held
/// below the modulus.
///
/// Structural: an honest witness is always the canonical decomposition, so only the constraints
/// tell `v` from `v + p`, through the borrow-forced chain against the modulus limbs. Each of
/// those limbs must reach an emitted instruction as a constant, which a dropped chain or a
/// wrong bound would not leave. The chain's top limb must mint no borrow column: a witnessed
/// top borrow lets the prover answer zero, which admits `v + p`.
#[test]
fn a_field_read_as_a_wide_integer_is_held_below_the_modulus() {
    for field in fields() {
        let id = field.id();
        let field_bits = field.field_bit_size() as usize;
        let mut ssa = HLSSA::with_main_in(field, "main".to_string());
        let main = ssa.get_unique_entrypoint_id();
        let value = ssa.fresh_value();
        let result = ssa.fresh_value();
        HLSSABuilder::new(&mut ssa).modify_function(main, |fb| {
            fb.function
                .add_return_type(Type::witness_of(Type::int(300)));
            let entry = fb.function.get_entry_id();
            fb.function
                .get_block_mut(entry)
                .push_parameter(value, Type::witness_of(Type::field()));
            let mut block = fb.test_block(entry);
            block.emit(OpCode::Cast {
                result,
                value,
                target: CastTarget::Int(300),
            });
            block.terminate_return(vec![result]);
        });
        run_pass(&mut ssa);

        let ops = emitted(&ssa);
        let bounded: Vec<usize> = ops
            .iter()
            .filter_map(|op| match op {
                OpCode::Rangecheck { max_bits, .. } => Some(*max_bits),
                _ => None,
            })
            .collect();
        let widths = limb_widths(field_bits, witness_limb_bits(field));
        // The decomposition bounds each limb at its width; the chain bounds each borrow but the
        // top one at one bit and each difference limb at its width.
        let mut expected = widths.clone();
        for (index, width) in widths.iter().enumerate() {
            if index + 1 < widths.len() {
                expected.push(1);
            }
            expected.push(*width);
        }
        assert_eq!(
            bounded, expected,
            "{id:?}: the limbs, the borrows below the top and the differences are each bounded \
             once, and the top borrow is forced rather than witnessed"
        );
        let written = ops
            .iter()
            .filter(|op| matches!(op, OpCode::WriteWitness { .. }))
            .count();
        assert_eq!(
            written,
            2 * widths.len() - 1,
            "{id:?}: one column per limb and per borrow below the top"
        );

        let constants: Vec<IntBits> = ops
            .iter()
            .flat_map(|op| op.get_inputs().copied().collect::<Vec<_>>())
            .filter_map(|input| match ssa.get_const(input).as_deref() {
                Some(Constant::Int(pattern)) => Some(pattern.clone()),
                _ => None,
            })
            .collect();
        let modulus = IntBits::from_biguint(
            field_bits,
            &field_modulus(field)
                .to_biguint()
                .expect("a modulus is positive"),
        );
        // Counted, since a limb value can recur: goldilocks' limbs are
        // `1, 0, 2^16 - 1, 2^16 - 1`.
        let limbs: Vec<IntBits> = widths
            .iter()
            .enumerate()
            .map(|(index, width)| modulus.bit_range(index * witness_limb_bits(field), *width))
            .collect();
        for (index, limb) in limbs.iter().enumerate() {
            let count = |values: &[IntBits]| values.iter().filter(|v| *v == limb).count();
            assert!(
                count(&constants) >= count(&limbs),
                "{id:?}: limb {index} of the modulus, {limb:?}, bounds nothing"
            );
        }
    }
}

/// The limbs a widening does not reach are the zero constant, which is pinned by being one.
///
/// So a widening mints witnesses for the **source's** limbs and no more: padding a 320-bit
/// target from a 128-bit source costs three constants, not three columns.
#[test]
fn the_limbs_above_the_source_cost_no_witness() {
    let mut narrow_source = program_widening(128, 320);
    let mut wider_source = program_widening(192, 320);
    run_pass(&mut narrow_source);
    run_pass(&mut wider_source);

    let count = |ssa: &HLSSA| {
        emitted(ssa)
            .iter()
            .filter(|op| matches!(op, OpCode::WriteWitness { .. }))
            .count()
    };
    assert_eq!(count(&narrow_source), 2);
    assert_eq!(count(&wider_source), 3);
}

/// `main(lhs, rhs: WitnessOf<int(bits)>) { op }`, returning the operation's result where it
/// has a type.
fn program_chaining(
    bits: usize,
    returns: Option<Type>,
    op: impl FnOnce(ValueId, ValueId, ValueId) -> OpCode,
) -> HLSSA {
    program_chaining_in(bn254(), bits, returns, op)
}

/// [`program_chaining`] over `field`.
fn program_chaining_in(
    field: FieldConfig,
    bits: usize,
    returns: Option<Type>,
    op: impl FnOnce(ValueId, ValueId, ValueId) -> OpCode,
) -> HLSSA {
    let mut ssa = HLSSA::with_main_in(field, "main".to_string());
    let main = ssa.get_unique_entrypoint_id();
    let (lhs, rhs, result) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let mut builder = HLSSABuilder::new(&mut ssa);
    builder.modify_function(main, |fb| {
        let entry = fb.function.get_entry_id();
        for value in [lhs, rhs] {
            fb.function
                .get_block_mut(entry)
                .push_parameter(value, Type::witness_of(Type::int(bits)));
        }
        let returned = match returns {
            Some(returned) => {
                fb.function.add_return_type(returned);
                vec![result]
            }
            None => vec![],
        };
        let mut block = fb.test_block(entry);
        block.emit(op(result, lhs, rhs));
        block.terminate_return(returned);
    });
    ssa
}

fn difference(bits: usize) -> HLSSA {
    difference_in(bn254(), bits)
}

/// [`difference`] over `field`.
fn difference_in(field: FieldConfig, bits: usize) -> HLSSA {
    program_chaining_in(
        field,
        bits,
        Some(Type::witness_of(Type::int(bits))),
        |result, lhs, rhs| OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::USub,
            result,
            lhs,
            rhs,
        },
    )
}

/// Every carry the chain witnesses is a bit, and every limb of the answer is bounded at its
/// width.
///
/// Structural, because perturbing a column by one cannot test the carry checks: a carry of two
/// moves its limb's answer by `2^w` and fails that limb's range check anyway. The bit check
/// stops a carry that is a large field element, which moves value between two neighboring limbs
/// while each limb still passes.
#[test]
fn every_carry_is_a_bit_and_every_limb_of_the_chain_is_bounded() {
    for field in fields() {
        let id = field.id();
        let h = witness_limb_bits(field);
        for bits in limbed_widths(field) {
            let k = limb_widths(bits, h).len();
            let sum = |kind| {
                program_chaining_in(
                    field,
                    bits,
                    Some(Type::witness_of(Type::int(bits))),
                    move |result, lhs, rhs| OpCode::BinaryArithOp {
                        kind,
                        result,
                        lhs,
                        rhs,
                    },
                )
            };
            let ordering = program_chaining_in(
                field,
                bits,
                Some(Type::witness_of(Type::int(1))),
                |result, lhs, rhs| OpCode::Cmp {
                    kind: CmpKind::ULt,
                    result,
                    lhs,
                    rhs,
                },
            );
            let assertion =
                program_chaining_in(field, bits, None, |_, lhs, rhs| OpCode::AssertCmp {
                    kind: CmpKind::ULt,
                    lhs,
                    rhs,
                });

            // A checked operation has no carry out of its top limb, and an ordering witnesses
            // the one it has; an asserted ordering fixes it, so witnesses none either.
            for (what, mut ssa, carries) in [
                ("sum", sum(BinaryArithOpKind::UAdd), k - 1),
                ("difference", sum(BinaryArithOpKind::USub), k - 1),
                ("ordering", ordering, k),
                ("assertion", assertion, k - 1),
            ] {
                run_pass(&mut ssa);
                let ops = emitted(&ssa);

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
                let bits_checked: Vec<ValueId> = ops
                    .iter()
                    .filter_map(|op| match op {
                        OpCode::Rangecheck { value, max_bits: 1 } => Some(*value),
                        _ => None,
                    })
                    .collect();
                let limbs_checked: Vec<usize> = ops
                    .iter()
                    .filter_map(|op| match op {
                        OpCode::Rangecheck { max_bits, .. } if *max_bits > 1 => Some(*max_bits),
                        _ => None,
                    })
                    .collect();

                assert_eq!(
                    written.len(),
                    carries,
                    "{id:?} int{bits} {what}: one column per carry"
                );
                assert_eq!(
                    written, bits_checked,
                    "{id:?} int{bits} {what}: each carry is a bit"
                );
                assert_eq!(
                    limbs_checked,
                    limb_widths(bits, h),
                    "{id:?} int{bits} {what}: each limb of the answer is bounded at its width"
                );
            }
        }
    }
}

/// The columns a signed operation adds to the chain are its sign bits, and each is cut out of
/// its top limb: witnessed, checked to be a bit, and pinned by the rest of that limb being
/// range-checked one bit narrower.
#[test]
fn every_sign_bit_is_cut_out_of_its_top_limb() {
    for field in fields() {
        let id = field.id();
        let h = witness_limb_bits(field);
        let audit = |what: &str, ssa: &mut HLSSA, columns: usize, narrower: Vec<usize>, ties| {
            run_pass(ssa);
            let ops = emitted(ssa);
            let mut written: Vec<u64> = ops
                .iter()
                .filter_map(|op| match op {
                    OpCode::WriteWitness {
                        result: Some(result),
                        ..
                    } => Some(result.0),
                    _ => None,
                })
                .collect();
            let mut bits_checked: Vec<u64> = ops
                .iter()
                .filter_map(|op| match op {
                    OpCode::Rangecheck { value, max_bits: 1 } => Some(value.0),
                    _ => None,
                })
                .collect();
            let mut checked: Vec<usize> = ops
                .iter()
                .filter_map(|op| match op {
                    OpCode::Rangecheck { max_bits, .. } if *max_bits > 1 => Some(*max_bits),
                    _ => None,
                })
                .collect();
            let relations = ops
                .iter()
                .filter(|op| matches!(op, OpCode::Constrain { .. }))
                .count();

            written.sort_unstable();
            bits_checked.sort_unstable();
            checked.sort_unstable();
            let mut narrower = narrower;
            narrower.sort_unstable();
            assert_eq!(written.len(), columns, "{what}: its columns");
            assert_eq!(written, bits_checked, "{what}: each column is a bit");
            assert_eq!(checked, narrower, "{what}: the range checks");
            assert_eq!(relations, ties, "{what}: the relations between the bits");
        };

        for bits in limbed_widths(field) {
            let widths = limb_widths(bits, h);
            let (k, top) = (widths.len(), widths[widths.len() - 1]);
            let with_cuts = |cuts: usize, top_checked: bool| {
                let mut checks = widths.clone();
                if !top_checked {
                    checks.pop();
                }
                checks.extend(std::iter::repeat_n(top - 1, cuts));
                checks
            };
            let arith = |kind| {
                program_chaining_in(
                    field,
                    bits,
                    Some(Type::witness_of(Type::int(bits))),
                    move |result, lhs, rhs| OpCode::BinaryArithOp {
                        kind,
                        result,
                        lhs,
                        rhs,
                    },
                )
            };
            let mut ordering = program_chaining_in(
                field,
                bits,
                Some(Type::witness_of(Type::int(1))),
                |result, lhs, rhs| OpCode::Cmp {
                    kind: CmpKind::SLt,
                    result,
                    lhs,
                    rhs,
                },
            );
            let mut assertion =
                program_chaining_in(field, bits, None, |_, lhs, rhs| OpCode::AssertCmp {
                    kind: CmpKind::SLt,
                    lhs,
                    rhs,
                });

            // Both operands' signs and the answer's, beside a carry out of every limb.
            for kind in [BinaryArithOpKind::SAdd, BinaryArithOpKind::SSub] {
                let what = format!("{id:?} int{bits} {kind:?}");
                audit(&what, &mut arith(kind), k + 3, with_cuts(3, false), 1);
            }
            // Both operands' signs, and the chain an unsigned ordering has.
            audit(
                &format!("{id:?} int{bits} SLt"),
                &mut ordering,
                k + 2,
                with_cuts(2, true),
                0,
            );
            audit(
                &format!("{id:?} int{bits} asserted SLt"),
                &mut assertion,
                k + 1,
                with_cuts(2, true),
                0,
            );
        }

        // A sign extension into the representation cuts its source's sign and checks nothing
        // else: every limb past the source is that bit times all ones.
        let source = limbed_widths(field)[1];
        let top = limb_widths(source, h).last().copied().expect("a limb");
        let mut extension = program_chaining_in(
            field,
            source,
            Some(Type::witness_of(Type::int(1000))),
            |result, value, _| OpCode::SExt {
                result,
                value,
                from_bits: source,
                to_bits: 1000,
            },
        );
        audit(
            &format!("{id:?} int{source} to int1000 SExt"),
            &mut extension,
            1,
            vec![top - 1],
            0,
        );
    }
}

/// An operand's sign is cut once per block, however many signed operations read it, and an
/// operation that reads one operand twice cuts it once.
#[test]
fn an_operand_sign_is_cut_once_per_block() {
    let bits = 320usize;
    let top = *limb_widths(bits, witness_limb_bits(bn254()))
        .last()
        .expect("a limb");
    let mut ssa = HLSSA::with_main("main".to_string());
    let main = ssa.get_unique_entrypoint_id();
    let (lhs, rhs) = (ssa.fresh_value(), ssa.fresh_value());
    let (sum, ordering, double) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let mut builder = HLSSABuilder::new(&mut ssa);
    builder.modify_function(main, |fb| {
        let entry = fb.function.get_entry_id();
        for value in [lhs, rhs] {
            fb.function
                .get_block_mut(entry)
                .push_parameter(value, Type::witness_of(Type::int(bits)));
        }
        fb.function
            .add_return_type(Type::witness_of(Type::int(bits)));
        fb.function.add_return_type(Type::witness_of(Type::int(1)));
        fb.function
            .add_return_type(Type::witness_of(Type::int(bits)));
        let mut block = fb.test_block(entry);
        block.emit(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::SAdd,
            result: sum,
            lhs,
            rhs,
        });
        block.emit(OpCode::Cmp {
            kind: CmpKind::SLt,
            result: ordering,
            lhs,
            rhs,
        });
        block.emit(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::SAdd,
            result: double,
            lhs,
            rhs: lhs,
        });
        block.terminate_return(vec![sum, ordering, double]);
    });
    run_pass(&mut ssa);

    let cuts = emitted(&ssa)
        .iter()
        .filter(|op| matches!(op, OpCode::Rangecheck { max_bits, .. } if *max_bits == top - 1))
        .count();
    assert_eq!(cuts, 4, "the sign cuts");
}

/// A sign extension whose source's top limb is known fills at compile time.
#[test]
fn a_sign_extension_of_a_known_top_limb_fills_at_compile_time() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main = ssa.get_unique_entrypoint_id();
    let (value, widened, extended) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let mut builder = HLSSABuilder::new(&mut ssa);
    builder.modify_function(main, |fb| {
        let entry = fb.function.get_entry_id();
        fb.function
            .get_block_mut(entry)
            .push_parameter(value, Type::witness_of(Type::int(64)));
        fb.function
            .add_return_type(Type::witness_of(Type::int(1000)));
        let mut block = fb.test_block(entry);
        block.emit(OpCode::Cast {
            result: widened,
            value,
            target: CastTarget::Int(320),
        });
        block.emit(OpCode::SExt {
            result: extended,
            value: widened,
            from_bits: 320,
            to_bits: 1000,
        });
        block.terminate_return(vec![extended]);
    });
    run_pass(&mut ssa);

    let ops = emitted(&ssa);
    assert!(
        !ops.iter()
            .any(|op| matches!(op, OpCode::Rangecheck { max_bits: 1, .. })),
        "the known sign was cut"
    );
    let constant = |value: &ValueId| ssa.get_const(*value).is_some();
    assert!(
        !ops.iter().any(|op| matches!(
            op,
            OpCode::BinaryArithOp { lhs, rhs, .. } if constant(lhs) && constant(rhs)
        )),
        "a limb of the fill is arithmetic on two constants"
    );
}

/// A guarded chain checks nothing where the guard is off, since the operands are then whatever
/// the branch not taken left, and its answer is zero there.
#[test]
fn a_guarded_chain_is_checked_only_under_its_guard() {
    for field in fields() {
        let id = field.id();
        let bits = limbed_widths(field)[1];
        let mut ssa = difference_in(field, bits);
        let condition = guard_every_instruction(&mut ssa);
        run_pass(&mut ssa);
        let ops = emitted(&ssa);

        assert!(
            !ops.iter().any(|op| matches!(op, OpCode::Rangecheck { .. })),
            "{id:?}: no range check is unguarded"
        );
        let guarded = ops
            .iter()
            .filter(|op| {
                matches!(op, OpCode::Guard { condition: c, inner }
                if *c == condition && matches!(inner.as_ref(), OpCode::Rangecheck { .. }))
            })
            .count();
        let k = limb_widths(bits, witness_limb_bits(field)).len();
        assert_eq!(
            guarded,
            2 * k - 1,
            "{id:?}: every carry and every limb, under the guard"
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

/// The one width where each operand fits an element but the sum does not goes through the
/// chain, on a decomposition of each operand; one bit narrower, the pass leaves it alone.
#[test]
fn a_sum_the_element_cannot_carry_takes_the_chain_without_limbs() {
    let field = bn254();
    let wide = widest_cell_sum_bits(field) + 1;
    assert_eq!(
        wide,
        multi_cell_int_bits(field),
        "the band is one width wide"
    );

    let mut ssa = difference(wide);
    let main = ssa.get_unique_entrypoint_id();
    let operands: Vec<ValueId> = ssa
        .get_function(main)
        .get_entry()
        .get_parameters()
        .map(|(value, _)| *value)
        .collect();
    run_pass(&mut ssa);
    let ops = emitted(&ssa);
    let tied = ops
        .iter()
        .filter(|op| matches!(op, OpCode::Constrain { .. }))
        .count();
    assert_eq!(
        tied, 2,
        "each operand is decomposed and tied back to its element"
    );
    assert!(
        !ops.iter().any(|op| matches!(
            op,
            OpCode::BinaryArithOp { lhs, rhs, .. }
                if operands.contains(lhs) && operands.contains(rhs)
        )),
        "the single-cell difference is gone"
    );

    let mut narrower = difference(wide - 1);
    let before = format!("{:?}", emitted(&narrower));
    run_pass(&mut narrower);
    assert_eq!(
        format!("{:?}", emitted(&narrower)),
        before,
        "one bit narrower, the sum fits its element"
    );
}

/// An operand in the band is decomposed once per block, however many chained operations read
/// it and whichever side it is on.
#[test]
fn a_band_operand_is_decomposed_once_per_block() {
    let bits = multi_cell_int_bits(bn254());
    let constraints = |ssa: &HLSSA| {
        emitted(ssa)
            .iter()
            .filter(|op| matches!(op, OpCode::Constrain { .. }))
            .count()
    };

    // `x + x`: one value, one decomposition.
    let mut doubled = program_chaining(
        bits,
        Some(Type::witness_of(Type::int(bits))),
        |result, lhs, _| OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::UAdd,
            result,
            lhs,
            rhs: lhs,
        },
    );
    run_pass(&mut doubled);
    assert_eq!(constraints(&doubled), 1, "x + x decomposes x once");

    // `lhs - rhs` and then `lhs < rhs` in the same block: two values, two decompositions.
    let mut ssa = difference(bits);
    let main = ssa.get_unique_entrypoint_id();
    let operands: Vec<ValueId> = ssa
        .get_function(main)
        .get_entry()
        .get_parameters()
        .map(|(value, _)| *value)
        .collect();
    let less = ssa.fresh_value();
    ssa.get_function_mut(main)
        .get_entry_mut()
        .push_test_instruction(OpCode::Cmp {
            kind: CmpKind::ULt,
            result: less,
            lhs: operands[0],
            rhs: operands[1],
        });
    run_pass(&mut ssa);
    // Counting reconstructions alone cannot tell a reused decomposition from an ordering that
    // was never lowered, which would leave the count at two as well.
    assert!(
        !emitted(&ssa).iter().any(|op| matches!(
            op,
            OpCode::Cmp { lhs, rhs, .. } if *lhs == operands[0] && *rhs == operands[1]
        )),
        "the ordering went through the chain"
    );
    assert_eq!(
        constraints(&ssa),
        2,
        "the ordering reuses the decompositions the difference made"
    );
}

// THE SCHOOLBOOK
// --------------------------------------------------------------------------------------------

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
                Step::Column(index) => {
                    assert!(plan.evaluated, "a whole column in a summed plan");
                    assert_eq!(index, column, "a column added to another column");
                    for left in 0..=column {
                        let right = column - left;
                        if right < count && !(&lhs[left] * &rhs[right]).is_zero() {
                            assert!(added.insert((left, right)), "a product added twice");
                            accumulated += &lhs[left] * &rhs[right];
                        }
                    }
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
    if plan.evaluated {
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
            } else if plan.evaluated {
                continue;
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
        assert!(plan.evaluated, "int{bits} is evaluated on bn254");
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
    for evaluate in [false, true] {
        assert!(
            plan_product(
                &full_bounds(&goldilocks),
                &full_bounds(&goldilocks),
                &[],
                &goldilocks,
                63,
                evaluate
            )
            .is_none()
        );
    }

    // Five products to a column do not fit 130 bits, so an evaluation falls back to summing.
    let injective = 2 * 64 + 2;
    let fallback = plan_product(&full, &full, &[], &widths, injective, true)
        .expect("two products and a carry fit 130 bits");
    assert!(
        !fallback.evaluated,
        "a column past the modulus cannot be evaluated"
    );

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

/// `main(lhs: WitnessOf<int(bits)>, rhs: rhs_type) -> WitnessOf<int(bits)> { lhs op rhs }`.
fn operation_with(kind: BinaryArithOpKind, bits: usize, rhs_type: Type) -> HLSSA {
    operation_with_in(bn254(), kind, bits, rhs_type)
}

/// [`operation_with`] over `field`.
fn operation_with_in(
    field: FieldConfig,
    kind: BinaryArithOpKind,
    bits: usize,
    rhs_type: Type,
) -> HLSSA {
    let mut ssa = HLSSA::with_main_in(field, "main".to_string());
    let main = ssa.get_unique_entrypoint_id();
    let (lhs, rhs, result) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let mut builder = HLSSABuilder::new(&mut ssa);
    builder.modify_function(main, |fb| {
        let entry = fb.function.get_entry_id();
        let block = fb.function.get_block_mut(entry);
        block.push_parameter(lhs, Type::witness_of(Type::int(bits)));
        block.push_parameter(rhs, rhs_type);
        fb.function
            .add_return_type(Type::witness_of(Type::int(bits)));
        let mut block = fb.test_block(entry);
        block.emit(OpCode::BinaryArithOp {
            kind,
            result,
            lhs,
            rhs,
        });
        block.terminate_return(vec![result]);
    });
    ssa
}

/// `main(lhs, rhs: WitnessOf<int(bits)>) -> WitnessOf<int(bits)> { lhs * rhs }`.
fn product(bits: usize) -> HLSSA {
    product_in(bn254(), bits)
}

/// [`product`] over `field`.
fn product_in(field: FieldConfig, bits: usize) -> HLSSA {
    product_with(field, bits, Type::witness_of(Type::int(bits)))
}

/// Every carry a product witnesses is range-checked at the width its plan gives, and every limb
/// of the answer at its width, however the columns are formed.
///
/// Evaluated (both operands witnessed), each column is a separate witness pinned only by the
/// `2k - 1` evaluations, and nothing is held to zero. Summed (a pure right operand), the
/// columns are linear and every left limb above the lowest is held to a zero overflow.
///
/// Structural for the reason [`every_carry_is_a_bit_and_every_limb_of_the_chain_is_bounded`]
/// gives: perturbing a column by one trips the limb check before the carry check.
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
                assert_eq!(plan.evaluated, evaluated);
                let carry_bits: Vec<usize> = plan
                    .columns
                    .iter()
                    .flatten()
                    .filter_map(|step| match step {
                        Step::Reduce { carry_bits } => *carry_bits,
                        _ => None,
                    })
                    .collect();

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
                let (carries, columns): (Vec<ValueId>, Vec<ValueId>) =
                    written.iter().partition(|value| checked(**value).is_some());
                assert_eq!(
                    carries
                        .iter()
                        .map(|value| checked(*value).unwrap())
                        .collect::<Vec<_>>(),
                    carry_bits,
                    "{id:?} int{bits} {what}: one column per carry, each checked at its planned width"
                );
                assert_eq!(
                    columns.len(),
                    if evaluated { k } else { 0 },
                    "{id:?} int{bits} {what}: a witnessed column per column of the answer"
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
                    limbs, widths,
                    "{id:?} int{bits} {what}: each limb of the answer is bounded at its width"
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
/// one of polynomials, and each reads every limb of both operands and every column.
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
            let mut stack = vec![value];
            while let Some(value) = stack.pop() {
                match definitions.get(&value) {
                    Some(OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::UAdd,
                        lhs,
                        rhs,
                        ..
                    }) => stack.extend([*lhs, *rhs]),
                    Some(OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::UMul,
                        lhs,
                        rhs,
                        ..
                    }) if matches!(ssa.get_const(*rhs).as_deref(), Some(Constant::Field(_))) => {
                        let Some(Constant::Field(scale)) = ssa.get_const(*rhs).as_deref().cloned()
                        else {
                            unreachable!()
                        };
                        out.push((*lhs, scale));
                    }
                    _ => out.push((value, field.one())),
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
                2 * k,
                "{id:?}: evaluation {index} reads every right limb and column"
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
        let k = limb_widths(bits, witness_limb_bits(field)).len();
        let guarded = ops
            .iter()
            .filter(|op| {
                matches!(op, OpCode::Guard { condition: c, inner }
                if *c == condition && matches!(inner.as_ref(), OpCode::Rangecheck { .. }))
            })
            .count();
        assert_eq!(
            guarded,
            2 * k - 1,
            "{id:?}: every carry and every limb, under the guard"
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

// THE DIVISION
// --------------------------------------------------------------------------------------------

/// What a structural audit of one lowered program reads: every instruction by its results, and
/// every range check by the value it bounds, looking through casts on both sides.
struct Audit {
    ops: Vec<OpCode>,
    definitions: HashMap<ValueId, OpCode>,
}

impl Audit {
    fn of(ssa: &HLSSA) -> Self {
        let ops = emitted(ssa);
        let definitions = ops
            .iter()
            .flat_map(|op| op.get_results().map(move |result| (*result, op.clone())))
            .collect();
        Self { ops, definitions }
    }

    /// `value` with every cast it went through taken off.
    fn root(&self, mut value: ValueId) -> ValueId {
        while let Some(OpCode::Cast { value: source, .. }) = self.definitions.get(&value) {
            value = *source;
        }
        value
    }

    /// Every witness column written, in order.
    fn written(&self) -> Vec<ValueId> {
        self.ops
            .iter()
            .filter_map(|op| match op {
                OpCode::WriteWitness {
                    result: Some(result),
                    ..
                } => Some(*result),
                _ => None,
            })
            .collect()
    }

    /// Every range check as the root of what it bounds, its width, and its guard.
    fn checks(&self) -> Vec<(ValueId, usize, Option<ValueId>)> {
        self.ops
            .iter()
            .filter_map(|op| match op {
                OpCode::Rangecheck { value, max_bits } => {
                    Some((self.root(*value), *max_bits, None))
                }
                OpCode::Guard { condition, inner } => match inner.as_ref() {
                    OpCode::Rangecheck { value, max_bits } => {
                        Some((self.root(*value), *max_bits, Some(*condition)))
                    }
                    _ => None,
                },
                _ => None,
            })
            .collect()
    }

    /// The width each written column is checked at, in the order they are written, [`None`]
    /// for a column nothing checks.
    fn written_checks(&self) -> Vec<Option<usize>> {
        let checks = self.checks();
        self.written()
            .into_iter()
            .map(|column| {
                checks
                    .iter()
                    .find(|(root, _, _)| *root == column)
                    .map(|(_, bits, _)| *bits)
            })
            .collect()
    }

    fn constraints(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| matches!(op, OpCode::Constrain { .. }))
            .count()
    }
}

/// The carries the division's schoolbook plans, at full operand ranges.
fn division_carries(field: FieldConfig, widths: &[usize], evaluate: bool) -> Vec<usize> {
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
    assert_eq!(plan.evaluated, evaluate);
    plan.columns
        .iter()
        .flatten()
        .filter_map(|step| match step {
            Step::Reduce { carry_bits } => *carry_bits,
            _ => None,
        })
        .collect()
}

/// Every column a division witnesses is range-checked, and every column of `q·d + r` is held
/// to the dividend's limb instead of being checked, however the product is formed.
///
/// Checks, in order: the quotient's and the remainder's limbs at their widths; the schoolbook's
/// carries at the widths its plan gives; and the borrows of `r < d`, each a bit. Only an
/// evaluated product's columns go unchecked, as its evaluations pin them. The rest is the
/// chain's answer, one check per limb. The constraints are the product's (evaluations or
/// overflow terms) plus one equality per dividend limb.
///
/// Structural for the reason [`every_carry_and_every_limb_of_a_product_is_bounded`] gives, and
/// because a pinned column has no check to perturb: the equality is its bound.
#[test]
fn every_column_of_a_division_is_bounded_and_every_limb_pinned() {
    for field in fields() {
        let id = field.id();
        let h = witness_limb_bits(field);
        for kind in [BinaryArithOpKind::UDiv, BinaryArithOpKind::URem] {
            for bits in limbed_widths(field) {
                let widths = limb_widths(bits, h);
                let k = widths.len();
                for (what, rhs_type, evaluated) in [
                    ("evaluated", Type::witness_of(Type::int(bits)), true),
                    ("summed", Type::int(bits), false),
                ] {
                    let mut ssa = operation_with_in(field, kind, bits, rhs_type);
                    run_pass(&mut ssa);
                    let audit = Audit::of(&ssa);
                    let what = format!("{id:?} int{bits} {kind:?}, {what}");

                    let mut expected: Vec<Option<usize>> = Vec::new();
                    expected.extend(widths.iter().map(|width| Some(*width)));
                    expected.extend(widths.iter().map(|width| Some(*width)));
                    if evaluated {
                        expected.extend(std::iter::repeat_n(None, k));
                    }
                    expected.extend(
                        division_carries(field, &widths, evaluated)
                            .into_iter()
                            .map(Some),
                    );
                    expected.extend(std::iter::repeat_n(Some(1), k - 1));
                    assert_eq!(audit.written_checks(), expected, "{what}");

                    let written = audit.written();
                    let rest: Vec<usize> = audit
                        .checks()
                        .into_iter()
                        .filter(|(root, _, _)| !written.contains(root))
                        .map(|(_, bits, _)| bits)
                        .collect();
                    assert_eq!(rest, widths, "{what}: the chain's limbs, and nothing else");

                    let product = if evaluated { 2 * k - 1 } else { k - 1 };
                    assert_eq!(audit.constraints(), product + k, "{what}");
                }
            }
        }
    }
}

/// A constant divisor bounds both answers, so they cost only the limbs they can reach: the
/// quotient's top limb is checked narrower, the remainder is one limb at the divisor's width,
/// and `r < d` reads only that limb.
#[test]
fn a_constant_divisor_narrows_the_quotient_and_the_remainder() {
    let bits = 320usize;
    let mut ssa = operation_with(BinaryArithOpKind::UDiv, bits, Type::int(bits));
    let main = ssa.get_unique_entrypoint_id();
    let ten = ssa.add_const(Constant::Int(IntBits::from_u128(bits, 10)));
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
                    panic!("the program is one division")
                };
                let rewritten = OpCode::BinaryArithOp {
                    kind,
                    result,
                    lhs,
                    rhs: ten,
                };
                Located::new(rewritten, op.location().clone())
            })
            .collect();
        entry.put_instructions(instructions);
    }
    run_pass(&mut ssa);
    let audit = Audit::of(&ssa);

    let h = witness_limb_bits(bn254());
    let quotient_top = ((BigUint::one() << bits) - 1u8) / 10u8 >> (4 * h);
    let checks = audit.written_checks();
    assert_eq!(
        checks[..6],
        [
            Some(h),
            Some(h),
            Some(h),
            Some(h),
            Some(quotient_top.bits() as usize),
            Some(4)
        ],
        "the quotient's five limbs, the top one narrower, and a four-bit remainder"
    );
    assert!(
        checks[6..]
            .iter()
            .all(|check| check.is_some_and(|bits| bits > 1)),
        "carries only, and no borrow: `r < 10` is one limb"
    );
    assert_eq!(
        audit.constraints(),
        5,
        "one equality per limb of the dividend, and nothing to hold to zero"
    );
}

/// A guarded division guards only what reads its operands, and its answer is zero where the
/// guard is off without being selected.
///
/// The pure side divides zero by one where the guard is off, so the quotient, the remainder and
/// every column and carry the schoolbook builds from them are zero there, and their checks hold
/// unguarded whatever the operands. `q·d + r == n` and `r < d` read the operands, and are the
/// only checks under the guard.
#[test]
fn a_guarded_division_is_checked_only_under_its_guard() {
    for field in fields() {
        let id = field.id();
        let bits = limbed_widths(field)[1];
        let k = limb_widths(bits, witness_limb_bits(field)).len();
        let mut ssa = operation_with_in(
            field,
            BinaryArithOpKind::URem,
            bits,
            Type::witness_of(Type::int(bits)),
        );
        let condition = guard_every_instruction(&mut ssa);
        run_pass(&mut ssa);
        let audit = Audit::of(&ssa);

        let (unguarded, guarded): (Vec<_>, Vec<_>) = audit
            .checks()
            .into_iter()
            .partition(|(_, _, guard)| guard.is_none());
        assert!(
            unguarded.len() > 2 * k,
            "{id:?}: the quotient's and the remainder's limbs, and the schoolbook's columns and carries"
        );
        assert_eq!(
            guarded.len(),
            2 * k - 1,
            "{id:?}: the chain of `r < d`, each limb and each carry but the forced top one"
        );
        assert!(
            guarded
                .iter()
                .all(|(_, _, guard)| *guard == Some(condition)),
            "{id:?}: under the division's guard"
        );

        let flag: Vec<ValueId> = audit
            .ops
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
        let flagged = audit
        .ops
        .iter()
        .filter(|op| {
            matches!(op, OpCode::Constrain { a, b, .. } if flag.contains(a) || flag.contains(b))
        })
        .count();
        assert_eq!(
            flagged, k,
            "{id:?}: each equality, and nothing else, holds only where the guard does"
        );

        let selected = audit
            .ops
            .iter()
            .filter(|op| matches!(op, OpCode::Select { .. }))
            .count();
        assert_eq!(
            selected, 3,
            "{id:?}: the dividend and divisor on the pure side, and the divisor again away from zero"
        );
    }
}

/// A division the single cell cannot hold is lowered here, on a decomposition of each operand
/// while they still have an element; one it can hold, and the double lane's width, are left
/// to the single-cell lowering.
#[test]
fn a_division_the_single_cell_cannot_hold_is_lowered_here() {
    let field = bn254();
    let widest = widest_injective_int_bits(field) / 2;
    for kind in [BinaryArithOpKind::UDiv, BinaryArithOpKind::URem] {
        for bits in [widest + 1, 2 * HOST_LIMB_BITS + 1] {
            let mut ssa = operation_with(kind, bits, Type::witness_of(Type::int(bits)));
            let main = ssa.get_unique_entrypoint_id();
            let operands: Vec<ValueId> = ssa
                .get_function(main)
                .get_entry()
                .get_parameters()
                .map(|(value, _)| *value)
                .collect();
            run_pass(&mut ssa);
            assert!(
                !emitted(&ssa).iter().any(|op| matches!(
                    op,
                    OpCode::BinaryArithOp { lhs, rhs, .. }
                        if operands.contains(lhs) && operands.contains(rhs)
                )),
                "int{bits} {kind:?}: the single-cell division is gone"
            );
        }
        for bits in [widest, 2 * HOST_LIMB_BITS] {
            let mut ssa = operation_with(kind, bits, Type::witness_of(Type::int(bits)));
            let before = format!("{:?}", emitted(&ssa));
            run_pass(&mut ssa);
            assert_eq!(format!("{:?}", emitted(&ssa)), before, "int{bits} {kind:?}");
        }
    }
}

/// `d / d` needs no quotient or remainder column: it is one wherever `d` is not zero, so the
/// only thing checked is `0 < d`, the chain's borrows and limbs.
#[test]
fn a_value_divided_by_itself_checks_only_that_it_is_not_zero() {
    let bits = 320usize;
    let widths = limb_widths(bits, witness_limb_bits(bn254()));
    let mut ssa = operation_with(
        BinaryArithOpKind::UDiv,
        bits,
        Type::witness_of(Type::int(bits)),
    );
    let main = ssa.get_unique_entrypoint_id();
    {
        let entry = ssa.get_function_mut(main).get_entry_mut();
        let lhs = entry
            .get_parameters()
            .next()
            .map(|(value, _)| *value)
            .unwrap();
        let instructions: Vec<_> = entry
            .take_instructions()
            .into_iter()
            .map(|op| {
                let OpCode::BinaryArithOp { kind, result, .. } = op.as_ref().clone() else {
                    panic!("the program is one division")
                };
                let rewritten = OpCode::BinaryArithOp {
                    kind,
                    result,
                    lhs,
                    rhs: lhs,
                };
                Located::new(rewritten, op.location().clone())
            })
            .collect();
        entry.put_instructions(instructions);
    }
    run_pass(&mut ssa);
    let audit = Audit::of(&ssa);
    assert_eq!(
        audit.written_checks(),
        vec![Some(1); widths.len() - 1],
        "the borrows only"
    );
    assert_eq!(audit.checks().len(), 2 * widths.len() - 1);
    assert_eq!(audit.constraints(), 0);
}

/// `main(a, b: WitnessOf<int64>) { (a as int(bits)) == rhs }`, where `rhs` is `b` widened the
/// same way or the given constant, compared or asserted.
fn program_comparing_widened(bits: usize, constant: Option<BigUint>, assert: bool) -> HLSSA {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main = ssa.get_unique_entrypoint_id();
    let [a, b, wide_a, wide_b, result] = std::array::from_fn(|_| ssa.fresh_value());
    let mut builder = HLSSABuilder::new(&mut ssa);
    builder.modify_function(main, |fb| {
        let entry = fb.function.get_entry_id();
        for value in [a, b] {
            fb.function
                .get_block_mut(entry)
                .push_parameter(value, Type::witness_of(Type::int(64)));
        }
        if !assert {
            fb.function.add_return_type(Type::witness_of(Type::int(1)));
        }
        let rhs = match &constant {
            Some(pattern) => fb
                .ssa
                .add_const(Constant::Int(IntBits::from_biguint(bits, pattern))),
            None => wide_b,
        };
        let mut block = fb.test_block(entry);
        for (result, value) in [(wide_a, a), (wide_b, b)] {
            block.emit(OpCode::Cast {
                result,
                value,
                target: CastTarget::Int(bits),
            });
        }
        if assert {
            block.emit(OpCode::AssertCmp {
                kind: CmpKind::Eq,
                lhs: wide_a,
                rhs,
            });
            block.terminate_return(vec![]);
        } else {
            block.emit(OpCode::Cmp {
                kind: CmpKind::Eq,
                result,
                lhs: wide_a,
                rhs,
            });
            block.terminate_return(vec![result]);
        }
    });
    ssa
}

/// An equality between limbs both known at compile time is decided here, at no cost.
///
/// The limbs above a widened value's source are witnessed constants, which the comparison's
/// lowering would otherwise treat as witnesses, with a gadget each. So two values widened from
/// 64 to 320 bits compare in their low limb alone, and a constant with a bit above that limb
/// decides the comparison outright.
#[test]
fn an_equality_of_limbs_known_at_compile_time_is_decided_here() {
    let bits = 320usize;
    let compared = |ssa: &HLSSA| {
        emitted(ssa)
            .iter()
            .filter(|op| matches!(op, OpCode::Cmp { .. } | OpCode::AssertCmp { .. }))
            .count()
    };

    for assert in [false, true] {
        let mut ssa = program_comparing_widened(bits, None, assert);
        run_pass(&mut ssa);
        assert_eq!(
            compared(&ssa),
            1,
            "two widened values compare in the low limb alone, assert={assert}"
        );

        let small = BigUint::from(7u8);
        let mut ssa = program_comparing_widened(bits, Some(small), assert);
        run_pass(&mut ssa);
        assert_eq!(
            compared(&ssa),
            1,
            "a small constant is compared in the low limb alone, assert={assert}"
        );
    }

    let unreachable: BigUint = BigUint::from(1u8) << 300;
    let mut ssa = program_comparing_widened(bits, Some(unreachable.clone()), false);
    run_pass(&mut ssa);
    assert_eq!(compared(&ssa), 0, "a limb known to differ decides it");
    let ops = emitted(&ssa);
    let Some(OpCode::Cast { value, target, .. }) = ops.last() else {
        panic!("the answer is delivered last")
    };
    assert_eq!(*target, CastTarget::WitnessOf, "the answer stays witnessed");
    assert!(
        matches!(ssa.get_const(*value).as_deref(), Some(Constant::Int(pattern)) if pattern.is_zero()),
        "the answer is false"
    );

    // An assertion that cannot hold is kept, to be refused as any other is.
    let mut ssa = program_comparing_widened(bits, Some(unreachable), true);
    run_pass(&mut ssa);
    assert_eq!(
        compared(&ssa),
        2,
        "the low limb and the one known to differ"
    );
}

/// A division whose answer the lowering knows limb by limb hands those limbs on as known, so an
/// equality the answer reaches is decided here rather than compared.
///
/// `0 / d` knows every limb of both answers, so `(0 / d) == 0` compares nothing; a zero divisor
/// is still refused by the division's `0 < d` check.
#[test]
fn a_known_answer_limb_is_known_to_the_comparison_it_reaches() {
    let bits = 320usize;
    for kind in [BinaryArithOpKind::UDiv, BinaryArithOpKind::URem] {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();
        let [divisor, answer, result] = std::array::from_fn(|_| ssa.fresh_value());
        let mut builder = HLSSABuilder::new(&mut ssa);
        builder.modify_function(main, |fb| {
            let entry = fb.function.get_entry_id();
            fb.function
                .get_block_mut(entry)
                .push_parameter(divisor, Type::witness_of(Type::int(bits)));
            fb.function.add_return_type(Type::witness_of(Type::int(1)));
            let zero = fb.ssa.add_const(Constant::Int(IntBits::zero(bits)));
            let mut block = fb.test_block(entry);
            block.emit(OpCode::BinaryArithOp {
                kind,
                result: answer,
                lhs: zero,
                rhs: divisor,
            });
            block.emit(OpCode::Cmp {
                kind: CmpKind::Eq,
                result,
                lhs: answer,
                rhs: zero,
            });
            block.terminate_return(vec![result]);
        });
        run_pass(&mut ssa);

        // The one equality left is the pure-side zero test of the divisor, at the division's
        // width; a limb comparison would be at a limb's width.
        let ops = emitted(&ssa);
        let compared: Vec<&OpCode> = ops
            .iter()
            .filter(|op| {
                matches!(
                    op,
                    OpCode::Cmp {
                        kind: CmpKind::Eq,
                        ..
                    }
                )
            })
            .collect();
        let the_hints_own = |rhs: &ValueId| {
            matches!(
                ssa.get_const(*rhs).as_deref(),
                Some(Constant::Int(pattern)) if pattern.bits() == bits && pattern.is_zero()
            )
        };
        assert!(
            matches!(compared[..], [OpCode::Cmp { rhs, .. }] if the_hints_own(rhs)),
            "{kind:?}: the equality compared limbs it knew: {compared:?}"
        );
        let Some(OpCode::Cast { value, target, .. }) = ops.last() else {
            panic!("{kind:?}: the answer is delivered last")
        };
        assert_eq!(
            *target,
            CastTarget::WitnessOf,
            "{kind:?}: the answer stays witnessed"
        );
        assert!(
            matches!(ssa.get_const(*value).as_deref(), Some(Constant::Int(pattern)) if pattern.is_one()),
            "{kind:?}: the answer is true"
        );
        assert!(
            ops.iter()
                .any(|op| matches!(op, OpCode::Rangecheck { max_bits: 1, .. })),
            "{kind:?}: the chain for `0 < d` still borrows"
        );
    }
}

/// A constant entering the representation as witnessed is its limbs as witnessed constants,
/// with no column and no range check between them.
#[test]
fn a_constant_injected_into_the_representation_costs_nothing() {
    let bits = 328usize;
    let mut ssa = HLSSA::with_main("main".to_string());
    let main = ssa.get_unique_entrypoint_id();
    let result = ssa.fresh_value();
    let pattern: BigUint = (BigUint::from(1u8) << 320) + 5u8;
    let mut builder = HLSSABuilder::new(&mut ssa);
    builder.modify_function(main, |fb| {
        fb.function
            .add_return_type(Type::witness_of(Type::int(bits)));
        let entry = fb.function.get_entry_id();
        let value = fb
            .ssa
            .add_const(Constant::Int(IntBits::from_biguint(bits, &pattern)));
        let mut block = fb.test_block(entry);
        block.emit(OpCode::Cast {
            result,
            value,
            target: CastTarget::WitnessOf,
        });
        block.terminate_return(vec![result]);
    });
    run_pass(&mut ssa);

    let ops = emitted(&ssa);
    assert!(
        !ops.iter()
            .any(|op| matches!(op, OpCode::Rangecheck { .. } | OpCode::WriteWitness { .. })),
        "a constant is pinned by being one: {ops:?}"
    );
    let limbs: Vec<BigUint> = ops
        .iter()
        .map(|op| match op {
            OpCode::Cast {
                value,
                target: CastTarget::WitnessOf,
                ..
            } => match ssa.get_const(*value).as_deref() {
                Some(Constant::Int(limb)) => BigUint::from(limb),
                other => panic!("a limb that is not a constant: {other:?}"),
            },
            other => panic!("an injected constant emits {other:?}"),
        })
        .collect();
    let limb_count = limb_widths(bits, witness_limb_bits(bn254())).len();
    let mut expected = vec![BigUint::zero(); limb_count];
    expected[0] = BigUint::from(5u8);
    expected[limb_count - 1] = BigUint::from(1u8);
    assert_eq!(limbs, expected);
}

/// A quotient and a remainder of the same operands in one block are one gadget: the second
/// reads its answer out of the first, so the pair writes no more columns and checks no more
/// than the quotient alone.
#[test]
fn a_quotient_beside_its_remainder_is_one_division() {
    let bits = 320usize;
    let program = |kinds: &[BinaryArithOpKind]| {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();
        let (lhs, rhs) = (ssa.fresh_value(), ssa.fresh_value());
        let results: Vec<ValueId> = kinds.iter().map(|_| ssa.fresh_value()).collect();
        let mut builder = HLSSABuilder::new(&mut ssa);
        builder.modify_function(main, |fb| {
            let entry = fb.function.get_entry_id();
            let block = fb.function.get_block_mut(entry);
            block.push_parameter(lhs, Type::witness_of(Type::int(bits)));
            block.push_parameter(rhs, Type::witness_of(Type::int(bits)));
            for _ in kinds {
                fb.function
                    .add_return_type(Type::witness_of(Type::int(bits)));
            }
            let mut block = fb.test_block(entry);
            for (kind, result) in kinds.iter().zip(&results) {
                block.emit(OpCode::BinaryArithOp {
                    kind: *kind,
                    result: *result,
                    lhs,
                    rhs,
                });
            }
            block.terminate_return(results.clone());
        });
        run_pass(&mut ssa);
        let audit = Audit::of(&ssa);
        (
            audit.written().len(),
            audit.checks().len(),
            audit.constraints(),
        )
    };

    let quotient = program(&[BinaryArithOpKind::UDiv]);
    let both = program(&[BinaryArithOpKind::UDiv, BinaryArithOpKind::URem]);
    assert_eq!(both, quotient);
}

// THE SHIFT
// --------------------------------------------------------------------------------------------

/// `ssa`'s one operation with its right operand replaced by the constant `pattern`.
fn with_constant_rhs(ssa: &mut HLSSA, pattern: IntBits) {
    let main = ssa.get_unique_entrypoint_id();
    let constant = ssa.add_const(Constant::Int(pattern));
    let entry = ssa.get_function_mut(main).get_entry_mut();
    let instructions: Vec<_> = entry
        .take_instructions()
        .into_iter()
        .map(|op| {
            let OpCode::BinaryArithOp {
                kind, result, lhs, ..
            } = op.as_ref().clone()
            else {
                panic!("the program is one operation")
            };
            let rewritten = OpCode::BinaryArithOp {
                kind,
                result,
                lhs,
                rhs: constant,
            };
            Located::new(rewritten, op.location().clone())
        })
        .collect();
    entry.put_instructions(instructions);
}

/// Every column a shift by a witnessed amount writes is bounded, and every half of a split limb
/// is range-checked.
///
/// Written columns, in order: `r` (bounded by the table); the bits of `q`; `f = 2^r` (pinned by
/// the table); for a right shift, the cofactor (pinned by its product with `f`); then each
/// limb's high half, one bit under its width for a left shift and at its width for a right one.
/// A left shift does not split a narrow top limb; it cuts the answer's top limb instead, whose
/// bits above its width are the last column, at the limb width. Checked but not written: each
/// split limb's low half at the limb width, the amount bound where the decomposition passes the
/// width, and the cut top limb at its width. Constraints: one per upper limb of the amount, the
/// decomposition, and the cofactor's.
///
/// Structural for the reason [`every_carry_and_every_limb_of_a_product_is_bounded`] gives: a
/// low half is a difference, not a column, so perturbing columns cannot reach it.
#[test]
fn every_column_of_a_shift_is_bounded_and_every_half_checked() {
    for field in fields() {
        let id = field.id();
        let h = witness_limb_bits(field);
        for bits in limbed_widths(field) {
            let widths = limb_widths(bits, h);
            let k = widths.len();
            let stages = ceil_log2(k);
            let amount_limbs = k;
            for kind in [BinaryArithOpKind::UShl, BinaryArithOpKind::UShr] {
                let left = kind == BinaryArithOpKind::UShl;
                let mut ssa =
                    operation_with_in(field, kind, bits, Type::witness_of(Type::int(bits)));
                run_pass(&mut ssa);
                let audit = Audit::of(&ssa);
                let what = format!("{id:?} int{bits} {kind:?}");

                let mut expected: Vec<Option<usize>> = vec![None];
                expected.extend(std::iter::repeat_n(Some(1), stages));
                expected.push(None);
                if !left {
                    expected.push(None);
                }
                let top = widths[k - 1];
                let narrow_top = left && top < h;
                let split = if narrow_top { k - 1 } else { k };
                expected.extend(
                    widths[..split]
                        .iter()
                        .map(|width| Some(if left { width - 1 } else { *width })),
                );
                if narrow_top {
                    expected.push(Some(h));
                }
                assert_eq!(audit.written_checks(), expected, "{what}");

                let written = audit.written();
                let mut rest: Vec<usize> = audit
                    .checks()
                    .into_iter()
                    .filter(|(root, _, _)| !written.contains(root))
                    .map(|(_, bits, _)| bits)
                    .collect();
                rest.sort_unstable();
                let mut want = vec![h; split];
                if h << stages != bits {
                    want.push(ceil_log2(bits));
                }
                if narrow_top {
                    want.push(top);
                }
                want.sort_unstable();
                assert_eq!(
                    rest, want,
                    "{what}: the low halves and the bounds, and nothing else"
                );

                // The table's size is what bounds `r` below `h`.
                let tables: Vec<usize> = audit
                    .ops
                    .iter()
                    .filter_map(|op| match op {
                        OpCode::Lookup {
                            target: LookupTarget::Pow2(size),
                            ..
                        } => Some(usize::from(*size)),
                        _ => None,
                    })
                    .collect();
                assert_eq!(
                    tables,
                    vec![h.trailing_zeros() as usize],
                    "{what}: one row for `r`, in the table of the limb's `2^r`"
                );
                assert_eq!(
                    audit.constraints(),
                    (amount_limbs - 1) + 1 + usize::from(!left),
                    "{what}"
                );
            }
        }
    }
}

/// A shift by a constant cuts each limb once where the amount does not divide it, witnesses the
/// upper piece and checks both, and costs nothing where it moves whole limbs.
///
/// By 70 at 320 bits, every result limb starts six bits into a source limb, so each of the four
/// limbs that reach the answer is cut at 58 for a left shift and at 6 for a right one, and the
/// fifth moves past the top or below the bottom whole. By 128 nothing is cut.
#[test]
fn a_shift_by_a_constant_cuts_each_limb_at_most_once() {
    let bits = 320usize;
    for kind in [BinaryArithOpKind::UShl, BinaryArithOpKind::UShr] {
        let left = kind == BinaryArithOpKind::UShl;
        let mut ssa = operation_with(kind, bits, Type::int(bits));
        with_constant_rhs(&mut ssa, IntBits::from_u128(bits, 70));
        run_pass(&mut ssa);
        let audit = Audit::of(&ssa);
        let (upper, lower) = if left { (6, 58) } else { (58, 6) };
        assert_eq!(
            audit.written_checks(),
            vec![Some(upper); 4],
            "{kind:?} by 70: one upper piece per limb it cuts"
        );
        let written = audit.written();
        let rest: Vec<usize> = audit
            .checks()
            .into_iter()
            .filter(|(root, _, _)| !written.contains(root))
            .map(|(_, bits, _)| bits)
            .collect();
        assert_eq!(
            rest,
            vec![lower; 4],
            "{kind:?} by 70: and the piece below it"
        );
        assert_eq!(audit.constraints(), 0, "{kind:?} by 70");

        let mut ssa = operation_with(kind, bits, Type::int(bits));
        with_constant_rhs(&mut ssa, IntBits::from_u128(bits, 128));
        run_pass(&mut ssa);
        let audit = Audit::of(&ssa);
        assert!(audit.written().is_empty(), "{kind:?} by 128");
        assert!(audit.checks().is_empty(), "{kind:?} by 128");
    }
}

/// A witnessed amount's limbs above the lowest are held to zero, which refuses only amounts
/// past the width while every width is below `2^h`, the most the lowest limb holds.
#[test]
fn every_legal_shift_amount_fits_the_amounts_lowest_limb() {
    for field in fields() {
        let lowest = BigUint::one() << witness_limb_bits(field);
        assert!(
            BigUint::from(MAX_SUPPORTED_INT_BITS) < lowest,
            "{:?}",
            field.id()
        );
    }
}

/// A pure amount witnesses nothing about itself: no bits of `q`, no `r`, no power of two and no
/// table row. Its check is one comparison, and each limb's split is by a constant, so the only
/// columns are the limbs' high halves.
///
/// A witnessed guard changes only the comparison, which it guards: the amount is not zeroed
/// where the guard is off, so nothing about it is witnessed there either.
#[test]
fn a_pure_amount_is_split_by_a_constant() {
    let h = witness_limb_bits(bn254());
    let bits = 320usize;
    let widths = limb_widths(bits, h);
    for kind in [BinaryArithOpKind::UShl, BinaryArithOpKind::UShr] {
        for guarded in [false, true] {
            let what = format!("{kind:?}, guarded: {guarded}");
            let left = kind == BinaryArithOpKind::UShl;
            let mut ssa = operation_with(kind, bits, Type::int(bits));
            let condition = guarded.then(|| guard_every_instruction(&mut ssa));
            run_pass(&mut ssa);
            let audit = Audit::of(&ssa);
            let expected: Vec<Option<usize>> = widths
                .iter()
                .map(|width| Some(if left { width - 1 } else { *width }))
                .collect();
            assert_eq!(audit.written_checks(), expected, "{what}");
            assert_eq!(audit.constraints(), 0, "{what}");
            assert!(
                !audit
                    .ops
                    .iter()
                    .any(|op| matches!(op, OpCode::Lookup { .. })),
                "{what}: no table row"
            );
            let asserts: Vec<Option<ValueId>> = audit
                .ops
                .iter()
                .filter_map(|op| match op {
                    OpCode::AssertCmp { .. } => Some(None),
                    OpCode::Guard { condition, inner }
                        if matches!(inner.as_ref(), OpCode::AssertCmp { .. }) =>
                    {
                        Some(Some(*condition))
                    }
                    _ => None,
                })
                .collect();
            assert_eq!(
                asserts,
                vec![condition],
                "{what}: the amount is compared, under the guard where there is one"
            );
        }
    }
}

/// Wrap every instruction of `ssa`'s entry block in a guard on a new witnessed condition, which
/// is returned.
fn guard_every_instruction(ssa: &mut HLSSA) -> ValueId {
    let main = ssa.get_unique_entrypoint_id();
    let condition = ssa.fresh_value();
    let entry = ssa.get_function_mut(main).get_entry_mut();
    entry.push_parameter(condition, Type::witness_of(Type::int(1)));
    let instructions: Vec<_> = entry
        .take_instructions()
        .into_iter()
        .map(|op| {
            let guarded = OpCode::Guard {
                condition,
                inner: Box::new(op.as_ref().clone()),
            };
            Located::new(guarded, op.location().clone())
        })
        .collect();
    entry.put_instructions(instructions);
    condition
}

/// A witnessed operand's known limbs drop out of a product as a constant's do.
///
/// `(a as int320) · b` plans no term against `a`'s four upper limbs, which the widening knows
/// to be zero. What is left is one partial product per limb of `b`, which is cheaper summed
/// than the nine evaluations a product of two full operands pays, so the product constrains no
/// evaluation and no overflow term at all.
#[test]
fn a_known_limb_of_a_witnessed_operand_drops_out_of_the_product() {
    let bits = 320usize;
    let mut ssa = HLSSA::with_main("main".to_string());
    let main = ssa.get_unique_entrypoint_id();
    let (narrow, wide, widened, product) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );
    let mut builder = HLSSABuilder::new(&mut ssa);
    builder.modify_function(main, |fb| {
        fb.function
            .add_return_type(Type::witness_of(Type::int(bits)));
        let entry = fb.function.get_entry_id();
        let block = fb.function.get_block_mut(entry);
        block.push_parameter(narrow, Type::witness_of(Type::int(64)));
        block.push_parameter(wide, Type::witness_of(Type::int(bits)));
        let mut block = fb.test_block(entry);
        block.emit(OpCode::Cast {
            result: widened,
            value: narrow,
            target: CastTarget::Int(bits),
        });
        block.emit(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::UMul,
            result: product,
            lhs: widened,
            rhs: wide,
        });
        block.terminate_return(vec![product]);
    });
    run_pass(&mut ssa);
    let audit = Audit::of(&ssa);

    let k = limb_widths(bits, witness_limb_bits(bn254())).len();
    let witnessed_products = audit
        .ops
        .iter()
        .filter(|op| {
            let OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::UMul,
                lhs,
                rhs,
                ..
            } = op
            else {
                return false;
            };
            [lhs, rhs]
                .into_iter()
                .all(|value| ssa.get_const(*value).is_none())
        })
        .count();
    // The one constraint is the widening's, tying `a`'s limb back to `a`.
    assert_eq!(audit.constraints(), 1, "no evaluation and no overflow term");
    // Per column: the partial product and its hint.
    assert_eq!(
        witnessed_products,
        2 * k,
        "one partial product per limb of `b`"
    );
}

fn run_pass(ssa: &mut HLSSA) {
    let flow = FlowAnalysis::run(ssa);
    let types = Types::new().run(ssa, &flow);
    let mut store = AnalysisStore::new();
    store.insert_with_deps(flow, vec![]);
    store.insert_with_deps(types, vec![]);
    WideWitnessInts::new().run(ssa, &store);
}

// SPREADS
// --------------------------------------------------------------------------------------

/// `main(x: witness int(bits))` over `field`, answering `op(x)`, whose results are
/// `result_types`.
fn program_answering(
    field: FieldConfig,
    bits: usize,
    result_types: Vec<Type>,
    op: impl FnOnce(ValueId, &[ValueId]) -> OpCode,
) -> HLSSA {
    let mut ssa = HLSSA::with_main_in(field, "main".to_string());
    let main = ssa.get_unique_entrypoint_id();
    let value = ssa.fresh_value();
    let results: Vec<ValueId> = result_types.iter().map(|_| ssa.fresh_value()).collect();
    let mut builder = HLSSABuilder::new(&mut ssa);
    builder.modify_function(main, |fb| {
        for ty in &result_types {
            fb.function.add_return_type(ty.clone());
        }
        let entry = fb.function.get_entry_id();
        fb.function
            .get_block_mut(entry)
            .push_parameter(value, Type::witness_of(Type::int(bits)));
        let mut block = fb.test_block(entry);
        block.emit(op(value, &results));
        block.terminate_return(results);
    });
    ssa
}

/// The narrow spreads a widest spread becomes, one per half-limb the read reaches, each at the
/// bits of its half-limb the read takes, and one zero constraint per limb wholly above the
/// read.
#[test]
fn a_spread_at_the_widest_width_is_one_narrow_spread_per_half_limb() {
    for field in fields() {
        let id = field.id();
        let bits = MAX_SPREAD_INPUT_BITS;
        let half = witness_half_limb_bits(field);
        let limb = witness_limb_bits(field);
        for read in [bits, 100] {
            let mut ssa = program_answering(
                field,
                bits,
                vec![Type::witness_of(Type::int(2 * bits))],
                |value, results| OpCode::Spread {
                    result: results[0],
                    value,
                    value_bits: read,
                },
            );
            run_pass(&mut ssa);

            let ops = emitted(&ssa);
            let reads: Vec<usize> = ops
                .iter()
                .filter_map(|op| match op {
                    OpCode::Spread { value_bits, .. } => Some(*value_bits),
                    _ => None,
                })
                .collect();
            assert_eq!(reads.len(), read.div_ceil(half), "{id:?} read {read}");
            assert!(
                reads.iter().all(|&taken| taken <= half),
                "{id:?} read {read}"
            );
            assert_eq!(reads.iter().sum::<usize>(), read, "{id:?} read {read}");
            let zeros = ops
                .iter()
                .filter(|op| matches!(op, OpCode::Constrain { .. }))
                .count();
            assert_eq!(
                zeros,
                bits.div_ceil(limb) - read.div_ceil(limb),
                "{id:?} read {read}"
            );
        }
    }
}

/// Under a guard, every narrow spread and unspread is emitted under it and every zero check and
/// one-bit check is built under it, so that none of them holds where the guard does not.
#[test]
fn a_guarded_spread_or_unspread_builds_every_check_under_its_guard() {
    for field in fields() {
        let id = field.id();
        let bits = 600usize;
        for unspread in [false, true] {
            let mut ssa = HLSSA::with_main_in(field, "main".to_string());
            let main = ssa.get_unique_entrypoint_id();
            let (condition, value) = (ssa.fresh_value(), ssa.fresh_value());
            let results = [ssa.fresh_value(), ssa.fresh_value()];
            let mut builder = HLSSABuilder::new(&mut ssa);
            builder.modify_function(main, |fb| {
                let entry = fb.function.get_entry_id();
                let inner = if unspread {
                    fb.function
                        .add_return_type(Type::witness_of(Type::int(bits / 2)));
                    fb.function
                        .add_return_type(Type::witness_of(Type::int(bits / 2)));
                    OpCode::Unspread {
                        result_odd: results[0],
                        result_even: results[1],
                        value,
                        value_bits: 129,
                    }
                } else {
                    fb.function
                        .add_return_type(Type::witness_of(Type::int(2 * bits)));
                    OpCode::Spread {
                        result: results[0],
                        value,
                        value_bits: 100,
                    }
                };
                let block = fb.function.get_block_mut(entry);
                block.push_parameter(condition, Type::witness_of(Type::int(1)));
                block.push_parameter(value, Type::witness_of(Type::int(bits)));
                let mut block = fb.test_block(entry);
                block.emit(OpCode::Guard {
                    condition,
                    inner: Box::new(inner),
                });
                let answers = if unspread {
                    results.to_vec()
                } else {
                    vec![results[0]]
                };
                block.terminate_return(answers);
            });
            run_pass(&mut ssa);

            let ops = emitted(&ssa);
            assert!(
                !ops.iter().any(|op| matches!(
                    op,
                    OpCode::Spread { .. } | OpCode::Unspread { .. } | OpCode::Rangecheck { .. }
                )),
                "{id:?}: an unguarded check under a guard: {ops:?}"
            );
            let guarded = ops
                .iter()
                .filter(|op| match op {
                    OpCode::Guard { inner, .. } => matches!(
                        **inner,
                        OpCode::Spread { .. } | OpCode::Unspread { .. } | OpCode::Rangecheck { .. }
                    ),
                    _ => false,
                })
                .count();
            assert!(
                guarded > 0,
                "{id:?}: nothing was built under the guard: {ops:?}"
            );
            let one = ssa.add_const(Constant::Field(field.one()));
            assert!(
                ops.iter()
                    .all(|op| !matches!(op, OpCode::Constrain { a, .. } if *a == one)),
                "{id:?}: a zero check holds whatever the guard: {ops:?}"
            );
        }
    }
}

/// A spread or unspread of a widened value spends no narrow lookup on the limbs above the
/// source, which are the zero constant.
#[test]
fn a_limb_known_to_be_zero_costs_a_spread_or_unspread_nothing() {
    let (from, to) = (200usize, 600usize);
    let limb = witness_limb_bits(bn254());
    for unspread in [false, true] {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();
        let (value, wide) = (ssa.fresh_value(), ssa.fresh_value());
        let results = [ssa.fresh_value(), ssa.fresh_value()];
        let mut builder = HLSSABuilder::new(&mut ssa);
        builder.modify_function(main, |fb| {
            let entry = fb.function.get_entry_id();
            let (op, answers) = if unspread {
                for _ in 0..2 {
                    fb.function
                        .add_return_type(Type::witness_of(Type::int(to / 2)));
                }
                let op = OpCode::Unspread {
                    result_odd: results[0],
                    result_even: results[1],
                    value: wide,
                    value_bits: to,
                };
                (op, results.to_vec())
            } else {
                fb.function
                    .add_return_type(Type::witness_of(Type::int(2 * to)));
                let op = OpCode::Spread {
                    result: results[0],
                    value: wide,
                    value_bits: to,
                };
                (op, vec![results[0]])
            };
            fb.function
                .get_block_mut(entry)
                .push_parameter(value, Type::witness_of(Type::int(from)));
            let mut block = fb.test_block(entry);
            block.emit(OpCode::Cast {
                result: wide,
                value,
                target: CastTarget::Int(to),
            });
            block.emit(op);
            block.terminate_return(answers);
        });
        run_pass(&mut ssa);

        let narrow = emitted(&ssa)
            .iter()
            .filter(|op| matches!(op, OpCode::Spread { .. } | OpCode::Unspread { .. }))
            .count();
        let source_limbs = from.div_ceil(limb);
        let expected = if unspread {
            source_limbs
        } else {
            source_limbs * (limb / witness_half_limb_bits(bn254()))
        };
        assert_eq!(narrow, expected, "unspread {unspread}");
    }
}

/// The narrow unspreads a widest unspread becomes, one per limb the read reaches, except that a
/// limb the read takes one bit of is range-checked to a bit instead, having no odd bit.
#[test]
fn an_unspread_at_the_widest_width_is_one_narrow_unspread_per_limb() {
    for field in fields() {
        let id = field.id();
        let bits = MAX_SUPPORTED_INT_BITS;
        let limb = witness_limb_bits(field);
        for read in [bits, 1000, 2 * limb + 1] {
            let mut ssa = program_answering(
                field,
                bits,
                vec![Type::witness_of(Type::int(bits / 2)); 2],
                |value, results| OpCode::Unspread {
                    result_odd: results[0],
                    result_even: results[1],
                    value,
                    value_bits: read,
                },
            );
            run_pass(&mut ssa);

            let ops = emitted(&ssa);
            let reads: Vec<usize> = ops
                .iter()
                .filter_map(|op| match op {
                    OpCode::Unspread { value_bits, .. } => Some(*value_bits),
                    _ => None,
                })
                .collect();
            let one_bit = usize::from(read % limb == 1);
            assert_eq!(
                reads.len(),
                read.div_ceil(limb) - one_bit,
                "{id:?} read {read}"
            );
            assert_eq!(
                reads.iter().sum::<usize>() + one_bit,
                read,
                "{id:?} read {read}"
            );
            let bits_checks = ops
                .iter()
                .filter(|op| matches!(op, OpCode::Rangecheck { max_bits: 1, .. }))
                .count();
            assert_eq!(bits_checks, one_bit, "{id:?} read {read}");
            let zeros = ops
                .iter()
                .filter(|op| matches!(op, OpCode::Constrain { .. }))
                .count();
            assert_eq!(
                zeros,
                bits.div_ceil(limb) - read.div_ceil(limb),
                "{id:?} read {read}"
            );
        }
    }
}
