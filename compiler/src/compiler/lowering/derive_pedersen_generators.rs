//! Compile-time lowering for Noir's `derive_pedersen_generators` builtin.
//!
//! Noir deliberately requires both builtin arguments to be compile-time constants. The
//! targeted inlining exposes concrete call-site arguments before the builtin is converted. This
//! module evaluates those arguments and computes the points immediately, so no placeholder or
//! derivation-specific SSA specialization is needed.

use crate::{
    collections::HashMap,
    compiler::{
        Field,
        analysis::click_cooper::lattice,
        ssa::ValueId,
        ssa::hlssa::{Blob, Constant, HLFunction, HLSSA, OpCode},
    },
};

/// Evaluate one builtin call from the constants already emitted in `function`.
pub(super) fn derive(
    ssa: &HLSSA,
    function: &HLFunction,
    domain_separator: ValueId,
    starting_index: ValueId,
    num_generators: u32,
) -> Option<Vec<(Field, Field)>> {
    let constants = known_constants(ssa, function);
    let domain =
        constant_for(ssa, &constants, domain_separator).and_then(domain_separator_bytes)?;
    let starting_index = constant_for(ssa, &constants, starting_index).and_then(constant_u32)?;

    Some(
        bn254_blackbox_solver::derive_generators(&domain, num_generators, starting_index)
            .into_iter()
            .map(|generator| (Field::from(generator.x), Field::from(generator.y)))
            .collect(),
    )
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
