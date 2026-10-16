//! Rewrite a bit or byte decomposition of an integer into decompositions of field elements.
//!
//! The field form checks its digits only modulo `p`, leaving canonicity to Noir's standard library,
//! which checks it only for a `Field`. So an integer is decomposed as its field image only when it
//! and the digits asked for both fit an injective width: the digit sum is then below `p`, so the
//! digits are unique, and a value with more digits than asked for is rejected.
//!
//! Otherwise it is cut into chunks of whole bytes the field carries injectively, each decomposed
//! into only the digits its width needs, the rest zero, and any bits past the digits asked for are
//! asserted zero.

use crate::compiler::{
    passes::{
        instruction_lowering::{InstructionLoweringRule, LoweringContext, integer_bits},
        shared::limbs::widest_injective_int_bits,
    },
    ssa::{
        ValueId,
        hlssa::{
            BinaryArithOpKind, CastTarget, Endianness, OpCode, Radix, SequenceTargetType, Type,
            builder::{HLBlockEmitter, HLEmitter},
        },
    },
};
use mavros_int_semantics::IntBits;

pub(super) struct IntegerDigits;

impl InstructionLoweringRule for IntegerDigits {
    fn needs_value_ranges(&self) -> bool {
        false
    }

    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        let (value, digit_bits, radix) = match instruction {
            OpCode::ToBits { value, .. } => (*value, 1, None),
            OpCode::ToRadix { value, radix, .. } => (*value, 8, Some(*radix)),
            _ => return false,
        };
        let Some(bits) = context
            .types()
            .try_get_value_type(value)
            .and_then(integer_bits)
        else {
            return false;
        };

        let (result, endianness, count) = match instruction {
            OpCode::ToBits {
                result,
                endianness,
                count,
                ..
            }
            | OpCode::ToRadix {
                result,
                endianness,
                count,
                ..
            } => (*result, *endianness, *count),
            _ => unreachable!("matched above"),
        };
        let widest = widest_injective_int_bits(b.field());
        if bits <= widest && count * digit_bits <= widest {
            let image = b.cast_to_field(value);
            let mut op = instruction.clone();
            match &mut op {
                OpCode::ToBits { value, .. } | OpCode::ToRadix { value, .. } => *value = image,
                _ => unreachable!("matched above"),
            }
            b.emit(op);
            return true;
        }
        // FIELD-ASSUMPTION: L4-decompose
        // A chunk is the widest whole number of bytes the field carries injectively.
        let chunk_bits = widest / 8 * 8;
        assert!(
            chunk_bits > 0,
            "this field carries no whole byte injectively"
        );
        if let Some(Radix::Dyn(radix)) = radix {
            let bytes = b.int_const(IntBits::from_u128(32, 256));
            b.assert_eq(radix, bytes);
        }

        let digits_bits = count * digit_bits;
        if digits_bits < bits {
            let amount = b.int_const(IntBits::from_u128(bits, digits_bits as u128));
            let rest = b.bin(BinaryArithOpKind::UShr, value, amount);
            let zero = b.int_const(IntBits::zero(bits));
            b.assert_eq(rest, zero);
        }

        let digit_type = Type::int(digit_bits);
        let mut little: Vec<ValueId> = Vec::with_capacity(count);
        for start in (0..digits_bits).step_by(chunk_bits) {
            let chunk_digits = (digits_bits - start).min(chunk_bits) / digit_bits;
            if start >= bits {
                let zero = b.int_const(IntBits::zero(digit_bits));
                little.extend(std::iter::repeat_n(zero, chunk_digits));
                continue;
            }
            let shifted = if start == 0 {
                value
            } else {
                let amount = b.int_const(IntBits::from_u128(bits, start as u128));
                b.bin(BinaryArithOpKind::UShr, value, amount)
            };
            let chunk_width = (bits - start).min(chunk_bits);
            let chunk = b.cast_to(CastTarget::Int(chunk_width), shifted);
            let image = b.cast_to_field(chunk);
            // The chunk is below `2^chunk_width`, so the digits past those its width needs are zero.
            let needed = chunk_width.div_ceil(digit_bits).min(chunk_digits);
            let decomposed = match radix {
                None => b.to_bits(image, Endianness::Little, needed),
                Some(_) => b.to_radix(image, Radix::Bytes, Endianness::Little, needed),
            };
            for index in 0..needed {
                let index = b.int_const(IntBits::from_u128(32, index as u128));
                little.push(b.array_get(decomposed, index));
            }
            let zero = b.int_const(IntBits::zero(digit_bits));
            little.extend(std::iter::repeat_n(zero, chunk_digits - needed));
        }
        debug_assert_eq!(little.len(), count);

        if endianness == Endianness::Big {
            little.reverse();
        }
        b.emit(OpCode::MkSeq {
            result,
            elems: little,
            seq_type: SequenceTargetType::Array(count),
            elem_type: digit_type,
        });
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{
        pass_manager::{AnalysisStore, Pass},
        passes::instruction_lowering::InstructionLowering,
        ssa::{Terminator, hlssa::HLSSA},
    };
    use mavros_artifacts::FieldConfig;

    /// `main(x: int(bits)) { to_bits(x, count) }` (or `to_radix` into bytes), run through the
    /// operand-shape rules. Returns the digit width of every field decomposition left behind.
    fn field_decompositions(bits: usize, bytes: bool, count: usize) -> Vec<usize> {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (value, result) = (ssa.fresh_value(), ssa.fresh_value());
        let function = ssa.get_unique_entrypoint_mut();
        let entry = function.get_entry_mut();
        entry.push_parameter(value, Type::int(bits));
        let digit_bits = if bytes { 8 } else { 1 };
        entry.push_test_instruction(if bytes {
            OpCode::ToRadix {
                result,
                value,
                radix: Radix::Bytes,
                endianness: Endianness::Little,
                count,
            }
        } else {
            OpCode::ToBits {
                result,
                value,
                endianness: Endianness::Little,
                count,
            }
        });
        function.add_return_type(Type::int(digit_bits).array_of(count));
        function
            .get_entry_mut()
            .set_terminator(Terminator::Return(vec![result]));

        InstructionLowering::operand_shapes().run(&mut ssa, &AnalysisStore::new());

        let mut widths = Vec::new();
        for (_, block) in ssa.get_unique_entrypoint().get_blocks() {
            for instruction in block.get_instructions() {
                match instruction {
                    OpCode::ToBits { count, .. } => widths.push(*count),
                    OpCode::ToRadix { count, .. } => widths.push(count * 8),
                    _ => {}
                }
            }
        }
        widths
    }

    /// Every field decomposition the rule leaves has a digit sum below the modulus, so its digits
    /// are unique; one more digit would also admit those of `x + p`.
    #[test]
    fn every_decomposition_left_has_one_answer() {
        let widest = widest_injective_int_bits(FieldConfig::bn254());
        for (bits, bytes, count) in [
            (64, false, 64),
            (64, false, 253),
            (64, false, 254),
            (8, false, 254),
            (253, false, 254),
            (64, true, 31),
            (64, true, 32),
            (300, false, 300),
            (300, true, 38),
            (64, false, 400),
        ] {
            let widths = field_decompositions(bits, bytes, count);
            assert!(
                !widths.is_empty(),
                "int{bits} into {count}: nothing decomposed"
            );
            for width in widths {
                assert!(
                    width <= widest,
                    "int{bits} into {count} {}: a {width}-bit field decomposition",
                    if bytes { "bytes" } else { "bits" }
                );
            }
        }
    }

    /// Where the digits fit, an integer is decomposed as its field image, in one decomposition.
    #[test]
    fn digits_that_fit_decompose_the_image_whole() {
        assert_eq!(field_decompositions(64, false, 253), vec![253]);
        assert_eq!(field_decompositions(64, true, 31), vec![248]);
    }
}
