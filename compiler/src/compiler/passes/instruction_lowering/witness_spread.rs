//! Lowers witness-tainted `Spread` and `Unspread` into witness hints plus spread lookups.
//!
//! The lookups are what reject a value with bits above the op's read, so a guarded one keeps its
//! guard: its lookups are flagged by the guard's condition, and hold only where it does.
//!
//! The later lookup spilling lowering lowers wide spread lookups into word-sized lookups. This
//! keeps word splitting in one place without making witness spread/unspread special there.

use crate::compiler::{
    analysis::types::FunctionTypeInfo,
    passes::instruction_lowering::{InstructionLoweringRule, LoweringContext},
    ssa::{
        ValueId,
        hlssa::{
            CastTarget, OpCode, Type, TypeExpr,
            builder::{HLBlockEmitter, HLEmitter},
        },
    },
};

pub struct LowerWitnessSpreadOps {}

impl InstructionLoweringRule for LowerWitnessSpreadOps {
    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        let types = context.types();
        let (guard, op) = match instruction {
            OpCode::Guard { condition, inner } => (Some(*condition), inner.as_ref()),
            other => (None, other),
        };
        let witnessed = |value: &ValueId| types.get_value_type(*value).is_witness_of();
        match op {
            OpCode::Spread {
                result,
                value,
                value_bits,
            } if witnessed(value) => {
                let flag = self.flag(b, types, guard);
                self.lower_witness_spread(b, types, flag, *result, *value, *value_bits);
                true
            }
            OpCode::Unspread {
                result_odd,
                result_even,
                value,
                value_bits,
            } if witnessed(value) => {
                let flag = self.flag(b, types, guard);
                self.lower_witness_unspread(
                    b,
                    types,
                    flag,
                    *result_odd,
                    *result_even,
                    *value,
                    *value_bits,
                );
                true
            }
            _ => false,
        }
    }
}

impl LowerWitnessSpreadOps {
    pub fn new() -> Self {
        Self {}
    }

    /// The flag the lookups are built under: the guard's condition, or one where there is none.
    fn flag(
        &self,
        b: &mut HLBlockEmitter<'_>,
        types: &FunctionTypeInfo,
        guard: Option<ValueId>,
    ) -> ValueId {
        match guard {
            Some(condition) => b.ensure_field(condition, types.get_value_type(condition)),
            None => b.field_const(b.field().one()),
        }
    }

    fn lower_witness_spread(
        &self,
        b: &mut HLBlockEmitter<'_>,
        function_type_info: &FunctionTypeInfo,
        flag: ValueId,
        result: ValueId,
        value: ValueId,
        value_bits: usize,
    ) {
        let spread_wit = self.write_spread_witness_and_lookup(b, flag, value, value_bits);
        b.emit(OpCode::Cast {
            result,
            value: spread_wit,
            target: cast_target_for_type(function_type_info.get_value_type(result)),
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_witness_unspread(
        &self,
        b: &mut HLBlockEmitter<'_>,
        function_type_info: &FunctionTypeInfo,
        flag: ValueId,
        result_odd: ValueId,
        result_even: ValueId,
        value: ValueId,
        value_bits: usize,
    ) {
        let value_pure = b.value_of(value);
        let (odd_hint, even_hint) = b.unspread(value_pure, value_bits);

        self.write_unspread_result(b, function_type_info, result_odd, odd_hint);
        self.write_unspread_result(b, function_type_info, result_even, even_hint);

        // The even stream starts at bit zero, so at an odd width it carries the one bit more. The
        // two lookups bound each stream at its own width and `value = 2·spread(odd) + spread(even)`
        // then pins `value` below `2^value_bits`, with its bits interleaved exactly once.
        let (odd_bits, even_bits) = (value_bits / 2, value_bits.div_ceil(2));
        let odd_spread = self.write_spread_witness_and_lookup(b, flag, result_odd, odd_bits);
        let two = b.field_const(b.field().constant(2));
        let two_odd_spread = b.umul(two, odd_spread);
        let value_field = b.cast_to_field(value);
        let even_spread = b.usub(value_field, two_odd_spread);

        let even_field = b.cast_to_field(result_even);
        b.lookup_spread(spread_table_width(even_bits), even_field, even_spread, flag);
    }

    fn write_unspread_result(
        &self,
        b: &mut HLBlockEmitter<'_>,
        function_type_info: &FunctionTypeInfo,
        result: ValueId,
        hint: ValueId,
    ) {
        let hint_field = b.cast_to_field(hint);
        let hint_wit = b.write_witness(hint_field);
        b.emit(OpCode::Cast {
            result,
            value: hint_wit,
            target: cast_target_for_type(function_type_info.get_value_type(result)),
        });
    }

    fn write_spread_witness_and_lookup(
        &self,
        b: &mut HLBlockEmitter<'_>,
        flag: ValueId,
        value: ValueId,
        value_bits: usize,
    ) -> ValueId {
        let value_pure = b.value_of(value);
        let value_field = b.cast_to_field(value);
        let spread_hint = b.spread(value_pure, value_bits);
        let spread_hint_field = b.cast_to_field(spread_hint);
        let spread_wit = b.write_witness(spread_hint_field);
        b.lookup_spread(
            spread_table_width(value_bits),
            value_field,
            spread_wit,
            flag,
        );
        spread_wit
    }
}

/// The width a spread lookup is keyed at, which lookup spilling then chunks into table-sized pieces.
///
/// A lookup carries its width as a `u8`, and that is enough here: the key and its spread are both a
/// single field element, so a width reaching this lowering is one whose spread the field holds.
fn spread_table_width(value_bits: usize) -> u8 {
    u8::try_from(value_bits)
        .unwrap_or_else(|_| ice!("a {value_bits}-bit spread has no single-element lookup"))
}

fn cast_target_for_type(ty: &Type) -> CastTarget {
    match ty.strip_all_witness().expr {
        // A `CastTarget` is a raw-bits conversion, so there is one target per width and no sign to
        // choose: `TypeExpr::Int(n)` says only "an n-bit integer", and `CastTarget::Int(n)` says
        // only "reinterpret at n bits". Sign extension is the separate `SExt` opcode.
        TypeExpr::Int(bits) => CastTarget::Int(bits),
        other => ice!(
            "Expected integer type for witness spread result, got {:?}",
            other
        ),
    }
}
