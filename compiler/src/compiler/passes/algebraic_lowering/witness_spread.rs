//! Lowers witness-tainted `Spread` and `Unspread` into witness hints plus spread lookups.
//!
//! The later `ExplicitWitness` pass lowers wide spread lookups into word-sized lookups. This keeps
//! word splitting in one place without making witness spread/unspread special there.

use ark_ff::Field as _;

use crate::compiler::{
    Field,
    analysis::types::FunctionTypeInfo,
    ssa::{
        ValueId,
        hlssa::{
            CastTarget, OpCode, Type, TypeExpr,
            builder::{HLBlockEmitter, HLEmitter},
        },
    },
};

use super::{AlgebraicLoweringRule, LoweringContext};

pub struct LowerWitnessSpreadOps {}

impl AlgebraicLoweringRule for LowerWitnessSpreadOps {
    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        let function_type_info = context.types();
        match instruction {
            OpCode::Spread {
                result,
                value,
                bits,
            } => {
                if function_type_info.get_value_type(*value).is_witness_of() {
                    self.lower_witness_spread(b, function_type_info, *result, *value, *bits);
                    true
                } else {
                    false
                }
            }
            OpCode::Unspread {
                result_odd,
                result_even,
                value,
                bits,
            } => {
                if function_type_info.get_value_type(*value).is_witness_of() {
                    self.lower_witness_unspread(
                        b,
                        function_type_info,
                        *result_odd,
                        *result_even,
                        *value,
                        *bits,
                    );
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }
}

impl LowerWitnessSpreadOps {
    pub fn new() -> Self {
        Self {}
    }

    fn lower_witness_spread(
        &self,
        b: &mut HLBlockEmitter<'_>,
        function_type_info: &FunctionTypeInfo,
        result: ValueId,
        value: ValueId,
        bits: u8,
    ) {
        let spread_wit = self.write_spread_witness_and_lookup(b, value, bits);
        b.emit(OpCode::Cast {
            result,
            value: spread_wit,
            target: cast_target_for_type(function_type_info.get_value_type(result)),
        });
    }

    fn lower_witness_unspread(
        &self,
        b: &mut HLBlockEmitter<'_>,
        function_type_info: &FunctionTypeInfo,
        result_odd: ValueId,
        result_even: ValueId,
        value: ValueId,
        bits: u8,
    ) {
        let value_pure = b.value_of(value);
        let (odd_hint, even_hint) = b.unspread(value_pure, bits);

        self.write_unspread_result(b, function_type_info, result_odd, odd_hint);
        self.write_unspread_result(b, function_type_info, result_even, even_hint);

        let odd_spread = self.write_spread_witness_and_lookup(b, result_odd, bits);
        let two = b.field_const(Field::from(2));
        let two_odd_spread = b.mul(two, odd_spread);
        let value_field = b.cast_to_field(value);
        let even_spread = b.sub(value_field, two_odd_spread);

        let even_field = b.cast_to_field(result_even);
        let one = b.field_const(Field::ONE);
        b.lookup_spread(bits, even_field, even_spread, one);
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
        value: ValueId,
        bits: u8,
    ) -> ValueId {
        let value_pure = b.value_of(value);
        let value_field = b.cast_to_field(value);
        let spread_hint = b.spread(value_pure, bits);
        let spread_hint_field = b.cast_to_field(spread_hint);
        let spread_wit = b.write_witness(spread_hint_field);
        let one = b.field_const(Field::ONE);
        b.lookup_spread(bits, value_field, spread_wit, one);
        spread_wit
    }
}

fn cast_target_for_type(ty: &Type) -> CastTarget {
    match ty.strip_all_witness().expr {
        TypeExpr::U(bits) => CastTarget::U(bits),
        TypeExpr::I(bits) => CastTarget::I(bits),
        other => panic!(
            "Expected integer type for witness spread result, got {:?}",
            other
        ),
    }
}
