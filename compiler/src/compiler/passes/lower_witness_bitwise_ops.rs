//! Lowers witness-tainted bitwise operations before the main explicit-witness pass.
//!
//! This pass emits `Spread`/`Unspread` operations, except for `u64` bitwise ops where it keeps a
//! two-limb `u32` decomposition. `ExplicitWitness` is responsible for the witness writes and lookup
//! constraints needed by those bitwise operations.

use ark_ff::Field as _;

use crate::compiler::{
    Field,
    analysis::{flow_analysis::FlowAnalysis, types::FunctionTypeInfo},
    pass_manager::AnalysisId,
    ssa::{
        ValueId,
        hlssa::{
            BinaryArithOpKind, CastTarget, OpCode, TypeExpr,
            builder::{HLBlockEmitter, HLEmitter},
        },
    },
};

use super::lowering_pass::LoweringPass;

pub struct LowerWitnessBitwiseOps {}

impl LoweringPass for LowerWitnessBitwiseOps {
    const NAME: &'static str = "lower_witness_bitwise_ops";

    fn preserved_analyses(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }

    fn process_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        function_type_info: &FunctionTypeInfo,
        instruction: OpCode,
    ) {
        match instruction {
            OpCode::BinaryArithOp {
                kind:
                    kind @ (BinaryArithOpKind::And | BinaryArithOpKind::Or | BinaryArithOpKind::Xor),
                result,
                lhs,
                rhs,
            } => {
                let lhs_witness = function_type_info.get_value_type(lhs).is_witness_of();
                let rhs_witness = function_type_info.get_value_type(rhs).is_witness_of();
                if lhs_witness || rhs_witness {
                    self.lower_binary_bitwise(
                        b,
                        function_type_info,
                        kind,
                        result,
                        lhs,
                        rhs,
                        lhs_witness,
                        rhs_witness,
                    );
                } else {
                    b.emit(OpCode::BinaryArithOp {
                        kind,
                        result,
                        lhs,
                        rhs,
                    });
                }
            }
            OpCode::Not { result, value } => {
                self.lower_not(b, function_type_info, result, value);
            }
            other => b.emit(other),
        }
    }
}

impl LowerWitnessBitwiseOps {
    pub fn new() -> Self {
        Self {}
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_binary_bitwise(
        &self,
        b: &mut HLBlockEmitter<'_>,
        function_type_info: &FunctionTypeInfo,
        kind: BinaryArithOpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        lhs_witness: bool,
        rhs_witness: bool,
    ) {
        let bits = unsigned_bits(function_type_info, lhs, "bitwise operand");
        assert!(
            bits <= 128,
            "bitwise spread width too large for natural-width Spread lowering: {bits}"
        );

        if bits == 1 {
            self.lower_u1_bitwise(b, kind, result, lhs, rhs);
            return;
        }

        let result_word = if bits == 64 {
            let lhs_limbs = decompose_u64_input(b, lhs, lhs_witness);
            let rhs_limbs = decompose_u64_input(b, rhs, rhs_witness);
            let result_limbs = lower_u64_limb_bitwise(b, kind, lhs_limbs, rhs_limbs);
            combine_u32_limbs(b, result_limbs)
        } else {
            lower_word_bitwise(b, kind, lhs, rhs, bits as u8)
        };

        b.emit(OpCode::Cast {
            result,
            value: result_word,
            target: CastTarget::U(bits),
        });
    }

    fn lower_u1_bitwise(
        &self,
        b: &mut HLBlockEmitter<'_>,
        kind: BinaryArithOpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
    ) {
        let target = CastTarget::U(1);
        let lhs_field = b.cast_to_field(lhs);
        let rhs_field = b.cast_to_field(rhs);

        let result_field = match kind {
            BinaryArithOpKind::And => b.mul(lhs_field, rhs_field),
            BinaryArithOpKind::Or => {
                let sum = b.add(lhs_field, rhs_field);
                let product = b.mul(lhs_field, rhs_field);
                b.sub(sum, product)
            }
            BinaryArithOpKind::Xor => {
                let sum = b.add(lhs_field, rhs_field);
                let two = b.field_const(Field::from(2));
                let product = b.mul(lhs_field, rhs_field);
                let two_product = b.mul(two, product);
                b.sub(sum, two_product)
            }
            _ => unreachable!(),
        };

        b.emit(OpCode::Cast {
            result,
            value: result_field,
            target,
        });
    }

    fn lower_not(
        &self,
        b: &mut HLBlockEmitter<'_>,
        function_type_info: &FunctionTypeInfo,
        result: ValueId,
        value: ValueId,
    ) {
        let (bits, cast_target) = integer_bits_and_cast(function_type_info, value, "bitwise not");
        let ones = b.field_const((Field::from(2).pow([bits as u64])) - Field::ONE);
        let value_field = b.cast_to_field(value);
        let not_value = b.sub(ones, value_field);
        b.emit(OpCode::Cast {
            result,
            value: not_value,
            target: cast_target,
        });
    }
}

#[derive(Clone, Copy)]
struct U64Limbs {
    lo: ValueId,
    hi: ValueId,
}

fn unsigned_bits(function_type_info: &FunctionTypeInfo, value: ValueId, context: &str) -> usize {
    match function_type_info
        .get_value_type(value)
        .strip_witness()
        .expr
    {
        TypeExpr::U(bits) => bits,
        other => panic!("{context}: expected unsigned integer type, got {:?}", other),
    }
}

fn integer_bits_and_cast(
    function_type_info: &FunctionTypeInfo,
    value: ValueId,
    context: &str,
) -> (usize, CastTarget) {
    match function_type_info
        .get_value_type(value)
        .strip_witness()
        .expr
    {
        TypeExpr::U(bits) => (bits, CastTarget::U(bits)),
        TypeExpr::I(bits) => (bits, CastTarget::I(bits)),
        other => panic!("{context}: expected integer type, got {:?}", other),
    }
}

fn spread_as_field(b: &mut impl HLEmitter, value: ValueId, bits: u8) -> ValueId {
    let spread = b.spread(value, bits);
    b.cast_to_field(spread)
}

fn lower_word_bitwise(
    b: &mut impl HLEmitter,
    kind: BinaryArithOpKind,
    lhs: ValueId,
    rhs: ValueId,
    bits: u8,
) -> ValueId {
    let lhs_spread = spread_as_field(b, lhs, bits);
    let rhs_spread = spread_as_field(b, rhs, bits);
    let input_spread_sum = b.add(lhs_spread, rhs_spread);
    let input_spread_sum = b.cast_to(CastTarget::U(bits as usize * 2), input_spread_sum);
    let (and_word, xor_word) = b.unspread(input_spread_sum, bits);

    match kind {
        BinaryArithOpKind::And => and_word,
        BinaryArithOpKind::Xor => xor_word,
        BinaryArithOpKind::Or => b.add(and_word, xor_word),
        _ => unreachable!(),
    }
}

fn lower_u64_limb_bitwise(
    b: &mut impl HLEmitter,
    kind: BinaryArithOpKind,
    lhs: U64Limbs,
    rhs: U64Limbs,
) -> U64Limbs {
    U64Limbs {
        lo: lower_word_bitwise(b, kind, lhs.lo, rhs.lo, 32),
        hi: lower_word_bitwise(b, kind, lhs.hi, rhs.hi, 32),
    }
}

fn combine_u32_limbs(b: &mut impl HLEmitter, limbs: U64Limbs) -> ValueId {
    let lo = b.cast_to_field(limbs.lo);
    let hi = b.cast_to_field(limbs.hi);
    let shift = b.field_const(Field::from(1u128 << 32));
    let shifted_hi = b.mul(hi, shift);
    b.add(lo, shifted_hi)
}

fn decompose_u64_input(b: &mut impl HLEmitter, value: ValueId, is_witness: bool) -> U64Limbs {
    if !is_witness {
        return extract_u64_limbs(b, value);
    }

    let pure_value = b.value_of(value);
    let hi_hint = extract_u64_limb(b, pure_value, 32);
    let hi_field = b.cast_to_field(hi_hint);
    let hi_wit = b.write_witness(hi_field);
    let lo = derive_low_u32_limb(b, value, hi_wit);

    U64Limbs {
        lo,
        hi: b.cast_to(CastTarget::U(32), hi_wit),
    }
}

fn extract_u64_limbs(b: &mut impl HLEmitter, value: ValueId) -> U64Limbs {
    U64Limbs {
        lo: extract_u64_limb(b, value, 0),
        hi: extract_u64_limb(b, value, 32),
    }
}

fn extract_u64_limb(b: &mut impl HLEmitter, value: ValueId, offset: usize) -> ValueId {
    let shifted = if offset == 0 {
        value
    } else {
        let divisor = b.u_const(64, 1u128 << offset);
        b.div(value, divisor)
    };
    let modulus = b.u_const(64, 1u128 << 32);
    let limb = b.modulo(shifted, modulus);
    b.cast_to(CastTarget::U(32), limb)
}

fn derive_low_u32_limb(b: &mut impl HLEmitter, value: ValueId, hi_field: ValueId) -> ValueId {
    let value_field = b.cast_to_field(value);
    let shift = b.field_const(Field::from(1u128 << 32));
    let shifted_hi = b.mul(hi_field, shift);
    let lo_field = b.sub(value_field, shifted_hi);
    b.cast_to(CastTarget::U(32), lo_field)
}
