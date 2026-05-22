//! Lowers witness-tainted bitwise operations before the main explicit-witness pass.
//!
//! This pass emits natural-width `Spread` operations, except for `u64` bitwise ops where it uses a
//! two-limb `u32` decomposition. `ExplicitWitness` is responsible for spilling wide spreads into
//! the smaller spread lookups supported by the backend.

use std::collections::HashMap;

use ark_ff::{AdditiveGroup, Field as _};

use crate::compiler::{
    Field,
    analysis::{
        flow_analysis::FlowAnalysis,
        types::{FunctionTypeInfo, TypeInfo},
    },
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::{
        BlockId, ValueId,
        hlssa::{
            BinaryArithOpKind, CastTarget, HLBlock, HLSSA, OpCode, TypeExpr,
            builder::{HLEmitter, HLInstrBuilder, HLSSABuilder},
        },
    },
};

pub struct LowerWitnessBitwiseOps {}

impl Pass for LowerWitnessBitwiseOps {
    fn name(&self) -> &'static str {
        "lower_witness_bitwise_ops"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![TypeInfo::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        self.do_run(ssa, store.get::<TypeInfo>());
    }

    fn preserves(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }
}

impl LowerWitnessBitwiseOps {
    pub fn new() -> Self {
        Self {}
    }

    fn do_run(&self, ssa: &mut HLSSA, type_info: &TypeInfo) {
        let fids: Vec<_> = ssa.get_function_ids().collect();
        let mut sb = HLSSABuilder::new(ssa);
        for function_id in fids {
            let function_type_info = type_info.get_function(function_id);
            sb.modify_function(function_id, |fb| {
                let mut new_blocks = HashMap::<BlockId, HLBlock>::new();
                for (bid, mut block) in fb.function.take_blocks().into_iter() {
                    let mut new_instructions = Vec::new();
                    for instruction in block.take_instructions().into_iter() {
                        let b =
                            &mut HLInstrBuilder::new(fb.function, fb.ssa, &mut new_instructions);
                        self.process_instruction(b, function_type_info, instruction);
                    }
                    block.put_instructions(new_instructions);
                    new_blocks.insert(bid, block);
                }
                fb.function.put_blocks(new_blocks);
            });
        }
    }

    fn process_instruction(
        &self,
        b: &mut HLInstrBuilder<'_>,
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
                    b.push(OpCode::BinaryArithOp {
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
            other => b.push(other),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_binary_bitwise(
        &self,
        b: &mut HLInstrBuilder<'_>,
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

        if bits == 64 {
            self.lower_binary_bitwise_u64(b, kind, result, lhs, rhs, lhs_witness, rhs_witness);
            return;
        }

        let bits = bits as u8;

        let one = b.field_const(Field::ONE);
        let zero = b.field_const(Field::ZERO);
        let two = b.field_const(Field::from(2));

        let lhs_pure = if lhs_witness { b.value_of(lhs) } else { lhs };
        let rhs_pure = if rhs_witness { b.value_of(rhs) } else { rhs };

        let and_hint = b.and(lhs_pure, rhs_pure);
        let xor_hint = b.xor(lhs_pure, rhs_pure);
        let and_hint_field = b.cast_to_field(and_hint);
        let xor_hint_field = b.cast_to_field(xor_hint);
        let and_wit = b.write_witness(and_hint_field);
        let xor_wit = b.write_witness(xor_hint_field);
        let and_wit_int = b.cast_to(CastTarget::U(bits as usize), and_wit);
        let xor_wit_int = b.cast_to(CastTarget::U(bits as usize), xor_wit);

        let lhs_spread = spread_as_field(b, lhs, bits);
        let rhs_spread = spread_as_field(b, rhs, bits);
        let and_spread = spread_as_field(b, and_wit_int, bits);
        let xor_spread = spread_as_field(b, xor_wit_int, bits);

        let input_spread_sum = b.add(lhs_spread, rhs_spread);
        let two_and_spread = b.mul(two, and_spread);
        let output_spread_sum = b.add(two_and_spread, xor_spread);
        let spread_diff = b.sub(input_spread_sum, output_spread_sum);
        b.constrain(spread_diff, one, zero);

        let result_word = match kind {
            BinaryArithOpKind::And => and_wit,
            BinaryArithOpKind::Xor => xor_wit,
            BinaryArithOpKind::Or => b.add(and_wit, xor_wit),
            _ => unreachable!(),
        };

        b.push(OpCode::Cast {
            result,
            value: result_word,
            target: CastTarget::U(bits as usize),
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_binary_bitwise_u64(
        &self,
        b: &mut HLInstrBuilder<'_>,
        kind: BinaryArithOpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        lhs_witness: bool,
        rhs_witness: bool,
    ) {
        let lhs_pure = if lhs_witness { b.value_of(lhs) } else { lhs };
        let rhs_pure = if rhs_witness { b.value_of(rhs) } else { rhs };

        let lhs_limbs = decompose_u64_input(b, lhs, lhs_pure, lhs_witness);
        let rhs_limbs = decompose_u64_input(b, rhs, rhs_pure, rhs_witness);

        let and_hint = b.and(lhs_pure, rhs_pure);
        let xor_hint = b.xor(lhs_pure, rhs_pure);
        let and_hint = write_u64_hint(b, and_hint);
        let xor_hint = write_u64_hint(b, xor_hint);

        constrain_u32_limb_bitwise_identity(
            b,
            lhs_limbs.lo,
            rhs_limbs.lo,
            and_hint.limbs.lo,
            xor_hint.limbs.lo,
        );
        constrain_u32_limb_bitwise_identity(
            b,
            lhs_limbs.hi,
            rhs_limbs.hi,
            and_hint.limbs.hi,
            xor_hint.limbs.hi,
        );

        let result_word = match kind {
            BinaryArithOpKind::And => and_hint.word,
            BinaryArithOpKind::Xor => xor_hint.word,
            BinaryArithOpKind::Or => b.add(and_hint.word, xor_hint.word),
            _ => unreachable!(),
        };

        b.push(OpCode::Cast {
            result,
            value: result_word,
            target: CastTarget::U(64),
        });
    }

    fn lower_not(
        &self,
        b: &mut HLInstrBuilder<'_>,
        function_type_info: &FunctionTypeInfo,
        result: ValueId,
        value: ValueId,
    ) {
        let (bits, cast_target) = integer_bits_and_cast(function_type_info, value, "bitwise not");
        let ones = b.field_const((Field::from(2).pow([bits as u64])) - Field::ONE);
        let value_field = b.cast_to_field(value);
        let not_value = b.sub(ones, value_field);
        b.push(OpCode::Cast {
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

#[derive(Clone, Copy)]
struct U64Hint {
    word: ValueId,
    limbs: U64Limbs,
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

fn spread_as_field(b: &mut HLInstrBuilder<'_>, value: ValueId, bits: u8) -> ValueId {
    let spread = b.spread(value, bits);
    b.cast_to_field(spread)
}

fn decompose_u64_input(
    b: &mut HLInstrBuilder<'_>,
    value: ValueId,
    pure_value: ValueId,
    is_witness: bool,
) -> U64Limbs {
    let pure_limbs = extract_u64_limbs(b, pure_value);
    if !is_witness {
        return pure_limbs;
    }

    let hi_field = b.cast_to_field(pure_limbs.hi);
    let hi_wit = b.write_witness(hi_field);
    let lo = derive_low_u32_limb(b, value, hi_wit);

    U64Limbs {
        lo,
        hi: b.cast_to(CastTarget::U(32), hi_wit),
    }
}

fn write_u64_hint(b: &mut HLInstrBuilder<'_>, hint: ValueId) -> U64Hint {
    let hint_field = b.cast_to_field(hint);
    let word_wit = b.write_witness(hint_field);
    let pure_limbs = extract_u64_limbs(b, hint);
    let hi_field = b.cast_to_field(pure_limbs.hi);
    let hi_wit = b.write_witness(hi_field);
    let lo = derive_low_u32_limb_from_field(b, word_wit, hi_wit);

    U64Hint {
        word: word_wit,
        limbs: U64Limbs {
            lo,
            hi: b.cast_to(CastTarget::U(32), hi_wit),
        },
    }
}

fn extract_u64_limbs(b: &mut HLInstrBuilder<'_>, value: ValueId) -> U64Limbs {
    U64Limbs {
        lo: extract_u64_limb(b, value, 0),
        hi: extract_u64_limb(b, value, 32),
    }
}

fn extract_u64_limb(b: &mut HLInstrBuilder<'_>, value: ValueId, offset: usize) -> ValueId {
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

fn derive_low_u32_limb(b: &mut HLInstrBuilder<'_>, value: ValueId, hi_field: ValueId) -> ValueId {
    let value_field = b.cast_to_field(value);
    derive_low_u32_limb_from_field(b, value_field, hi_field)
}

fn derive_low_u32_limb_from_field(
    b: &mut HLInstrBuilder<'_>,
    value_field: ValueId,
    hi_field: ValueId,
) -> ValueId {
    let shift = b.field_const(Field::from(1u128 << 32));
    let shifted_hi = b.mul(hi_field, shift);
    let lo_field = b.sub(value_field, shifted_hi);
    b.cast_to(CastTarget::U(32), lo_field)
}

fn constrain_u32_limb_bitwise_identity(
    b: &mut HLInstrBuilder<'_>,
    lhs: ValueId,
    rhs: ValueId,
    and_value: ValueId,
    xor_value: ValueId,
) {
    let one = b.field_const(Field::ONE);
    let zero = b.field_const(Field::ZERO);
    let two = b.field_const(Field::from(2));
    let lhs_spread = spread_as_field(b, lhs, 32);
    let rhs_spread = spread_as_field(b, rhs, 32);
    let and_spread = spread_as_field(b, and_value, 32);
    let xor_spread = spread_as_field(b, xor_value, 32);

    let input_spread_sum = b.add(lhs_spread, rhs_spread);
    let two_and_spread = b.mul(two, and_spread);
    let output_spread_sum = b.add(two_and_spread, xor_spread);
    let spread_diff = b.sub(input_spread_sum, output_spread_sum);
    b.constrain(spread_diff, one, zero);
}
