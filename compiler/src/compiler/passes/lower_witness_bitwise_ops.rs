//! Lowers witness-tainted bitwise operations before the main explicit-witness pass.
//!
//! This pass emits natural-width `Spread` operations for non-byte-aligned widths. Byte-aligned
//! widths preserve the old byte-decomposed lowering shape so this refactor does not perturb existing
//! bytecode/R1CS sizes. `ExplicitWitness` is responsible for spilling wide spreads into the smaller
//! spread lookups supported by the backend.

use std::collections::HashMap;

use ark_ff::Field as _;

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
            BinaryArithOpKind, CastTarget, Endianness, HLBlock, HLSSA, OpCode, Radix, TypeExpr,
            builder::{HLEmitter, HLInstrBuilder, HLSSABuilder},
        },
    },
};

const DIRECT_SPREAD_LOOKUP_BITS: u8 = 16;

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
            bits <= 64,
            "bitwise spread width too large for natural-width Spread lowering: {bits}"
        );

        if bits == 1 {
            self.lower_u1_bitwise(b, kind, result, lhs, rhs, lhs_witness, rhs_witness);
            return;
        }

        if bits % 8 == 0 {
            self.lower_byte_decomposed_bitwise(
                b,
                kind,
                lhs,
                rhs,
                lhs_witness,
                rhs_witness,
                bits,
                result,
            );
            return;
        }

        if bits < DIRECT_SPREAD_LOOKUP_BITS as usize {
            self.lower_small_bitwise(
                b,
                kind,
                result,
                lhs,
                rhs,
                bits as u8,
                lhs_witness,
                rhs_witness,
            );
            return;
        }

        let bits = bits as u8;

        let lhs_spread = spread_as_field(b, lhs, bits);
        let rhs_spread = spread_as_field(b, rhs, bits);
        let input_spread_sum = b.add(lhs_spread, rhs_spread);
        let input_spread_sum = b.cast_to(CastTarget::U(bits as usize * 2), input_spread_sum);
        let (and_wit, xor_wit) = b.unspread(input_spread_sum, bits);

        let result_word = match kind {
            BinaryArithOpKind::And => and_wit,
            BinaryArithOpKind::Xor => xor_wit,
            BinaryArithOpKind::Or => {
                let and_field = b.cast_to_field(and_wit);
                let xor_field = b.cast_to_field(xor_wit);
                b.add(and_field, xor_field)
            }
            _ => unreachable!(),
        };

        b.push(OpCode::Cast {
            result,
            value: result_word,
            target: CastTarget::U(bits as usize),
        });
    }

    fn lower_u1_bitwise(
        &self,
        b: &mut HLInstrBuilder<'_>,
        kind: BinaryArithOpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        lhs_witness: bool,
        rhs_witness: bool,
    ) {
        let target = CastTarget::U(1);
        let lhs_field = b.cast_to_field(lhs);
        let rhs_field = b.cast_to_field(rhs);

        let result_field = match kind {
            BinaryArithOpKind::And => {
                if lhs_witness && rhs_witness {
                    let lhs_pure = b.value_of(lhs);
                    let rhs_pure = b.value_of(rhs);
                    let result_hint = b.and(lhs_pure, rhs_pure);
                    let result_hint_field = b.cast_to_field(result_hint);
                    let result_wit = b.write_witness(result_hint_field);
                    b.constrain(lhs_field, rhs_field, result_wit);
                    result_wit
                } else {
                    b.mul(lhs_field, rhs_field)
                }
            }
            BinaryArithOpKind::Or => {
                let sum = b.add(lhs_field, rhs_field);
                if lhs_witness && rhs_witness {
                    let lhs_pure = b.value_of(lhs_field);
                    let rhs_pure = b.value_of(rhs_field);
                    let product_hint = b.mul(lhs_pure, rhs_pure);
                    let product_wit = b.write_witness(product_hint);
                    b.constrain(lhs_field, rhs_field, product_wit);
                    b.sub(sum, product_wit)
                } else {
                    let product = b.mul(lhs_field, rhs_field);
                    b.sub(sum, product)
                }
            }
            BinaryArithOpKind::Xor => {
                let sum = b.add(lhs_field, rhs_field);
                let two = b.field_const(Field::from(2));
                if lhs_witness && rhs_witness {
                    let lhs_pure = b.value_of(lhs_field);
                    let rhs_pure = b.value_of(rhs_field);
                    let product_hint = b.mul(lhs_pure, rhs_pure);
                    let product_wit = b.write_witness(product_hint);
                    b.constrain(lhs_field, rhs_field, product_wit);
                    let two_product = b.mul(two, product_wit);
                    b.sub(sum, two_product)
                } else {
                    let product = b.mul(lhs_field, rhs_field);
                    let two_product = b.mul(two, product);
                    b.sub(sum, two_product)
                }
            }
            _ => unreachable!(),
        };

        b.push(OpCode::Cast {
            result,
            value: result_field,
            target,
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_small_bitwise(
        &self,
        b: &mut HLInstrBuilder<'_>,
        kind: BinaryArithOpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        bits: u8,
        lhs_witness: bool,
        rhs_witness: bool,
    ) {
        let one = b.field_const(Field::ONE);
        let two = b.field_const(Field::from(2));
        let lhs_pure = if lhs_witness { b.value_of(lhs) } else { lhs };
        let rhs_pure = if rhs_witness { b.value_of(rhs) } else { rhs };

        let lhs_spread = small_spread_as_field(b, lhs, lhs_pure, bits, lhs_witness, one);
        let rhs_spread = small_spread_as_field(b, rhs, rhs_pure, bits, rhs_witness, one);
        let input_spread_sum = b.add(lhs_spread, rhs_spread);

        let and_hint = b.and(lhs_pure, rhs_pure);
        let xor_hint = b.xor(lhs_pure, rhs_pure);
        let (and_wit, and_spread) = write_small_spread_witness(b, and_hint, bits, one);

        let xor_field = b.cast_to_field(xor_hint);
        let xor_wit = b.write_witness(xor_field);
        let two_and_spread = b.mul(two, and_spread);
        let xor_spread = b.sub(input_spread_sum, two_and_spread);
        b.lookup_spread(bits, xor_wit, xor_spread, one);

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
    fn lower_byte_decomposed_bitwise(
        &self,
        b: &mut HLInstrBuilder<'_>,
        kind: BinaryArithOpKind,
        lhs: ValueId,
        rhs: ValueId,
        lhs_witness: bool,
        rhs_witness: bool,
        bits: usize,
        result: ValueId,
    ) {
        assert!(bits % 8 == 0 && bits >= 8 && bits <= 64);
        let chunks = bits / 8;
        let one = b.field_const(Field::ONE);
        let zero = b.field_const(Field::from(0));
        let two = b.field_const(Field::from(2));
        let two_to_8 = b.field_const(Field::from(256u128));
        let two_to_16 = b.field_const(Field::from(1u128 << 16));

        let lhs_pure = if lhs_witness { b.value_of(lhs) } else { lhs };
        let rhs_pure = if rhs_witness { b.value_of(rhs) } else { rhs };
        let lhs_field = b.cast_to_field(lhs);
        let rhs_field = b.cast_to_field(rhs);

        let (lhs_bytes, lhs_spread) =
            self.spread_decompose(b, lhs_pure, lhs_field, chunks, one, lhs_witness);
        let (rhs_bytes, rhs_spread) =
            self.spread_decompose(b, rhs_pure, rhs_field, chunks, one, rhs_witness);

        let mut and_word = zero;
        let mut xor_word = zero;
        let mut and_spread = zero;
        let mut xor_spread = zero;
        for i in 0..chunks {
            let lhs_byte = b.cast_to(CastTarget::U(8), lhs_bytes[i]);
            let rhs_byte = b.cast_to(CastTarget::U(8), rhs_bytes[i]);
            let and_hint = b.and(lhs_byte, rhs_byte);
            let xor_hint = b.xor(lhs_byte, rhs_byte);

            let (and_byte, and_byte_spread) = self.write_spread_byte_witness(b, and_hint, one);
            let shifted_and_word = b.mul(and_word, two_to_8);
            and_word = b.add(shifted_and_word, and_byte);
            let shifted_and_spread = b.mul(and_spread, two_to_16);
            and_spread = b.add(shifted_and_spread, and_byte_spread);

            if i < chunks - 1 {
                let (xor_byte, xor_byte_spread) = self.write_spread_byte_witness(b, xor_hint, one);
                let shifted_xor_word = b.mul(xor_word, two_to_8);
                xor_word = b.add(shifted_xor_word, xor_byte);
                let shifted_xor_spread = b.mul(xor_spread, two_to_16);
                xor_spread = b.add(shifted_xor_spread, xor_byte_spread);
            } else {
                let xor_field = b.cast_to_field(xor_hint);
                let xor_byte = b.write_witness(xor_field);
                let shifted_xor_word = b.mul(xor_word, two_to_8);
                xor_word = b.add(shifted_xor_word, xor_byte);

                let input_spread = b.add(lhs_spread, rhs_spread);
                let two_and_spread = b.mul(two, and_spread);
                let remaining_after_and = b.sub(input_spread, two_and_spread);
                let shifted_xor_spread = b.mul(xor_spread, two_to_16);
                let xor_spread_last = b.sub(remaining_after_and, shifted_xor_spread);
                b.lookup_spread(8, xor_byte, xor_spread_last, one);
            }
        }

        let result_word = match kind {
            BinaryArithOpKind::And => and_word,
            BinaryArithOpKind::Xor => xor_word,
            BinaryArithOpKind::Or => b.add(and_word, xor_word),
            _ => unreachable!(),
        };

        b.push(OpCode::Cast {
            result,
            value: result_word,
            target: CastTarget::U(bits),
        });
    }

    fn spread_decompose(
        &self,
        b: &mut HLInstrBuilder<'_>,
        pure_value: ValueId,
        field_value: ValueId,
        chunks: usize,
        one: ValueId,
        is_witness: bool,
    ) -> (Vec<ValueId>, ValueId) {
        let zero = b.field_const(Field::from(0));
        let two_to_8 = b.field_const(Field::from(256u128));
        let two_to_16 = b.field_const(Field::from(1u128 << 16));
        let pure_field = b.cast_to_field(pure_value);
        let bytes = b.to_radix(pure_field, Radix::Bytes, Endianness::Big, chunks);
        let mut pure_bytes = Vec::with_capacity(chunks);

        if !is_witness {
            let mut spread = zero;
            for i in 0..chunks {
                let idx = b.u_const(32, i as u128);
                let byte = b.array_get(bytes, idx);
                pure_bytes.push(byte);

                let byte_spread = b.spread(byte, 8);
                let byte_spread = b.cast_to_field(byte_spread);
                let shifted_spread = b.mul(spread, two_to_16);
                spread = b.add(shifted_spread, byte_spread);
            }
            return (pure_bytes, b.write_witness(spread));
        }

        let mut reconstructed_value = zero;
        let mut reconstructed_spread = zero;
        for i in 0..chunks - 1 {
            let idx = b.u_const(32, i as u128);
            let byte = b.array_get(bytes, idx);
            let (byte_wit, spread_wit) = self.write_spread_byte_witness(b, byte, one);

            let shifted_value = b.mul(reconstructed_value, two_to_8);
            reconstructed_value = b.add(shifted_value, byte_wit);
            let shifted_spread = b.mul(reconstructed_spread, two_to_16);
            reconstructed_spread = b.add(shifted_spread, spread_wit);
            pure_bytes.push(byte);
        }

        let last_idx = b.u_const(32, (chunks - 1) as u128);
        let last_byte_pure = b.array_get(bytes, last_idx);
        let shifted_value = b.mul(reconstructed_value, two_to_8);
        let last_byte = b.sub(field_value, shifted_value);
        let last_spread = b.spread(last_byte_pure, 8);
        let last_spread_field = b.cast_to_field(last_spread);
        let last_spread = b.write_witness(last_spread_field);
        b.lookup_spread(8, last_byte, last_spread, one);

        let shifted_spread = b.mul(reconstructed_spread, two_to_16);
        reconstructed_spread = b.add(shifted_spread, last_spread);
        pure_bytes.push(last_byte_pure);

        (pure_bytes, reconstructed_spread)
    }

    fn write_spread_byte_witness(
        &self,
        b: &mut HLInstrBuilder<'_>,
        byte: ValueId,
        one: ValueId,
    ) -> (ValueId, ValueId) {
        let spread = b.spread(byte, 8);
        let byte_field = b.cast_to_field(byte);
        let byte = b.write_witness(byte_field);
        let spread_field = b.cast_to_field(spread);
        let spread = b.write_witness(spread_field);
        b.lookup_spread(8, byte, spread, one);
        (byte, spread)
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

fn small_spread_as_field(
    b: &mut HLInstrBuilder<'_>,
    value: ValueId,
    pure_value: ValueId,
    bits: u8,
    is_witness: bool,
    one: ValueId,
) -> ValueId {
    let spread = b.spread(pure_value, bits);
    let spread_field = b.cast_to_field(spread);
    if !is_witness {
        return spread_field;
    }

    let input_field = b.cast_to_field(value);
    let spread_wit = b.write_witness(spread_field);
    b.lookup_spread(bits, input_field, spread_wit, one);
    spread_wit
}

fn write_small_spread_witness(
    b: &mut HLInstrBuilder<'_>,
    value: ValueId,
    bits: u8,
    one: ValueId,
) -> (ValueId, ValueId) {
    let value_field = b.cast_to_field(value);
    let value_wit = b.write_witness(value_field);
    let spread = b.spread(value, bits);
    let spread_field = b.cast_to_field(spread);
    let spread_wit = b.write_witness(spread_field);
    b.lookup_spread(bits, value_wit, spread_wit, one);
    (value_wit, spread_wit)
}
