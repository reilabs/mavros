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
            CastTarget, Endianness, HLBlock, HLSSA, LookupTarget, OpCode, Radix,
            builder::{HLEmitter, HLInstrBuilder, HLSSABuilder},
        },
    },
};

const SPREAD_SPILL_THRESHOLD_BITS: u8 = 16;

pub struct LookupSpilling {}

impl Pass for LookupSpilling {
    fn name(&self) -> &'static str {
        "lookup_spilling"
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

impl LookupSpilling {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(&self, ssa: &mut HLSSA, type_info: &TypeInfo) {
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
            OpCode::Lookup {
                target: LookupTarget::Rangecheck(bits),
                args,
                flag,
            } => {
                assert_eq!(args.len(), 1, "Rangecheck lookup must have exactly one key");
                self.spill_rangecheck_bits(b, args[0], bits as usize, flag);
            }
            OpCode::Lookup {
                target: LookupTarget::Spread(bits),
                args,
                flag,
            } if bits >= SPREAD_SPILL_THRESHOLD_BITS => {
                assert_eq!(
                    args.len(),
                    2,
                    "Spread lookup must have exactly one key and one result"
                );
                self.spill_wide_spread_lookup(b, function_type_info, args[0], args[1], flag, bits);
            }
            instruction => b.push(instruction),
        }
    }

    /// Rangecheck `value ∈ [0, 2^bits)` for any `bits ≥ 1`.
    fn spill_rangecheck_bits(
        &self,
        b: &mut HLInstrBuilder<'_>,
        value: ValueId,
        bits: usize,
        flag: ValueId,
    ) {
        assert!(bits >= 1, "rangecheck width must be at least 1 bit");

        if bits == 1 {
            let one = b.field_const(Field::ONE);
            if flag == one {
                b.constrain(value, value, value);
                return;
            }
            let value_plain = b.value_of(value);
            let square_hint = b.mul(value_plain, value_plain);
            let square = b.write_witness(square_hint);
            b.constrain(value, value, square);
            let diff = b.sub(square, value);
            let zero = b.field_const(Field::ZERO);
            b.constrain(flag, diff, zero);
            return;
        }
        if bits == 8 {
            b.lookup_rngchk_8(value, flag);
            return;
        }

        let full_bytes = bits / 8;
        let leftover_bits = bits % 8;
        let total_chunks = full_bytes + if leftover_bits > 0 { 1 } else { 0 };

        let pure_value = b.value_of(value);
        let bytes_val = b.fresh_value();
        b.push(OpCode::ToRadix {
            result: bytes_val,
            value: pure_value,
            radix: Radix::Bytes,
            endianness: Endianness::Big,
            count: total_chunks,
        });
        let two_to_8 = b.field_const(Field::from(256));

        let mut partial = b.field_const(Field::ZERO);
        let mut top_chunk: Option<ValueId> = None;
        for i in 0..total_chunks - 1 {
            let idx = b.u_const(32, i as u128);
            let byte = b.array_get(bytes_val, idx);
            let byte_field = b.cast_to_field(byte);
            let byte_wit = b.write_witness(byte_field);
            b.lookup_rngchk_8(byte_wit, flag);
            if i == 0 {
                top_chunk = Some(byte_wit);
            }
            let shift_prev = b.mul(partial, two_to_8);
            partial = b.add(shift_prev, byte_wit);
        }

        let partial_shifted = b.mul(partial, two_to_8);
        let lsb = b.sub(value, partial_shifted);
        b.lookup_rngchk_8(lsb, flag);
        if total_chunks == 1 {
            top_chunk = Some(lsb);
        }

        if leftover_bits > 0 {
            let top = top_chunk.expect("top_chunk set when total_chunks >= 1");
            let bound = b.field_const(Field::from((1u128 << leftover_bits) - 1));
            let gap = b.sub(bound, top);
            b.lookup_rngchk_8(gap, flag);
        }
    }

    fn spill_wide_spread_lookup(
        &self,
        b: &mut HLInstrBuilder<'_>,
        function_type_info: &FunctionTypeInfo,
        key: ValueId,
        expected_spread: ValueId,
        flag: ValueId,
        bits: u8,
    ) {
        assert!(
            bits <= 128,
            "wide Spread lookup spilling currently supports widths up to 128 bits, got {bits}"
        );

        let key_type = function_type_info.get_value_type(key);
        let key_is_witness = key_type.is_witness_of();
        let key_inner = key_type.strip_witness();
        let mut pure_key = if key_is_witness { b.value_of(key) } else { key };
        if key_inner.is_field() {
            pure_key = b.cast_to(CastTarget::U(bits as usize), pure_key);
        }

        let flag_field = b.ensure_field(flag, function_type_info.get_value_type(flag));
        let zero = b.field_const(Field::ZERO);
        let mut reconstructed_key = zero;
        let mut reconstructed_spread = zero;
        let mut offset = 0usize;
        let bits = bits as usize;

        while offset < bits {
            let chunk_bits = (bits - offset).min(8);
            let chunk = extract_low_chunk(b, pure_key, bits, offset, chunk_bits);
            let chunk_field = b.cast_to_field(chunk);
            let chunk_key = if key_is_witness {
                b.write_witness(chunk_field)
            } else {
                chunk_field
            };
            let is_last = offset + chunk_bits == bits;

            let chunk_spread = if is_last {
                let remaining_spread = b.sub(expected_spread, reconstructed_spread);
                let inv_spread_shift = two_pow(offset * 2)
                    .inverse()
                    .expect("non-zero power of two must be invertible");
                let inv_spread_shift = b.field_const(inv_spread_shift);
                b.mul(remaining_spread, inv_spread_shift)
            } else if key_is_witness {
                let spread_hint = b.spread(chunk, chunk_bits as u8);
                let spread_hint_field = b.cast_to_field(spread_hint);
                b.write_witness(spread_hint_field)
            } else {
                let spread = b.spread(chunk, chunk_bits as u8);
                b.cast_to_field(spread)
            };

            b.lookup_spread(chunk_bits as u8, chunk_key, chunk_spread, flag_field);

            let value_shift = b.field_const(two_pow(offset));
            let shifted_key = b.mul(chunk_key, value_shift);
            reconstructed_key = b.add(reconstructed_key, shifted_key);

            if !is_last {
                let spread_shift = b.field_const(two_pow(offset * 2));
                let shifted_spread = b.mul(chunk_spread, spread_shift);
                reconstructed_spread = b.add(reconstructed_spread, shifted_spread);
            }

            offset += chunk_bits;
        }

        let key_field = if key_inner.is_field() {
            key
        } else {
            b.cast_to_field(key)
        };
        let key_diff = b.sub(reconstructed_key, key_field);
        b.constrain(key_diff, flag_field, zero);
    }
}

fn extract_low_chunk(
    b: &mut HLInstrBuilder<'_>,
    value: ValueId,
    value_bits: usize,
    offset: usize,
    chunk_bits: usize,
) -> ValueId {
    let shifted = if offset == 0 {
        value
    } else {
        let divisor = b.u_const(value_bits, two_pow_u128(offset));
        b.div(value, divisor)
    };
    let modulus = b.u_const(value_bits, two_pow_u128(chunk_bits));
    let chunk = b.modulo(shifted, modulus);
    b.cast_to(CastTarget::U(chunk_bits), chunk)
}

fn two_pow(exponent: usize) -> Field {
    Field::from(2).pow([exponent as u64])
}

fn two_pow_u128(exponent: usize) -> u128 {
    assert!(
        exponent < 128,
        "u128 constant shift out of range for exponent {exponent}"
    );
    1u128 << exponent
}
