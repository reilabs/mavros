use ark_ff::{AdditiveGroup, Field as _};

use crate::compiler::{
    Field,
    analysis::{
        lookup_sizing::{Chunk, LookupSizing, TableKind},
        types::FunctionTypeInfo,
    },
    ssa::{
        ValueId,
        hlssa::{
            CastTarget, LookupTarget, MAX_SUPPORTED_UNSIGNED_BITS, OpCode,
            builder::{HLBlockEmitter, HLEmitter},
        },
    },
};

use super::{InstructionLoweringRule, LoweringContext};

/// Lowers rangecheck and spread lookups into chunked lookups against the table sizes chosen by the
/// [`LookupSizing`] analysis. A `w`-bit value is split into chunks; a full chunk of exactly the
/// table width is one lookup, while a residual chunk narrower than the table is bounded with the
/// "2-larger" trick (two lookups). Rangechecks may be served by a spread table's key column when
/// the optimizer found that cheaper.
pub struct LowerLookupSpillingOps {}

impl InstructionLoweringRule for LowerLookupSpillingOps {
    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        let sizing = context
            .lookup_sizing()
            .expect("lookup_spilling requires the LookupSizing analysis");
        match instruction {
            OpCode::Lookup {
                target: LookupTarget::Rangecheck(bits),
                args,
                flag,
            } => {
                assert_eq!(args.len(), 1, "Rangecheck lookup must have exactly one key");
                self.spill_rangecheck(b, context.types(), sizing, args[0], *bits as usize, *flag)
            }
            OpCode::Lookup {
                target: LookupTarget::DynRangecheck(_),
                args,
                flag,
            } => {
                // Dynamic-radix digit checks currently only support radix 256 (8-bit digits;
                // asserted in the cost model and R1CS gen), so treat them as static 8-bit
                // rangechecks and route them through the chosen tables.
                assert_eq!(
                    args.len(),
                    1,
                    "DynRangecheck lookup must have exactly one key"
                );
                self.spill_rangecheck(b, context.types(), sizing, args[0], 8, *flag)
            }
            OpCode::Lookup {
                target: LookupTarget::Spread(bits),
                args,
                flag,
            } => {
                assert_eq!(
                    args.len(),
                    2,
                    "Spread lookup must have exactly one key and one result"
                );
                self.spill_spread(b, context.types(), sizing, args[0], args[1], *flag, *bits)
            }
            _ => false,
        }
    }
}

impl LowerLookupSpillingOps {
    pub fn new() -> Self {
        Self {}
    }

    /// Rangecheck `value in [0, 2^bits)`. Returns `false` (pass through unchanged) when the value
    /// is served directly by a single same-width rangecheck table, otherwise emits the chunked
    /// lookups and returns `true`.
    fn spill_rangecheck(
        &self,
        b: &mut HLBlockEmitter<'_>,
        types: &FunctionTypeInfo,
        sizing: &LookupSizing,
        value: ValueId,
        bits: usize,
        flag: ValueId,
    ) -> bool {
        assert!(bits >= 1, "rangecheck width must be at least 1 bit");

        if bits == 1 {
            self.spill_one_bit_rangecheck(b, value, flag);
            return true;
        }

        assert!(
            bits <= MAX_SUPPORTED_UNSIGNED_BITS,
            "rangecheck spilling supports widths up to {MAX_SUPPORTED_UNSIGNED_BITS} bits, got {bits}"
        );

        let plan = sizing.decompose_rangecheck(bits as u8);
        if is_direct_passthrough(&plan, bits as u8, TableKind::Range) {
            return false;
        }

        let value_ty = types.get_value_type(value);
        let is_witness = value_ty.is_witness_of();
        let pure = if is_witness { b.value_of(value) } else { value };
        // Chunk extraction works on an unsigned integer; widen to the smallest backend-supported
        // width that holds the value (bytecode only materializes u-constants of width <= 64 or
        // exactly 128, so e.g. a 96-bit rangecheck must extract through u128).
        let extract_bits = if bits <= 64 {
            64
        } else {
            MAX_SUPPORTED_UNSIGNED_BITS
        };
        let pure_u = b.cast_to(CastTarget::U(extract_bits), pure);

        let offsets = cumulative_offsets(&plan);
        let mut rest = b.field_const(Field::ZERO);

        // Extract every chunk above the lowest as a witness and accumulate `rest`. The lowest
        // chunk (offset 0) is derived below as `value - rest`, which is a linear combination and
        // needs no reconstruction constraint.
        for i in 1..plan.len() {
            let chunk = plan[i];
            let raw_u =
                extract_low_chunk(b, pure_u, extract_bits, offsets[i], chunk.width as usize);
            let key_field = b.cast_to_field(raw_u);
            let key = if is_witness {
                b.write_witness(key_field)
            } else {
                key_field
            };
            self.emit_rangecheck_chunk(b, chunk, key, flag, is_witness);

            let shift = b.field_const(two_pow(offsets[i]));
            let shifted = b.mul(key, shift);
            rest = b.add(rest, shifted);
        }

        let low_chunk = plan[0];
        let low = b.sub(value, rest);
        self.emit_rangecheck_chunk(b, low_chunk, low, flag, is_witness);
        true
    }

    /// 1-bit rangecheck, lowered algebraically as `b·(b-1) = 0` rather than a table lookup.
    fn spill_one_bit_rangecheck(&self, b: &mut HLBlockEmitter<'_>, value: ValueId, flag: ValueId) {
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
    }

    /// Emit the lookup(s) for one rangecheck chunk: one lookup for a full chunk, plus the
    /// "2-larger" gap lookup for a partial chunk. The chunk may target a rangecheck table or, when
    /// the optimizer chose to share, a spread table's key column (which needs a `spread(key)`
    /// hint that is otherwise discarded).
    fn emit_rangecheck_chunk(
        &self,
        b: &mut HLBlockEmitter<'_>,
        chunk: Chunk,
        key: ValueId,
        flag: ValueId,
        is_witness: bool,
    ) {
        match chunk.table {
            TableKind::Range => {
                b.lookup_rngchk(LookupTarget::Rangecheck(chunk.table_size), key, flag);
                if chunk.partial {
                    let gap = self.partial_gap(b, chunk, key);
                    b.lookup_rngchk(LookupTarget::Rangecheck(chunk.table_size), gap, flag);
                }
            }
            TableKind::Spread => {
                let spread = self.spread_hint(b, key, chunk.table_size, is_witness);
                b.lookup_spread(chunk.table_size, key, spread, flag);
                if chunk.partial {
                    let gap = self.partial_gap(b, chunk, key);
                    let gap_spread = self.spread_hint(b, gap, chunk.table_size, is_witness);
                    b.lookup_spread(chunk.table_size, gap, gap_spread, flag);
                }
            }
        }
    }

    /// Spread `key in [0, 2^bits)` to `expected_spread`. Returns `false` (pass through unchanged)
    /// when served directly by a single same-width spread table, otherwise emits the chunked
    /// lookups and returns `true`.
    fn spill_spread(
        &self,
        b: &mut HLBlockEmitter<'_>,
        types: &FunctionTypeInfo,
        sizing: &LookupSizing,
        key: ValueId,
        expected_spread: ValueId,
        flag: ValueId,
        bits: u8,
    ) -> bool {
        assert!(
            bits as usize <= MAX_SUPPORTED_UNSIGNED_BITS,
            "spread spilling supports widths up to {MAX_SUPPORTED_UNSIGNED_BITS} bits, got {bits}"
        );

        let plan = sizing.decompose_spread(bits);
        if is_direct_passthrough(&plan, bits, TableKind::Spread) {
            return false;
        }

        let key_ty = types.get_value_type(key);
        let key_is_witness = key_ty.is_witness_of();
        let key_inner_is_field = key_ty.strip_witness().is_field();
        let flag_field = b.ensure_field(flag, types.get_value_type(flag));
        let zero = b.field_const(Field::ZERO);

        // Single chunk: the whole value, looked up directly (full, or partial with a gap). The key
        // is the value itself, so there is no reconstruction to constrain.
        if plan.len() == 1 {
            let chunk = plan[0];
            let key_field = if key_inner_is_field {
                key
            } else {
                b.cast_to_field(key)
            };
            b.lookup_spread(chunk.table_size, key_field, expected_spread, flag_field);
            if chunk.partial {
                let gap = self.partial_gap(b, chunk, key_field);
                let gap_spread = self.spread_hint(b, gap, chunk.table_size, key_is_witness);
                b.lookup_spread(chunk.table_size, gap, gap_spread, flag_field);
            }
            return true;
        }

        let mut pure_key = if key_is_witness { b.value_of(key) } else { key };
        if key_inner_is_field {
            pure_key = b.cast_to(CastTarget::U(bits as usize), pure_key);
        }

        // Order the partial (if any) first so the last chunk — whose spread we derive to keep the
        // reconstruction exact — is always a full chunk.
        let ordered = partial_first(&plan);
        let offsets = cumulative_offsets(&ordered);
        let last = ordered.len() - 1;

        let mut reconstructed_key = zero;
        let mut reconstructed_spread = zero;
        for i in 0..ordered.len() {
            let chunk = ordered[i];
            let offset = offsets[i];
            let chunk_u =
                extract_low_chunk(b, pure_key, bits as usize, offset, chunk.width as usize);
            let chunk_field = b.cast_to_field(chunk_u);
            let chunk_key = if key_is_witness {
                b.write_witness(chunk_field)
            } else {
                chunk_field
            };

            // The last (full) chunk's spread is derived from what's left of `expected_spread`,
            // which makes the spread reconstruction exact without a separate constraint.
            let chunk_spread = if i == last {
                let remaining = b.sub(expected_spread, reconstructed_spread);
                let inv_shift = two_pow(offset * 2)
                    .inverse()
                    .expect("non-zero power of two is invertible");
                let inv_shift = b.field_const(inv_shift);
                b.mul(remaining, inv_shift)
            } else {
                self.spread_hint(b, chunk_key, chunk.table_size, key_is_witness)
            };

            b.lookup_spread(chunk.table_size, chunk_key, chunk_spread, flag_field);
            if chunk.partial {
                let gap = self.partial_gap(b, chunk, chunk_key);
                let gap_spread = self.spread_hint(b, gap, chunk.table_size, key_is_witness);
                b.lookup_spread(chunk.table_size, gap, gap_spread, flag_field);
            }

            let key_shift = b.field_const(two_pow(offset));
            let shifted_key = b.mul(chunk_key, key_shift);
            reconstructed_key = b.add(reconstructed_key, shifted_key);

            if i != last {
                let spread_shift = b.field_const(two_pow(offset * 2));
                let shifted_spread = b.mul(chunk_spread, spread_shift);
                reconstructed_spread = b.add(reconstructed_spread, shifted_spread);
            }
        }

        let key_field = if key_inner_is_field {
            key
        } else {
            b.cast_to_field(key)
        };
        let key_diff = b.sub(reconstructed_key, key_field);
        b.constrain(key_diff, flag_field, zero);
        true
    }

    /// `(2^width - 1) - key`, the complement looked up by the "2-larger" trick to bound `key` to
    /// exactly `width` bits.
    fn partial_gap(&self, b: &mut HLBlockEmitter<'_>, chunk: Chunk, key: ValueId) -> ValueId {
        let bound = b.field_const(Field::from((1u128 << chunk.width) - 1));
        b.sub(bound, key)
    }

    /// Compute `spread(key)` as a hint matching the `table_size`-bit spread table, returned as a
    /// field (wrapped in a witness when the surrounding values are witnesses).
    fn spread_hint(
        &self,
        b: &mut HLBlockEmitter<'_>,
        key: ValueId,
        table_size: u8,
        is_witness: bool,
    ) -> ValueId {
        let pure = if is_witness { b.value_of(key) } else { key };
        let key_u = b.cast_to(CastTarget::U(table_size as usize), pure);
        let spread = b.spread(key_u, table_size);
        let spread_field = b.cast_to_field(spread);
        if is_witness {
            b.write_witness(spread_field)
        } else {
            spread_field
        }
    }
}

/// A plan that is a single full chunk in a same-width table of the expected kind is already a
/// direct lookup; spilling can leave it untouched and let R1CS generation materialize the table.
fn is_direct_passthrough(plan: &[Chunk], bits: u8, kind: TableKind) -> bool {
    plan.len() == 1
        && !plan[0].partial
        && plan[0].width == bits
        && plan[0].table_size == bits
        && plan[0].table == kind
}

/// Bit offset of each chunk, assigned lowest-order first in plan order.
fn cumulative_offsets(plan: &[Chunk]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(plan.len());
    let mut acc = 0;
    for chunk in plan {
        offsets.push(acc);
        acc += chunk.width as usize;
    }
    offsets
}

/// Reorder a plan to put the partial chunk (if any) first, so the last chunk is always full.
fn partial_first(plan: &[Chunk]) -> Vec<Chunk> {
    let mut ordered: Vec<Chunk> = plan.iter().copied().filter(|c| c.partial).collect();
    ordered.extend(plan.iter().copied().filter(|c| !c.partial));
    ordered
}

fn extract_low_chunk(
    b: &mut HLBlockEmitter<'_>,
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
        exponent < MAX_SUPPORTED_UNSIGNED_BITS,
        "u128 constant shift out of range for exponent {exponent}"
    );
    1u128 << exponent
}
