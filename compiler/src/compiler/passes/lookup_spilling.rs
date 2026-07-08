use ark_ff::{AdditiveGroup, Field as _};

use crate::collections::{HashMap, HashSet};
use crate::compiler::{
    Field,
    analysis::{
        flow_analysis::FlowAnalysis,
        lookup_sizing::{Chunk, LookupSizing, TableKind},
        types::{FunctionTypeInfo, Types},
    },
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::{
        FunctionId, SourceLocation, ValueId,
        hlssa::{
            CastTarget, Constant, HLSSA, HLSSAConstantsSnapshot, LookupTarget,
            MAX_SUPPORTED_UNSIGNED_BITS, OpCode, Type,
            builder::{HLBlockEmitter, HLEmitter, HLSSABuilder},
        },
    },
};

/// Which kind of lookup a decomposition helper serves.
#[derive(Clone, PartialEq, Eq, Hash)]
enum HelperKind {
    Rangecheck,
    Spread,
}

/// Identifies a decomposition-helper function: its kind, the lookup width, and the exact argument
/// types at the call sites (so the generated body matches the per-site output and the call is
/// well-typed). Sites with different arg types — or different conditionality — get distinct helpers.
#[derive(Clone, PartialEq, Eq, Hash)]
struct HelperKey {
    kind: HelperKind,
    width: u8,
    value_type: Type,
    spread_type: Option<Type>,
    flag_type: Type,
    /// Whether every site routed here passes a literal `1` flag. An unconditional helper bakes
    /// `flag = 1` into its body instead of taking it as a parameter, so the gated `flag·b·(b−1)=0`
    /// bit-bound (which needs an intermediate `b²` witness) collapses to the free `b·(b−1)=0`.
    /// Conditional and unconditional sites of the same `(kind, width, types)` get distinct helpers.
    flag_is_const_one: bool,
}

/// Whether `flag` is the literal constant `1` (an unconditional lookup). Such a flag lets the
/// spilled bit-bounds take their free algebraic form; see [`HelperKey::flag_is_const_one`].
fn flag_is_const_one(consts: &HLSSAConstantsSnapshot, flag: ValueId) -> bool {
    match consts.get(&flag).map(|c| &**c) {
        Some(Constant::Field(f)) => *f == Field::ONE,
        Some(Constant::U(_, v)) => *v == 1,
        _ => false,
    }
}

/// Factors multi-chunk rangecheck/spread decompositions into one shared helper function per
/// `(kind, width, arg types)`, called from every site, so the unrolled chunk sequence exists once
/// in the module instead of at each lookup. R1CS generation symbolically re-executes each call, so
/// constraint counts are unchanged; only the bytecode/WASM code size shrinks. Single same-width
/// (direct) lookups and 1-bit algebraic rangechecks are left inline.
pub struct LookupSpilling {}

impl LookupSpilling {
    pub fn new() -> Self {
        Self {}
    }

    /// The helper key for a lookup instruction, or `None` when it needs no helper (direct
    /// single-table lookup, or a 1-bit rangecheck handled inline).
    fn helper_key_for(
        &self,
        instr: &OpCode,
        types: &FunctionTypeInfo,
        sizing: &LookupSizing,
        consts: &HLSSAConstantsSnapshot,
    ) -> Option<HelperKey> {
        match instr {
            OpCode::Lookup {
                target: LookupTarget::DynRangecheck(_),
                ..
            } => unreachable!(
                "DynRangecheck is lowered to a static 8-bit rangecheck before spilling"
            ),
            OpCode::Lookup {
                target: LookupTarget::Rangecheck(bits),
                args,
                flag,
            } => {
                let width = *bits;
                if width == 1 {
                    return None; // 1-bit rangechecks are lowered inline (algebraic), no helper.
                }
                let uncond = flag_is_const_one(consts, *flag);
                let plan = sizing.decompose_rangecheck(width, !uncond);
                if is_direct_passthrough(&plan, width, TableKind::Range) {
                    return None;
                }
                Some(HelperKey {
                    kind: HelperKind::Rangecheck,
                    width,
                    value_type: types.get_value_type(args[0]).clone(),
                    spread_type: None,
                    flag_type: types.get_value_type(*flag).clone(),
                    flag_is_const_one: uncond,
                })
            }
            OpCode::Lookup {
                target: LookupTarget::Spread(bits),
                args,
                flag,
            } => {
                let uncond = flag_is_const_one(consts, *flag);
                let plan = sizing.decompose_spread(*bits, !uncond);
                if is_direct_passthrough(&plan, *bits, TableKind::Spread) {
                    return None;
                }
                Some(HelperKey {
                    kind: HelperKind::Spread,
                    width: *bits,
                    value_type: types.get_value_type(args[0]).clone(),
                    spread_type: Some(types.get_value_type(args[1]).clone()),
                    flag_type: types.get_value_type(*flag).clone(),
                    flag_is_const_one: uncond,
                })
            }
            _ => None,
        }
    }

    /// Build the helper function body for `key`. `index` disambiguates names across helpers that
    /// share a `(kind, width)` but differ in argument types.
    fn build_helper(
        &self,
        sb: &mut HLSSABuilder<'_>,
        sizing: &LookupSizing,
        key: &HelperKey,
        index: usize,
    ) -> FunctionId {
        let name = match key.kind {
            HelperKind::Rangecheck => format!("__rangecheck_spill_{}_{index}", key.width),
            HelperKind::Spread => format!("__spread_spill_{}_{index}", key.width),
        };
        let location = SourceLocation::synthetic(&name);
        let (id, ()) = sb.add_function(name, |fb| {
            let entry = fb.function().get_entry_id();
            let mut e = fb.block(entry).with_source_location(location);
            match key.kind {
                HelperKind::Rangecheck => {
                    let value = e.add_parameter(key.value_type.clone());
                    // An unconditional helper bakes `flag = 1` so the spilled bit-bounds take their
                    // free algebraic form; a conditional helper takes the flag as a parameter.
                    let flag = if key.flag_is_const_one {
                        e.field_const(Field::ONE)
                    } else {
                        e.add_parameter(key.flag_type.clone())
                    };
                    let is_witness = key.value_type.is_witness_of();
                    self.spill_rangecheck_inner(
                        &mut e,
                        sizing,
                        value,
                        key.width as usize,
                        flag,
                        is_witness,
                    );
                }
                HelperKind::Spread => {
                    let spread_key = e.add_parameter(key.value_type.clone());
                    let spread = e.add_parameter(key.spread_type.clone().unwrap());
                    let key_is_witness = key.value_type.is_witness_of();
                    let key_inner_is_field = key.value_type.strip_witness().is_field();
                    let flag_field = if key.flag_is_const_one {
                        e.field_const(Field::ONE)
                    } else {
                        let flag = e.add_parameter(key.flag_type.clone());
                        e.ensure_field(flag, &key.flag_type)
                    };
                    self.spill_spread_inner(
                        &mut e,
                        sizing,
                        spread_key,
                        spread,
                        flag_field,
                        key.width,
                        key_is_witness,
                        key_inner_is_field,
                    );
                }
            }
            e.terminate_return(vec![]);
        });
        id
    }

    /// Replace one lookup with a call to its helper (or lower it inline for the 1-bit case).
    /// Returns `false` to leave the instruction untouched (direct passthrough or non-lookup).
    fn rewrite_lookup(
        &self,
        e: &mut HLBlockEmitter<'_>,
        instr: &OpCode,
        types: &FunctionTypeInfo,
        sizing: &LookupSizing,
        consts: &HLSSAConstantsSnapshot,
        cache: &HashMap<HelperKey, FunctionId>,
    ) -> bool {
        match instr {
            OpCode::Lookup {
                target: LookupTarget::DynRangecheck(_),
                ..
            } => unreachable!(
                "DynRangecheck is lowered to a static 8-bit rangecheck before spilling"
            ),
            OpCode::Lookup {
                target: LookupTarget::Rangecheck(bits),
                args,
                flag,
            } if *bits == 1 => {
                let is_witness = types.get_value_type(args[0]).is_witness_of();
                self.spill_rangecheck_inner(e, sizing, args[0], 1, *flag, is_witness);
                true
            }
            OpCode::Lookup {
                target: LookupTarget::Rangecheck(_),
                args,
                flag,
            } => match self.helper_key_for(instr, types, sizing, consts) {
                Some(key) => {
                    let helper = cache[&key];
                    // Unconditional helpers bake `flag = 1` and take no flag parameter.
                    let call_args = if key.flag_is_const_one {
                        vec![args[0]]
                    } else {
                        vec![args[0], *flag]
                    };
                    e.call(helper, call_args, 0);
                    true
                }
                None => false,
            },
            OpCode::Lookup {
                target: LookupTarget::Spread(_),
                args,
                flag,
            } => match self.helper_key_for(instr, types, sizing, consts) {
                Some(key) => {
                    let helper = cache[&key];
                    let call_args = if key.flag_is_const_one {
                        vec![args[0], args[1]]
                    } else {
                        vec![args[0], args[1], *flag]
                    };
                    e.call(helper, call_args, 0);
                    true
                }
                None => false,
            },
            _ => false,
        }
    }
}

impl Pass for LookupSpilling {
    fn name(&self) -> &'static str {
        "lookup_spilling"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![LookupSizing::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        let sizing = store.get::<LookupSizing>();
        let flow = FlowAnalysis::run(ssa);
        let types = Types::new().run(ssa, &flow);
        let consts = ssa.const_snapshot();

        // Phase 1: collect the distinct helpers needed across the whole module.
        let original_fns: Vec<FunctionId> = ssa.get_function_ids().collect();
        let mut needed: Vec<HelperKey> = vec![];
        let mut seen: HashSet<HelperKey> = HashSet::default();
        for &fid in &original_fns {
            let fti = types.get_function(fid);
            for (_bid, block) in ssa.get_function(fid).get_blocks() {
                for instr in block.get_instructions() {
                    if let Some(key) = self.helper_key_for(instr, fti, sizing, &consts) {
                        if seen.insert(key.clone()) {
                            needed.push(key);
                        }
                    }
                }
            }
        }

        // Phase 2: create the helper functions.
        let mut cache: HashMap<HelperKey, FunctionId> = HashMap::default();
        {
            let mut sb = HLSSABuilder::new(ssa);
            for (index, key) in needed.iter().enumerate() {
                let id = self.build_helper(&mut sb, sizing, key, index);
                cache.insert(key.clone(), id);
            }
        }

        // Phase 3: rewrite every site to a call (or inline the 1-bit case).
        let mut sb = HLSSABuilder::new(ssa);
        for &fid in &original_fns {
            let fti = types.get_function(fid);
            sb.modify_function(fid, |fb| {
                let block_ids: Vec<_> = fb.function.get_blocks().map(|(bid, _)| *bid).collect();
                for block_id in block_ids {
                    let (instructions, terminator) = {
                        let mut block = fb.function.take_block(block_id);
                        let instructions = block.take_instructions();
                        let terminator = block.take_terminator();
                        fb.function.put_block(block_id, block);
                        (instructions, terminator)
                    };
                    // Every rewritten lookup scopes its own location below; emitting outside a
                    // scope is an ICE.
                    let mut e = fb
                        .block(block_id)
                        .with_scoped_source_locations("lookup_spilling");
                    for instr in instructions {
                        let location = instr.location().clone();
                        let rewritten = e.emit_with_location(location, |e| {
                            self.rewrite_lookup(e, &instr, fti, sizing, &consts, &cache)
                        });
                        if !rewritten {
                            e.emit_located(instr);
                        }
                    }
                    if let Some(terminator) = terminator {
                        e.set_terminator(terminator);
                    }
                }
            });
        }
    }
}

/// Chunk-emission helpers shared by [`LookupSpilling::build_helper`] (and the inline 1-bit case in
/// [`LookupSpilling::rewrite_lookup`]). A `w`-bit value is split into chunks against the table sizes
/// chosen by [`LookupSizing`]: a full chunk of exactly the table width is one lookup, a residual
/// chunk narrower than the table is bounded with the "2-larger" trick (two lookups), and a width-1
/// chunk is bounded algebraically. Rangechecks may be served by a spread table's key column.
impl LookupSpilling {
    /// Decompose `value in [0, 2^bits)` into chunked lookups. `is_witness` says whether the
    /// surrounding values are `WitnessOf`-wrapped (so chunk hints must be re-wrapped); the value is
    /// assumed castable to an unsigned integer.
    fn spill_rangecheck_inner(
        &self,
        b: &mut HLBlockEmitter<'_>,
        sizing: &LookupSizing,
        value: ValueId,
        bits: usize,
        flag: ValueId,
        is_witness: bool,
    ) {
        if bits == 1 {
            self.spill_one_bit_rangecheck(b, value, flag);
            return;
        }

        assert!(
            bits <= MAX_SUPPORTED_UNSIGNED_BITS,
            "rangecheck spilling supports widths up to {MAX_SUPPORTED_UNSIGNED_BITS} bits, got {bits}"
        );

        // `gated` mirrors the per-bit free/witnessed split in `spill_one_bit_rangecheck`: an
        // unconditional lookup (flag baked to the constant 1) bit-bounds for free, so its plan may
        // use width-1 chunks the cost model also priced for free.
        let gated = flag != b.field_const(Field::ONE);
        let plan = sizing.decompose_rangecheck(bits as u8, gated);

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

        // Extract every chunk above the lowest and accumulate `rest`; the lowest chunk (offset 0) is
        // derived below as `value - rest`, a linear combination needing no reconstruction
        // constraint. Always unrolled — a counted-loop fast path for uniform runs is future work.
        let mut rest = b.field_const(Field::ZERO);
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
        if chunk.width == 1 {
            // A 1-bit chunk needs no table: bound it algebraically with `key·(key-1) = 0`.
            self.spill_one_bit_rangecheck(b, key, flag);
            return;
        }
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

    /// Decompose a spread of `key in [0, 2^bits)` to `expected_spread` into chunked lookups.
    /// `flag_field` must already be a `Field`.
    #[allow(clippy::too_many_arguments)]
    fn spill_spread_inner(
        &self,
        b: &mut HLBlockEmitter<'_>,
        sizing: &LookupSizing,
        key: ValueId,
        expected_spread: ValueId,
        flag_field: ValueId,
        bits: u8,
        key_is_witness: bool,
        key_inner_is_field: bool,
    ) {
        assert!(
            bits as usize <= MAX_SUPPORTED_UNSIGNED_BITS,
            "spread spilling supports widths up to {MAX_SUPPORTED_UNSIGNED_BITS} bits, got {bits}"
        );

        // `gated` mirrors `spill_one_bit_rangecheck`: an unconditional spread (flag baked to 1)
        // bit-bounds for free, so its plan may use the free width-1 chunks (see the rangecheck path).
        let gated = flag_field != b.field_const(Field::ONE);
        let mut plan = sizing.decompose_spread(bits, gated);

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
            if chunk.width == 1 {
                // A 1-bit spread is the identity: bound the bit and pin `expected_spread == key`.
                self.spill_one_bit_rangecheck(b, key_field, flag_field);
                let diff = b.sub(expected_spread, key_field);
                b.constrain(flag_field, diff, zero);
                return;
            }
            b.lookup_spread(chunk.table_size, key_field, expected_spread, flag_field);
            if chunk.partial {
                let gap = self.partial_gap(b, chunk, key_field);
                self.emit_spread_partial_gap(b, sizing, chunk, gap, flag_field, key_is_witness);
            }
            return;
        }

        // Extract chunk to an unsigned integer; widen to a backend-materializable width (see the
        // rangecheck path for the `<= 64 or == 128` constraint on u-constants).
        let extract_bits = if (bits as usize) <= 64 {
            64
        } else {
            MAX_SUPPORTED_UNSIGNED_BITS
        };
        let pure = if key_is_witness { b.value_of(key) } else { key };
        let pure_u = b.cast_to(CastTarget::U(extract_bits), pure);
        let key_field = if key_inner_is_field {
            key
        } else {
            b.cast_to_field(key)
        };

        // Rotate a lookup-backed chunk (width >= 2 — a full table chunk or a partial) to the front
        // so it becomes the derived chunk 0: its own `lookup_spread` then closes *both* the key and
        // spread reconstructions, with no extra constraint. Only an all-bit plan has no such chunk;
        // its front bit's spread sum is closed by the one unavoidable explicit equality below.
        let closing = plan.iter().position(|c| c.width >= 2).unwrap_or(0);
        plan.rotate_left(closing);
        let offsets = cumulative_offsets(&plan);

        // Extract chunks `1..` and accumulate their weighted key and spread sums; chunk 0 is derived
        // below from what's left. Always unrolled — a counted-loop fast path for uniform runs is
        // future work.
        let mut rec_key = zero;
        let mut rec_spread = zero;
        for i in 1..plan.len() {
            let chunk = plan[i];
            let offset = offsets[i];
            let chunk_u = extract_low_chunk(b, pure_u, extract_bits, offset, chunk.width as usize);
            let chunk_field = b.cast_to_field(chunk_u);
            let chunk_key = if key_is_witness {
                b.write_witness(chunk_field)
            } else {
                chunk_field
            };
            // A 1-bit chunk's spread is the identity (`spread(bit) = bit`); a table chunk needs its
            // `spread(key)` hint and lookup.
            let chunk_spread = if chunk.width == 1 {
                self.spill_one_bit_rangecheck(b, chunk_key, flag_field);
                chunk_key
            } else {
                let spread = self.spread_hint(b, chunk_key, chunk.table_size, key_is_witness);
                b.lookup_spread(chunk.table_size, chunk_key, spread, flag_field);
                if chunk.partial {
                    let gap = self.partial_gap(b, chunk, chunk_key);
                    self.emit_spread_partial_gap(b, sizing, chunk, gap, flag_field, key_is_witness);
                }
                spread
            };
            let key_shift = b.field_const(two_pow(offset));
            let shifted_key = b.mul(chunk_key, key_shift);
            rec_key = b.add(rec_key, shifted_key);
            let spread_shift = b.field_const(two_pow(offset * 2));
            let shifted_spread = b.mul(chunk_spread, spread_shift);
            rec_spread = b.add(rec_spread, shifted_spread);
        }

        // Derive chunk 0 from what's left and pin it. Its `lookup_spread` closes the spread sum, so
        // no extra constraint — unless it's a bit (an all-bit plan), where there is no lookup to
        // close the sum and we tie it with the one unavoidable equality.
        let chunk0 = plan[0];
        let chunk0_key = b.sub(key_field, rec_key);
        let chunk0_spread = b.sub(expected_spread, rec_spread);
        if chunk0.width == 1 {
            self.spill_one_bit_rangecheck(b, chunk0_key, flag_field);
            let diff = b.sub(chunk0_key, chunk0_spread);
            b.constrain(flag_field, diff, zero);
        } else {
            b.lookup_spread(chunk0.table_size, chunk0_key, chunk0_spread, flag_field);
            if chunk0.partial {
                let gap = self.partial_gap(b, chunk0, chunk0_key);
                self.emit_spread_partial_gap(b, sizing, chunk0, gap, flag_field, key_is_witness);
            }
        }
    }

    /// `(2^width - 1) - key`, the complement looked up by the "2-larger" trick to bound `key` to
    /// exactly `width` bits.
    fn partial_gap(&self, b: &mut HLBlockEmitter<'_>, chunk: Chunk, key: ValueId) -> ValueId {
        let bound = b.field_const(Field::from((1u128 << chunk.width) - 1));
        b.sub(bound, key)
    }

    /// Emit the "2-larger" complement lookup for a *partial spread* chunk. The complement only needs
    /// to be range-bounded (to rule out field wraparound), not spread, so when a rangecheck table of
    /// size `>= chunk.width` exists it is a single 1-witness rangecheck. Otherwise it falls back to
    /// a spread lookup in the chunk's own table. This mirrors the cost model in `lookup_sizing`
    /// (a spread partial costs `2 + 1` when a range table is available, else `2 + 2`).
    fn emit_spread_partial_gap(
        &self,
        b: &mut HLBlockEmitter<'_>,
        sizing: &LookupSizing,
        chunk: Chunk,
        gap: ValueId,
        flag_field: ValueId,
        key_is_witness: bool,
    ) {
        if let Some(&table) = sizing
            .rangecheck_tables
            .iter()
            .filter(|&&t| t >= chunk.width)
            .min()
        {
            b.lookup_rngchk(LookupTarget::Rangecheck(table), gap, flag_field);
        } else {
            let gap_spread = self.spread_hint(b, gap, chunk.table_size, key_is_witness);
            b.lookup_spread(chunk.table_size, gap, gap_spread, flag_field);
        }
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
