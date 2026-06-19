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
        FunctionId, ValueId,
        hlssa::{
            Blob, CastTarget, Constant, HLSSA, HLSSAConstantsSnapshot, LookupTarget,
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
                target: target @ (LookupTarget::Rangecheck(_) | LookupTarget::DynRangecheck(_)),
                args,
                flag,
            } => {
                // Dynamic-radix digit checks are radix-256, i.e. 8-bit rangechecks.
                let width = match target {
                    LookupTarget::Rangecheck(bits) => *bits,
                    _ => 8,
                };
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
        let (id, ()) = sb.add_function(name, |fb| {
            let entry = fb.function().get_entry_id();
            let mut e = fb.block(entry);
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
                target: LookupTarget::Rangecheck(bits),
                args,
                flag,
            } if *bits == 1 => {
                let is_witness = types.get_value_type(args[0]).is_witness_of();
                self.spill_rangecheck_inner(e, sizing, args[0], 1, *flag, is_witness);
                true
            }
            OpCode::Lookup {
                target: LookupTarget::Rangecheck(_) | LookupTarget::DynRangecheck(_),
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
                    let mut e = fb.block(block_id);
                    for instr in instructions {
                        if !self.rewrite_lookup(&mut e, &instr, fti, sizing, &consts, &cache) {
                            e.emit(instr);
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

        // Extract every chunk above the lowest and accumulate `rest`; the lowest chunk (offset 0)
        // is derived below as `value - rest`, a linear combination that needs no reconstruction
        // constraint. When the chunks are a uniform run of full same-size rangecheck chunks (the
        // common case, e.g. a 32-bit value as four 8-bit chunks), extract them with a counted loop
        // over a blob constant array of shift divisors rather than unrolling, keeping the shared
        // helper body compact. R1CS generation unrolls the loop, so constraints are unchanged.
        let s = plan[0].width;
        let uniform_full = plan.len() >= 3
            && s > 1
            && plan
                .iter()
                .all(|c| matches!(c.table, TableKind::Range) && !c.partial && c.width == s);

        let rest = if uniform_full {
            self.emit_uniform_rangecheck_loop(
                b,
                pure_u,
                extract_bits,
                s,
                plan.len(),
                flag,
                is_witness,
            )
        } else {
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
            rest
        };

        let low_chunk = plan[0];
        let low = b.sub(value, rest);
        self.emit_rangecheck_chunk(b, low_chunk, low, flag, is_witness);
    }

    /// Extract the upper `n_chunks - 1` full `s`-bit rangecheck chunks of `pure_u` with a counted
    /// loop over a blob constant array of shift divisors (`2^(i·s)`), looking up each and
    /// accumulating their weighted sum. Returns `rest`; the caller derives the lowest chunk as
    /// `value - rest`. Used only for uniform decompositions (see `spill_rangecheck_inner`).
    #[allow(clippy::too_many_arguments)]
    fn emit_uniform_rangecheck_loop(
        &self,
        b: &mut HLBlockEmitter<'_>,
        pure_u: ValueId,
        extract_bits: usize,
        s: u8,
        n_chunks: usize,
        flag: ValueId,
        is_witness: bool,
    ) -> ValueId {
        let s_bits = s as usize;

        // Peel chunk 1 (offset `s`) to seed `rest` directly as a witness product, matching the
        // unrolled path which starts from a *constant* zero — seeding the loop accumulator with a
        // fresh `write_witness(0)` instead would add a spurious witness column per call.
        let div1 = b.u_const(extract_bits, two_pow_u128(s_bits));
        let chunk1_u = extract_low_chunk_dyn(b, pure_u, extract_bits, div1, s_bits);
        let chunk1_field = b.cast_to_field(chunk1_u);
        let key1 = if is_witness {
            b.write_witness(chunk1_field)
        } else {
            chunk1_field
        };
        b.lookup_rngchk(LookupTarget::Rangecheck(s), key1, flag);
        let shift1 = b.cast_to_field(div1);
        let rest_seed = b.mul(key1, shift1);

        // Remaining chunks 2..n_chunks: loop over a blob constant array of their shift divisors.
        let divisor_consts: Vec<Constant> = (2..n_chunks)
            .map(|i| Constant::U(extract_bits, two_pow_u128(i * s_bits)))
            .collect();
        if divisor_consts.is_empty() {
            return rest_seed;
        }
        let count = divisor_consts.len();
        let elem_ty = Type::u(extract_bits);
        let blob = b.emit_constant(Constant::Blob(Blob::new(elem_ty.clone(), divisor_consts)));
        let divisors = b.mk_seq_of_blob(elem_ty, blob);
        let rest_ty = if is_witness {
            Type::witness_of(Type::field())
        } else {
            Type::field()
        };
        let results = b.build_counted_loop(count, vec![(rest_seed, rest_ty)], |b, idx, accs| {
            let rest = accs[0];
            let divisor = b.array_get(divisors, idx);
            let chunk_u = extract_low_chunk_dyn(b, pure_u, extract_bits, divisor, s_bits);
            let key_field = b.cast_to_field(chunk_u);
            let key = if is_witness {
                b.write_witness(key_field)
            } else {
                key_field
            };
            b.lookup_rngchk(LookupTarget::Rangecheck(s), key, flag);
            // `rest += key · 2^offset`; the shift is the same divisor, viewed as a field.
            let shift = b.cast_to_field(divisor);
            let shifted = b.mul(key, shift);
            let rest_next = b.add(rest, shifted);
            vec![rest_next]
        });
        results[0]
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
        let plan = sizing.decompose_spread(bits, gated);

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

        let mut pure_key = if key_is_witness { b.value_of(key) } else { key };
        if key_inner_is_field {
            pure_key = b.cast_to(CastTarget::U(bits as usize), pure_key);
        }

        // Order the partial (if any) first so the last chunk — whose spread we derive to keep the
        // reconstruction exact — is always a full chunk.
        let ordered = partial_first(&plan);
        let offsets = cumulative_offsets(&ordered);
        let last = ordered.len() - 1;

        let key_field = if key_inner_is_field {
            key
        } else {
            b.cast_to_field(key)
        };

        let mut reconstructed_key = zero;
        let mut reconstructed_spread = zero;
        for i in 0..ordered.len() {
            let chunk = ordered[i];
            let offset = offsets[i];

            // The last (full) chunk's key and spread are both derived from what's left of `key`
            // and `expected_spread`. These are linear combinations of values already bounded by
            // their own lookups, so the recombination is exact with no separate constraint: the
            // derived chunk's lookup pins `spread(low_key) == low_spread` and `low_key`'s range,
            // and `key`/`expected_spread` equal the chunk sums by construction.
            let (chunk_key, chunk_spread) = if i == last {
                let key_remaining = b.sub(key_field, reconstructed_key);
                let inv_key_shift = two_pow(offset)
                    .inverse()
                    .expect("non-zero power of two is invertible");
                let inv_key_shift = b.field_const(inv_key_shift);
                let chunk_key = b.mul(key_remaining, inv_key_shift);

                let spread_remaining = b.sub(expected_spread, reconstructed_spread);
                let inv_spread_shift = two_pow(offset * 2)
                    .inverse()
                    .expect("non-zero power of two is invertible");
                let inv_spread_shift = b.field_const(inv_spread_shift);
                let chunk_spread = b.mul(spread_remaining, inv_spread_shift);
                (chunk_key, chunk_spread)
            } else {
                let chunk_u =
                    extract_low_chunk(b, pure_key, bits as usize, offset, chunk.width as usize);
                let chunk_field = b.cast_to_field(chunk_u);
                let chunk_key = if key_is_witness {
                    b.write_witness(chunk_field)
                } else {
                    chunk_field
                };
                // A 1-bit chunk's spread is the identity (`spread(bit) = bit`), so reuse the key
                // rather than witnessing a separate `spread(key)` hint.
                let chunk_spread = if chunk.width == 1 {
                    chunk_key
                } else {
                    self.spread_hint(b, chunk_key, chunk.table_size, key_is_witness)
                };
                (chunk_key, chunk_spread)
            };

            if chunk.width == 1 {
                // No table: bound the bit, and (for the derived last chunk, whose key and spread
                // are derived independently) tie `spread == key` so the reconstruction closes.
                self.spill_one_bit_rangecheck(b, chunk_key, flag_field);
                if i == last {
                    let diff = b.sub(chunk_key, chunk_spread);
                    b.constrain(flag_field, diff, zero);
                }
            } else {
                b.lookup_spread(chunk.table_size, chunk_key, chunk_spread, flag_field);
                if chunk.partial {
                    let gap = self.partial_gap(b, chunk, chunk_key);
                    self.emit_spread_partial_gap(b, sizing, chunk, gap, flag_field, key_is_witness);
                }
            }

            if i != last {
                let key_shift = b.field_const(two_pow(offset));
                let shifted_key = b.mul(chunk_key, key_shift);
                reconstructed_key = b.add(reconstructed_key, shifted_key);

                let spread_shift = b.field_const(two_pow(offset * 2));
                let shifted_spread = b.mul(chunk_spread, spread_shift);
                reconstructed_spread = b.add(reconstructed_spread, shifted_spread);
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

/// Like [`extract_low_chunk`] but with a runtime shift `divisor` (= `2^offset`, read from a
/// constant array) instead of a compile-time offset: `(value / divisor) mod 2^chunk_bits`.
fn extract_low_chunk_dyn(
    b: &mut HLBlockEmitter<'_>,
    value: ValueId,
    value_bits: usize,
    divisor: ValueId,
    chunk_bits: usize,
) -> ValueId {
    let shifted = b.div(value, divisor);
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
