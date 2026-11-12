use mavros_artifacts::FieldConfig;
use mavros_int_semantics::int_bits::{HOST_LIMB_BITS, HOST_WORD_BITS};

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        analysis::{
            flow_analysis::FlowAnalysis,
            lookup_sizing::{Chunk, LookupSizing, TableKind},
            types::{FunctionTypeInfo, Types},
        },
        pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
        passes::shared::{
            MIN_LOOPED_RUN,
            limbs::{narrow_int_bits, widest_injective_int_bits},
        },
        ssa::{
            FunctionId, SourceLocation, ValueId,
            hlssa::{
                CastTarget, Constant, HLSSA, HLSSAConstantsSnapshot, LookupTarget, OpCode, Type,
                builder::{HLBlockEmitter, HLEmitter, HLSSABuilder},
            },
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
fn flag_is_const_one(consts: &HLSSAConstantsSnapshot, flag: ValueId, field: FieldConfig) -> bool {
    match consts.get(&flag).map(|c| &**c) {
        Some(Constant::Field(f)) => *f == field.one(),
        Some(Constant::Int(v)) => v.is_one(),
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
        field: FieldConfig,
    ) -> Option<HelperKey> {
        // Matched in two levels, on the opcode and then _exhaustively_ on the target, so that a
        // target added later has to state whether it spills. Folding the non-spilling targets into
        // the opcode-level catch-all would read the same and check nothing.
        let OpCode::Lookup { target, args, flag } = instr else {
            return None;
        };

        match target {
            LookupTarget::DynRangecheck(_) => ice_unreachable!(
                "DynRangecheck is lowered to a static 8-bit rangecheck before spilling"
            ),
            LookupTarget::Rangecheck(bits) => {
                let width = *bits;
                if width == 1 {
                    return None; // 1-bit rangechecks are lowered inline (algebraic), no helper.
                }
                let uncond = flag_is_const_one(consts, *flag, field);
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
            LookupTarget::Spread(bits) => {
                let uncond = flag_is_const_one(consts, *flag, field);
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

            // Neither of these is ever spilled. A `Pow2` table is sized by the shifted operand's
            // width rather than chosen by `LookupSizing`, and an `Array` table by the array it is
            // built from, so for both there is no decomposition to plan and no chunk to factor into
            // a helper.
            LookupTarget::Pow2(_) | LookupTarget::Array(_) => None,
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
                        e.field_const(e.field().one())
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
                        e.field_const(e.field().one())
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
        // Two levels, exhaustive on the target, for the reason [`Self::helper_key_for`] gives.
        let OpCode::Lookup { target, args, flag } = instr else {
            return false;
        };

        match target {
            LookupTarget::DynRangecheck(_) => ice_unreachable!(
                "DynRangecheck is lowered to a static 8-bit rangecheck before spilling"
            ),
            LookupTarget::Rangecheck(bits) if *bits == 1 => {
                let is_witness = types.get_value_type(args[0]).is_witness_of();
                self.spill_rangecheck_inner(e, sizing, args[0], 1, *flag, is_witness);
                true
            }
            LookupTarget::Rangecheck(_) => {
                match self.helper_key_for(instr, types, sizing, consts, e.field()) {
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
                }
            }
            LookupTarget::Spread(_) => {
                match self.helper_key_for(instr, types, sizing, consts, e.field()) {
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
                }
            }

            // Never spilled, so never rewritten here; see [`Self::helper_key_for`].
            LookupTarget::Pow2(_) | LookupTarget::Array(_) => false,
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
                    if let Some(key) = self.helper_key_for(instr, fti, sizing, &consts, ssa.field())
                    {
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

        let injective_bits = widest_injective_int_bits(b.field());
        assert!(
            bits <= injective_bits,
            "a range check to {bits} bits holds for every element of a field that carries \
             int{injective_bits} injectively, so there is nothing to decompose"
        );

        // `gated` mirrors the per-bit free/witnessed split in `spill_one_bit_rangecheck`: an
        // unconditional lookup (flag baked to the constant 1) bit-bounds for free, so its plan may
        // use width-1 chunks the cost model also priced for free.
        let gated = flag != b.field_const(b.field().one());
        let plan = sizing.decompose_rangecheck(bits as u8, gated);

        let pure = if is_witness { b.value_of(value) } else { value };

        // Chunks are extracted from an unsigned integer wide enough to hold the value. The two
        // narrow lanes keep their widths (a 96-bit check extracts through a `u128`, not a 96-bit
        // integer) because the bytecode lanes are built around those widths, and a check within
        // them keeps that lowering. Past the host word there is no such lane, so the width is the
        // value's width rounded up to whole limbs.
        let extract_bits = if bits <= HOST_LIMB_BITS {
            HOST_LIMB_BITS
        } else if bits <= HOST_WORD_BITS {
            HOST_WORD_BITS
        } else {
            bits.next_multiple_of(HOST_LIMB_BITS)
        };
        let pure_u = b.cast_to(CastTarget::Int(extract_bits), pure);

        let offsets = cumulative_offsets(&plan);

        // Extract every chunk above the lowest and accumulate `rest`; the lowest chunk (offset 0)
        // is derived below as `value - rest`, a linear combination needing no reconstruction
        // constraint. A run of at least `MIN_LOOPED_RUN` equal chunks is checked in a counted loop
        // after its first chunk; the loop emits the witnesses and constraints of the unrolled run,
        // in the same order.
        let mut rest = b.field_const(b.field().zero());
        let mut i = 1;
        while i < plan.len() {
            let chunk = plan[i];
            let run = plan[i..].iter().take_while(|c| **c == chunk).count();
            let unrolled = if run >= MIN_LOOPED_RUN { 1 } else { run };
            for &offset in &offsets[i..i + unrolled] {
                let key = extract_low_chunk(b, pure_u, extract_bits, offset, chunk.width as usize);
                let place = b.field_const(b.field().two_pow(offset));
                rest =
                    self.accumulate_rangecheck_chunk(b, chunk, key, place, rest, flag, is_witness);
            }
            if run > unrolled {
                rest = self.spill_rangecheck_run(
                    b,
                    pure_u,
                    extract_bits,
                    chunk,
                    offsets[i + 1],
                    run - 1,
                    rest,
                    flag,
                    is_witness,
                );
            }
            i += run;
        }

        let low_chunk = plan[0];
        let low = b.usub(value, rest);
        self.emit_rangecheck_chunk(b, low_chunk, low, flag, is_witness);
    }

    /// Check the extracted chunk `raw_u` and return `rest` with the chunk added at `place`, its
    /// place value.
    #[allow(clippy::too_many_arguments)]
    fn accumulate_rangecheck_chunk(
        &self,
        b: &mut HLBlockEmitter<'_>,
        chunk: Chunk,
        raw_u: ValueId,
        place: ValueId,
        rest: ValueId,
        flag: ValueId,
        is_witness: bool,
    ) -> ValueId {
        let key = self.witnessed_chunk_key(b, raw_u, is_witness);
        self.emit_rangecheck_chunk(b, chunk, key, flag, is_witness);
        let shifted = b.umul(key, place);
        b.uadd(rest, shifted)
    }

    /// The chunk `raw_u` as the key its lookup reads: a field element, written to a witness column
    /// when the surrounding values are witnesses.
    fn witnessed_chunk_key(
        &self,
        b: &mut HLBlockEmitter<'_>,
        raw_u: ValueId,
        is_witness: bool,
    ) -> ValueId {
        let key_field = b.cast_to_field(raw_u);
        if is_witness {
            b.write_witness(key_field)
        } else {
            key_field
        }
    }

    /// Check `count` further chunks equal to `chunk`, the first at `offset`, in a counted loop, and
    /// return `rest` with each added at its place value. `rest` already holds a checked chunk, so
    /// it has the type every iteration leaves it with.
    #[allow(clippy::too_many_arguments)]
    fn spill_rangecheck_run(
        &self,
        b: &mut HLBlockEmitter<'_>,
        pure_u: ValueId,
        extract_bits: usize,
        chunk: Chunk,
        offset: usize,
        count: usize,
        rest: ValueId,
        flag: ValueId,
        is_witness: bool,
    ) -> ValueId {
        let width = chunk.width as usize;
        let divisor = b.two_pow_const(extract_bits, offset);
        let remaining = b.udiv(pure_u, divisor);
        let chunk_modulus = b.two_pow_const(extract_bits, width);
        let place = b.field_const(b.field().two_pow(offset));
        let step = b.field_const(b.field().two_pow(width));
        let rest_type = if is_witness {
            Type::witness_of(Type::field())
        } else {
            Type::field()
        };
        let results = b.build_counted_loop(
            count,
            vec![
                (remaining, Type::int(extract_bits)),
                (place, Type::field()),
                (rest, rest_type),
            ],
            |b, _, state| {
                let (remaining, place, rest) = (state[0], state[1], state[2]);
                let raw = b.urem(remaining, chunk_modulus);
                let raw_u = b.cast_to(CastTarget::Int(width), raw);
                let rest = self
                    .accumulate_rangecheck_chunk(b, chunk, raw_u, place, rest, flag, is_witness);
                let place = b.umul(place, step);
                let remaining = b.udiv(remaining, chunk_modulus);
                vec![remaining, place, rest]
            },
        );
        results[2]
    }

    /// 1-bit rangecheck, lowered algebraically as `b·(b-1) = 0` rather than a table lookup.
    fn spill_one_bit_rangecheck(&self, b: &mut HLBlockEmitter<'_>, value: ValueId, flag: ValueId) {
        let one = b.field_const(b.field().one());
        if flag == one {
            b.constrain(value, value, value);
            return;
        }
        let value_plain = b.value_of(value);
        let square_hint = b.umul(value_plain, value_plain);
        let square = b.write_witness(square_hint);
        b.constrain(value, value, square);
        let diff = b.usub(square, value);
        let zero = b.field_const(b.field().zero());
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
        let narrow_bits = narrow_int_bits(b.field());
        assert!(
            bits as usize <= narrow_bits,
            "spread spilling supports widths up to {narrow_bits} bits, got {bits}"
        );

        // `gated` mirrors `spill_one_bit_rangecheck`: an unconditional spread (flag baked to 1)
        // bit-bounds for free, so its plan may use the free width-1 chunks (see the rangecheck path).
        let gated = flag_field != b.field_const(b.field().one());
        let mut plan = sizing.decompose_spread(bits, gated);

        let zero = b.field_const(b.field().zero());

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
                let diff = b.usub(expected_spread, key_field);
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
        // rangecheck path for the `<= 64 or == 128` constraint on u-constants, and for why both
        // widths are the backend's rather than the narrow threshold).
        let extract_bits = if (bits as usize) <= HOST_LIMB_BITS {
            HOST_LIMB_BITS
        } else {
            HOST_WORD_BITS
        };
        let pure = if key_is_witness { b.value_of(key) } else { key };
        let pure_u = b.cast_to(CastTarget::Int(extract_bits), pure);
        let key_field = if key_inner_is_field {
            key
        } else {
            b.cast_to_field(key)
        };

        // Rotate a lookup-backed chunk (width >= 2 — a full table chunk or a partial) to the front
        // so it becomes the derived chunk 0: its own `lookup_spread` then closes _both_ the key and
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
            let key_shift = b.field_const(b.field().two_pow(offset));
            let shifted_key = b.umul(chunk_key, key_shift);
            rec_key = b.uadd(rec_key, shifted_key);
            let spread_shift = b.field_const(b.field().two_pow(offset * 2));
            let shifted_spread = b.umul(chunk_spread, spread_shift);
            rec_spread = b.uadd(rec_spread, shifted_spread);
        }

        // Derive chunk 0 from what's left and pin it. Its `lookup_spread` closes the spread sum, so
        // no extra constraint — unless it's a bit (an all-bit plan), where there is no lookup to
        // close the sum and we tie it with the one unavoidable equality.
        let chunk0 = plan[0];
        let chunk0_key = b.usub(key_field, rec_key);
        let chunk0_spread = b.usub(expected_spread, rec_spread);
        if chunk0.width == 1 {
            self.spill_one_bit_rangecheck(b, chunk0_key, flag_field);
            let diff = b.usub(chunk0_key, chunk0_spread);
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
        let bound = b.field_const(b.field().constant((1u128 << chunk.width) - 1));
        b.usub(bound, key)
    }

    /// Emit the "2-larger" complement lookup for a _partial spread_ chunk. The complement only needs
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
        let key_u = b.cast_to(CastTarget::Int(table_size as usize), pure);
        let spread = b.spread(key_u, usize::from(table_size));
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
        let divisor = b.two_pow_const(value_bits, offset);
        b.udiv(value, divisor)
    };
    let modulus = b.two_pow_const(value_bits, chunk_bits);
    let chunk = b.urem(shifted, modulus);
    b.cast_to(CastTarget::Int(chunk_bits), chunk)
}

#[cfg(test)]
mod tests {
    use mavros_artifacts::{Field, Goldilocks, InputValueOrdered, MavrosField};

    use super::*;
    use crate::{
        compiler::ssa::{
            Located, SourceLocation, Terminator,
            hlssa::{BinaryArithOpKind, CmpKind, Type, builder::HLSSABuilder},
        },
        driver::Driver,
        forgery::accepts_the_forgery,
    };

    /// Each chunk the loop of a spilled range check reads is held to its width. Moving `2^w` into
    /// one looped chunk from the chunk above it keeps the value the chunks spell, so only that
    /// chunk's lookup refuses it. That holds on both fields, and under a witnessed branch, whose
    /// guard gates every lookup.
    fn check_looped_chunks<F: MavrosField>(bits: usize, guarded: bool) {
        let field = F::CONFIG;
        let mut ssa = HLSSA::with_main_in(field, "main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        HLSSABuilder::new(&mut ssa).modify_function(main_id, |b| {
            let entry = b.function.get_entry_id();
            if !guarded {
                let mut e = b.block(entry).with_source_location(SourceLocation::test());
                let x = e.add_parameter(Type::field());
                e.rangecheck(x, bits);
                e.terminate_return(vec![]);
                return;
            }
            let (taken, after) = (b.function.add_block(), b.function.add_block());
            let x = {
                let mut e = b.block(entry).with_source_location(SourceLocation::test());
                let x = e.add_parameter(Type::field());
                let condition = e.add_parameter(Type::bool());
                e.terminate_jmp_if(condition, taken, after);
                x
            };
            {
                let mut e = b.block(taken).with_source_location(SourceLocation::test());
                e.rangecheck(x, bits);
                e.terminate_jmp(after, vec![]);
            }
            b.block(after)
                .with_source_location(SourceLocation::test())
                .terminate_return(vec![]);
        });

        // Whether `x` is accepted with the chunks the loop checks on its second and third
        // iterations, which lie `start + w` and `start + 2w` bits up, written as `forged`.
        let accepts = |x: &dyn Fn(usize, usize) -> crate::compiler::Field,
                       forged: &dyn Fn(usize) -> [u64; 2]| {
            let dumps = tempfile::TempDir::new().unwrap();
            let mut driver = Driver::from_ssa(ssa.clone(), dumps.path().to_path_buf(), false);
            driver.make_struct_access_static().unwrap();
            driver.monomorphize().unwrap();
            driver.spill_witness().unwrap();
            let (start, width) = forge_loop_chunks(driver.witness_spilled_ssa_mut(), bits, None);
            let mut inputs = vec![InputValueOrdered::Field(F::from_element(x(start, width)))];
            if guarded {
                inputs.push(InputValueOrdered::Field(F::ONE));
            }
            let forged = forged(width);
            accepts_the_forgery::<F>(ssa.clone(), &inputs, |ssa| {
                forge_loop_chunks(ssa, bits, Some(forged));
            })
        };
        let at = format!("{:?} {bits} bits guarded {guarded}", F::ID);
        // The chunk `start + 2w` bits up is 1, every other chunk 0.
        let value = |start: usize, width: usize| field.two_pow(start + 2 * width);
        assert!(
            accepts(&value, &|_| [0, 1]),
            "{at}: the honest chunks, written as elements, are refused"
        );
        assert!(
            !accepts(&value, &|width| [1 << width, 0]),
            "{at}: a looped chunk of 2^w is accepted"
        );
    }

    #[test]
    fn a_looped_chunk_of_a_spilled_range_check_is_held_to_its_width() {
        for guarded in [false, true] {
            check_looped_chunks::<Field>(128, guarded);
            check_looped_chunks::<Goldilocks>(60, guarded);
        }
    }

    /// The offset of the first chunk the loop of the helper of a `bits`-wide range check checks and
    /// the loop's chunk width, after writing the chunks of its second and third iterations as
    /// `forged`, if given.
    fn forge_loop_chunks(ssa: &mut HLSSA, bits: usize, forged: Option<[u64; 2]>) -> (usize, usize) {
        let field = ssa.field();
        let forged =
            forged.map(|values| values.map(|v| ssa.add_const(Constant::Field(field.constant(v)))));
        let iterations = [1u128, 2].map(|k| ssa.add_const(Constant::int(32, k)));
        let fresh = ssa.fresh_values(4);
        let consts = ssa.const_snapshot();
        let place_of = |value: ValueId| {
            let Some(Constant::Field(place)) = consts.get(&value).map(|c| c.as_ref()) else {
                panic!("the loop starts at a constant place");
            };
            (0..field.field_bit_size() as usize)
                .find(|&bits| field.two_pow(bits) == *place)
                .expect("a place is a power of two")
        };
        let width_of = |value: ValueId| {
            let Some(Constant::Int(modulus)) = consts.get(&value).map(|c| c.as_ref()) else {
                panic!("the chunk modulus is a constant");
            };
            let modulus = usize::try_from(modulus).expect("a chunk modulus fits a host word");
            assert!(
                modulus.is_power_of_two(),
                "a chunk modulus is a power of two"
            );
            modulus.trailing_zeros() as usize
        };

        let mut found = None;
        for (_, function) in ssa.iter_functions_mut() {
            if !function
                .get_name()
                .starts_with(&format!("__rangecheck_spill_{bits}_"))
            {
                continue;
            }
            // The loop header is the block whose parameters the entry's jump starts at
            // `(0, remaining, place, rest)`; its first parameter counts the iterations.
            let header_and_body =
                function
                    .get_blocks()
                    .find_map(|(_, block)| match block.get_terminator() {
                        Some(Terminator::JmpIf(_, body, _)) => {
                            Some((block.get_parameter_values().next().copied()?, *body))
                        }
                        _ => None,
                    });
            let (index, body) = header_and_body.expect("the helper loops over its chunks");
            let start = function
                .get_blocks()
                .find_map(|(_, block)| match block.get_terminator() {
                    Some(Terminator::Jmp(_, args))
                        if args.len() == 4 && consts.contains_key(&args[2]) =>
                    {
                        Some(place_of(args[2]))
                    }
                    _ => None,
                });
            let block = function.get_block_mut(body);
            let width = block
                .get_instructions()
                .find_map(|op| match op {
                    OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::URem,
                        rhs,
                        ..
                    } => Some(width_of(*rhs)),
                    _ => None,
                })
                .expect("the loop cuts each chunk off with a remainder");
            found = Some((start.expect("the loop starts at a constant place"), width));

            let Some(forged) = forged else { continue };
            let mut rewritten = Vec::new();
            for instruction in block.take_instructions() {
                let location = instruction.location().clone();
                let OpCode::WriteWitness {
                    result,
                    value,
                    pinned,
                } = *instruction
                else {
                    rewritten.push(instruction);
                    continue;
                };
                let [first_hit, first_pick, second_hit, second_pick] =
                    [fresh[0], fresh[1], fresh[2], fresh[3]];
                for op in [
                    OpCode::Cmp {
                        kind: CmpKind::Eq,
                        result: first_hit,
                        lhs: index,
                        rhs: iterations[0],
                    },
                    OpCode::Select {
                        result: first_pick,
                        cond: first_hit,
                        if_t: forged[0],
                        if_f: value,
                    },
                    OpCode::Cmp {
                        kind: CmpKind::Eq,
                        result: second_hit,
                        lhs: index,
                        rhs: iterations[1],
                    },
                    OpCode::Select {
                        result: second_pick,
                        cond: second_hit,
                        if_t: forged[1],
                        if_f: first_pick,
                    },
                    OpCode::WriteWitness {
                        result,
                        value: second_pick,
                        pinned,
                    },
                ] {
                    rewritten.push(Located::new(op, location.clone()));
                }
            }
            block.put_instructions(rewritten);
        }
        found.unwrap_or_else(|| panic!("a {bits}-bit range check is spilled into a helper"))
    }
}
