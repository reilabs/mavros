#![allow(unused_variables)]

use ark_ff::{AdditiveGroup as _, BigInteger as _};
use mavros_opcode_gen::interpreter;
use serde::{Deserialize, Deserializer, Serialize};

use std::{collections::BTreeMap, fmt::Display, ptr};

use crate::{
    ConstraintsLayout, Field, FlamegraphProfile, FlamegraphStackId, TableKind, WitnessLayout,
    array::{BoxedLayout, BoxedValue, DataType, StructDescriptor},
    int_limbs,
    interpreter::{Frame, Handler},
};

/// The number of u64 limbs making up a field element.
// FIELD-ASSUMPTION: L3-felt-limbs
pub const FELT_LIMBS: usize = 4;

/// A user-facing source position attached to generated VM code.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceLocation {
    pub file: String,
    pub line: u64,
    pub column: u64,
}

impl SourceLocation {
    pub fn new(file: impl Into<String>, line: u64, column: u64) -> Self {
        Self {
            file: file.into(),
            line,
            column,
        }
    }
}

pub(crate) fn relativize_source_path(path: &mut String, root: &std::path::Path) {
    if path.starts_with('<') && path.ends_with('>') {
        return;
    }
    if let Ok(relative) = std::path::Path::new(path.as_str()).strip_prefix(root) {
        *path = relative.to_string_lossy().into_owned();
    }
}

impl Display for SourceLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.file.starts_with('<') && self.file.ends_with('>') {
            write!(f, "{}", self.file)
        } else {
            write!(f, "{}:{}:{}", self.file, self.line, self.column)
        }
    }
}

/// One frame in a VM stack trace, ordered from the trapping frame to the entry point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StackFrame {
    pub function: String,
    pub location: SourceLocation,
}

impl Display for StackFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.function, self.location)
    }
}

/// A compact, run-length encoded map from VM bytecode word offsets to Noir source locations.
///
/// This is serialized separately from the executable bytecode so production programs never pay
/// for source paths and locations. A location applies from its `code_offset` up to the next
/// location in the same function.
pub const DEBUG_INFO_FORMAT_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DebugInfo {
    #[serde(deserialize_with = "deserialize_debug_info_format_version")]
    format_version: u32,
    pub files: Vec<String>,
    pub functions: Vec<DebugFunction>,
}

fn deserialize_debug_info_format_version<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: Deserializer<'de>,
{
    let version = u32::deserialize(deserializer)?;
    if version != DEBUG_INFO_FORMAT_VERSION {
        return Err(serde::de::Error::custom(format!(
            "unsupported VM debug info format version {version}; expected {DEBUG_INFO_FORMAT_VERSION}"
        )));
    }
    Ok(version)
}

impl Default for DebugInfo {
    fn default() -> Self {
        Self {
            format_version: DEBUG_INFO_FORMAT_VERSION,
            files: Vec::new(),
            functions: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DebugFunction {
    pub name: String,
    pub code_offset: usize,
    pub locations: Vec<DebugLocation>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DebugLocation {
    pub code_offset: usize,
    pub file_index: usize,
    pub line: u64,
    pub column: u64,
}

impl DebugInfo {
    pub fn format_version(&self) -> u32 {
        self.format_version
    }

    /// Strip a common root from real source paths while preserving synthetic `<...>` locations.
    pub fn relativize_source_paths(&mut self, root: &std::path::Path) {
        for file in &mut self.files {
            relativize_source_path(file, root);
        }
    }

    /// Resolve a word offset in the serialized program to its function and nearest source
    /// location. Callers can pass an address inside an opcode, not only its first word.
    pub fn stack_frame_at(&self, code_offset: usize) -> Option<StackFrame> {
        let function_index = self.function_index_at(code_offset)?;
        let function = &self.functions[function_index];
        let location_index = function
            .locations
            .partition_point(|location| location.code_offset <= code_offset)
            .checked_sub(1)?;

        let location = &function.locations[location_index];
        Some(StackFrame {
            function: function.name.clone(),
            location: SourceLocation::new(
                self.files.get(location.file_index)?.clone(),
                location.line,
                location.column,
            ),
        })
    }

    fn function_index_at(&self, code_offset: usize) -> Option<usize> {
        self.functions
            .partition_point(|function| function.code_offset <= code_offset)
            .checked_sub(1)
    }

    fn function_name(&self, function_index: usize) -> Option<&str> {
        self.functions
            .get(function_index)
            .map(|function| function.name.as_str())
    }
}

// ---------------------------------------------------------------------------
// Per-opcode profiling (enabled with the `vm-profile` feature).
//
// The counters live in `VM::opcode_profile` rather than in a global static:
// the dispatch loop holds a unique `&mut VM`, so plain `u64` adds suffice.
// Each generated handler reads the cycle counter at entry/exit and accumulates
// `(count, cycles)` into that buffer, indexed by opcode discriminant.
// ---------------------------------------------------------------------------

/// Read the architectural cycle counter. Cheap, monotonic, userspace-readable.
#[cfg(feature = "vm-profile")]
#[inline(always)]
pub fn read_cycles() -> u64 {
    #[cfg(target_arch = "aarch64")]
    {
        let v: u64;
        // Virtual count register; readable from EL0 on Linux/macOS.
        unsafe { core::arch::asm!("mrs {}, cntvct_el0", out(reg) v, options(nomem, nostack)) };
        v
    }
    #[cfg(target_arch = "x86_64")]
    {
        unsafe { core::arch::x86_64::_rdtsc() }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        0
    }
}

/// Frequency of the counter read by [`read_cycles`], in Hz.
#[cfg(all(feature = "vm-profile", target_arch = "aarch64"))]
fn cycle_freq_hz() -> u64 {
    let v: u64;
    unsafe { core::arch::asm!("mrs {}, cntfrq_el0", out(reg) v, options(nomem, nostack)) };
    v
}

#[cfg(feature = "vm-profile")]
impl VM {
    /// Print this VM's per-opcode profiling report to stderr, sorted by cycles.
    pub fn report_opcode_profile(&self, label: &str) {
        let mut rows: Vec<(&str, u64, u64)> = OPCODE_NAMES
            .iter()
            .zip(self.opcode_profile.iter())
            .map(|(name, &(count, cycles))| (*name, count, cycles))
            .filter(|(_, count, _)| *count > 0)
            .collect();

        let total_count: u64 = rows.iter().map(|(_, count, _)| *count).sum();
        let total_cycles: u64 = rows.iter().map(|(_, _, cycles)| *cycles).sum();
        rows.sort_by(|a, b| b.2.cmp(&a.2));

        #[cfg(target_arch = "aarch64")]
        let freq = cycle_freq_hz();
        #[cfg(not(target_arch = "aarch64"))]
        let freq = 0u64;

        eprintln!("\n=== VM opcode profile: {label} ===");
        eprintln!(
            "total opcodes executed: {total_count}   total counter ticks: {total_cycles}{}",
            if freq > 0 {
                format!(
                    "   (~{:.3} ms @ {} Hz)",
                    (total_cycles as f64 / freq as f64) * 1e3,
                    freq
                )
            } else {
                String::new()
            }
        );
        eprintln!(
            "{:<22} {:>14} {:>7} {:>16} {:>7} {:>10}",
            "opcode", "count", "count%", "ticks", "tick%", "ticks/op"
        );
        for (name, count, cycles) in &rows {
            eprintln!(
                "{:<22} {:>14} {:>6.2}% {:>16} {:>6.2}% {:>10.1}",
                name,
                count,
                100.0 * *count as f64 / total_count.max(1) as f64,
                cycles,
                100.0 * *cycles as f64 / total_cycles.max(1) as f64,
                *cycles as f64 / (*count).max(1) as f64,
            );
        }
        eprintln!("=== end profile: {label} ===\n");
    }
}

/// Element storage kind for array lookup opcodes.
/// Encoded as usize for compatibility with the opcode proc macro.
pub const ELEM_WORD: usize = 0;
pub const ELEM_FIELD: usize = 1;
pub const ELEM_WITNESS: usize = 2;
pub const ELEM_U128: usize = 3;

/// Read an array element as a Field and bump out_db accordingly.
#[inline(always)]
unsafe fn lookup_elem_bump_db(ptr: *mut u64, elem_kind: usize, coeff: Field, vm: &mut VM) {
    match elem_kind {
        ELEM_WORD => unsafe {
            let v = Field::from(*(ptr as *const u64));
            *vm.data.as_ad.out_db += coeff * v;
        },
        ELEM_FIELD => unsafe {
            let v = *(ptr as *const Field);
            *vm.data.as_ad.out_db += coeff * v;
        },
        ELEM_U128 => unsafe {
            let v = Field::from((*(ptr as *const Int128)).to_u128());
            *vm.data.as_ad.out_db += coeff * v;
        },
        ELEM_WITNESS => {
            let elem = BoxedValue(unsafe { *(ptr as *const *mut u64) });
            elem.bump_db(coeff, vm);
        }
        _ => unreachable!(),
    }
}

/// Read a pure (non-WitnessOf) array element as a Field value.
#[inline(always)]
unsafe fn read_pure_elem_as_field(ptr: *mut u64, elem_kind: usize) -> Field {
    match elem_kind {
        ELEM_WORD => Field::from(unsafe { *(ptr as *const u64) }),
        ELEM_FIELD => unsafe { *(ptr as *const Field) },
        ELEM_U128 => Field::from(unsafe { (*(ptr as *const Int128)).to_u128() }),
        _ => unreachable!(),
    }
}

#[inline(always)]
fn int_mask(bits: u64) -> u128 {
    if bits >= 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    }
}

/// The mask that holds a `bits`-wide value inside a single `u64` frame cell.
///
/// This is the masked-cell invariant in one place: an opcode whose result can exceed `bits`
/// re-applies it before storing, so the next opcode to read the cell (signed or unsigned) sees a
/// pattern with nothing above bit `bits - 1`.
///
/// One opcode is deliberately not held to that on its own. `cast_field_to_int` stores the whole low
/// limb of the field element, because it has no `bits` to narrow to; the invariant is restored by
/// codegen, which follows it with a `truncate_int` whenever the target is narrower than the cell
/// (`bytecode/mod.rs`, the `Field -> Int` cast arms). At `bits == 64` no truncation is needed and
/// none is emitted.
#[inline(always)]
fn cell_mask(bits: u64) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

/// The amount a `bits`-wide shift actually applies, given a requested amount of `b`.
///
/// Reduced modulo `bits`. It matches the LLVM backend, which reduces by the LLVM type's own width
/// (`llssa_to_llvm.rs::reduce_shift_count`) because LLVM treats a shift at or past the width as
/// poison. It also keeps the count in range for Rust's shift operators, which panic on an
/// over-shift in a debug build and mask to the _host_ width (`b & 63`) in release, a divergence
/// that is wrong for any `bits < 64`.
///
/// It is a **modulo, not a mask**. At a power-of-two width the two coincide, so at every width Noir
/// can name this answers exactly what `b & (bits - 1)` does. At a width that is not a power of two
/// they differ, and the mask is the wrong one of the pair: it stays below `bits`, which is all a
/// backstop needs, but it corrupts amounts that were already _in range_ — at `bits == 7` a mask
/// turns an amount of `1` into `0`, while `hlssa_to_r1cs` applies that same `1` literally. The
/// modulo is what makes the three evaluators agree at every width.
///
/// The power-of-two arm is a fast path, not a second semantics: `bits` arrives as an opcode
/// immediate, so the branch is decided by the instruction rather than by data.
///
/// A well-formed program never reaches the reduction. Both routes to a shift reject an
/// out-of-range amount before the opcode runs: `pure_guards`' shift-amount check for a pure
/// amount, and `witness_bitwise::emit_shift_amount_check` for a witness one. This is the backstop
/// for when they do not.
///
/// `bits == 0` answers `0` rather than dividing by zero. A zero-width cell holds nothing —
/// `cell_mask(0)` is `0` — so every shift of one answers `0` whatever the amount, and this stops
/// the amount reaching the host shift and panicking.
#[inline(always)]
fn shift_amount(b: u64, bits: u64) -> u32 {
    if bits == 0 {
        return 0;
    }
    if bits.is_power_of_two() {
        (b & (bits - 1)) as u32
    } else {
        (b % bits) as u32
    }
}

/// The amount a double-lane shift actually applies, given a requested amount of `b`.
///
/// The reduction is modulo the operand width, as it is in the other two lanes and as
/// `reduce_shift_count` is on the LLVM side.
///
/// At a power-of-two width the mask reaches only the low limb, and reading `b.lo` alone is then
/// exact rather than a discarded check: LLVM masks the whole amount with `bit_width - 1`, and the
/// low bits of a two-limb pattern are the low bits of its low limb. At any other width the whole
/// pattern is needed — the high limb of a 96-bit amount carries bits 64 to 95, and dropping them
/// would answer a different modulus.
#[inline(always)]
fn shift_amount_128(b: Int128, bits: u64) -> u32 {
    if bits == 0 {
        return 0;
    }
    if bits.is_power_of_two() {
        return (b.lo & (bits - 1)) as u32;
    }
    (b.to_u128() % u128::from(bits)) as u32
}

/// `a + b`, wrapping at the operand width, the body of `add_int128`.
///
/// Wrapping for the reason [`cell_add`] gives: the opcode is total because the rejection Noir owes
/// an overflowing addition is emitted ahead of it.
#[inline(always)]
fn double_add(a: Int128, b: Int128, bits: u64) -> Int128 {
    a.wrapping_add(b).truncate(bits)
}

/// `a - b`, wrapping at the operand width. Wrapping as [`double_add`] is.
#[inline(always)]
fn double_sub(a: Int128, b: Int128, bits: u64) -> Int128 {
    a.wrapping_sub(b).truncate(bits)
}

/// `a * b`, wrapping at the operand width. Wrapping as [`double_add`] is.
#[inline(always)]
fn double_mul(a: Int128, b: Int128, bits: u64) -> Int128 {
    a.wrapping_mul(b).truncate(bits)
}

/// `a << b`, wrapping at the operand width, with the amount reduced against that width.
#[inline(always)]
fn double_shl(a: Int128, b: Int128, bits: u64) -> Int128 {
    a.wrapping_shl(shift_amount_128(b, bits)).truncate(bits)
}

/// `a >> b`, zero-filling, with the amount reduced against the operand width.
///
/// Takes `bits` only to reduce the amount: a zero-filling shift of an operand already held to the
/// width cannot leave it, so there is nothing to re-mask. [`double_complement`] below needs its
/// width for the other reason.
#[inline(always)]
fn double_ushr(a: Int128, b: Int128, bits: u64) -> Int128 {
    a.wrapping_shr(shift_amount_128(b, bits))
}

/// `!a`, masked back into the operand width.
#[inline(always)]
fn double_complement(a: Int128, bits: u64) -> Int128 {
    (!a).truncate(bits)
}

/// Read a `bits`-wide cell as two's complement, in the `i64` the host can compute with.
///
/// Shift the pattern up so its sign bit is the host's, then shift back arithmetically. This is the
/// preamble every signed `_int` opcode shares — `sdiv_int`, `srem_int`, `slt_int`, `ashr_int` — and
/// it is what makes them robust to a cell that is dirty above `bits` as well: the up-shift discards
/// those bits before anything reads them.
#[inline(always)]
fn signed_cell(a: u64, bits: u64) -> i64 {
    let shift = 64 - bits;
    ((a << shift) as i64) >> shift
}

/// `a + b`, wrapping at the operand width, the body of `add_int`.
///
/// The wrapping is so that the opcode remains total: Noir rejects an integer addition that
/// overflows its width, and Mavros emits that rejection ahead of the opcode. By the time the opcode
/// runs the program either proved the sum in range or was already rejected, so we only have to be
/// _defined_ here (and match the LLVM backend).
#[inline(always)]
fn cell_add(a: u64, b: u64, bits: u64) -> u64 {
    a.wrapping_add(b) & cell_mask(bits)
}

/// `a - b`, wrapping at the operand width. The body of `sub_int`. Wrapping as [`cell_add`] is.
#[inline(always)]
fn cell_sub(a: u64, b: u64, bits: u64) -> u64 {
    a.wrapping_sub(b) & cell_mask(bits)
}

/// `a * b`, wrapping at the operand width, the body of `mul_int`.
///
/// One reading suffices for all three: `+`, `-` and `*` are ring operations modulo `2^bits`, so the
/// two's-complement answer and the magnitude answer are the same bit pattern. That is why there is
/// no `cell_smul` beside this, where division does need `cell_sdiv`.
#[inline(always)]
fn cell_mul(a: u64, b: u64, bits: u64) -> u64 {
    a.wrapping_mul(b) & cell_mask(bits)
}

/// `!a`, held to the operand width. The body of `not_int`.
#[inline(always)]
fn cell_complement(a: u64, bits: u64) -> u64 {
    !a & cell_mask(bits)
}

/// `a & b`, the body of `and_int`.
///
/// No `bits`, and that is the masked-cell invariant being _used_ rather than an omission: none of
/// the three bitwise helpers can set a bit its operands did not already carry, so an operand held
/// to `bits` yields a result held to `bits`.
#[inline(always)]
fn cell_and(a: u64, b: u64) -> u64 {
    a & b
}

/// `a | b`. The body of `or_int`. Width-free as [`cell_and`] is.
#[inline(always)]
fn cell_or(a: u64, b: u64) -> u64 {
    a | b
}

/// `a ^ b`. The body of `xor_int`. Width-free as [`cell_and`] is.
#[inline(always)]
fn cell_xor(a: u64, b: u64) -> u64 {
    a ^ b
}

/// `a << b`, wrapping at the operand width. The body of `shl_int`.
#[inline(always)]
fn cell_shl(a: u64, b: u64, bits: u64) -> u64 {
    (a << shift_amount(b, bits)) & cell_mask(bits)
}

/// `a >> b`, zero-filling. The body of `ushr_int`.
///
/// No re-mask: zero-filling an already-masked operand cannot leave the width.
#[inline(always)]
fn cell_ushr(a: u64, b: u64, bits: u64) -> u64 {
    a >> shift_amount(b, bits)
}

/// `a >> b`, sign-filling. The body of `ashr_int`.
#[inline(always)]
fn cell_ashr(a: u64, b: u64, bits: u64) -> u64 {
    (signed_cell(a, bits) >> shift_amount(b, bits)) as u64 & cell_mask(bits)
}

/// `a / b` unsigned.
///
/// This is a central place that records the rationale for all four division helpers.
///
/// All four are **total**. A bare Rust `/` aborts the process on a zero divisor, and at
/// `bits == 64` a bare signed `/` aborts again on `i64::MIN / -1`. The VM's contract is that it
/// does not trap during host execution: a failed execution must be reported at the VM's level using
/// `trap` instead.
///
/// The choice of returning 0 on malformed inputs is arbitrary, but matches existing convention in
/// Mavros. It ensures that the operation is total, as the checks occur external to the operation.
#[inline(always)]
fn cell_udiv(a: u64, b: u64) -> u64 {
    if b == 0 { 0 } else { a / b }
}

/// `a % b` read unsigned. Total, as [`cell_udiv`] describes.
#[inline(always)]
fn cell_urem(a: u64, b: u64) -> u64 {
    if b == 0 { 0 } else { a % b }
}

/// `a / b` read as two's complement, truncating toward zero. Total, as [`cell_udiv`] describes.
#[inline(always)]
fn cell_sdiv(a: u64, b: u64, bits: u64) -> u64 {
    let (a, b) = (signed_cell(a, bits), signed_cell(b, bits));
    if b == 0 {
        0
    } else {
        a.wrapping_div(b) as u64 & cell_mask(bits)
    }
}

/// `a % b` read as two's complement; the sign follows the dividend. Total, as [`cell_udiv`]
/// describes.
#[inline(always)]
fn cell_srem(a: u64, b: u64, bits: u64) -> u64 {
    let (a, b) = (signed_cell(a, bits), signed_cell(b, bits));
    if b == 0 {
        0
    } else {
        a.wrapping_rem(b) as u64 & cell_mask(bits)
    }
}

/// A 128-bit frame value: two `u64` cells holding a raw bit pattern, with no reading attached.
///
/// `Eq`/`PartialEq` are derived because bit-pattern equality is the same question under either
/// reading — which is why there is one `eq_int128` opcode and not a pair. The operations that
/// _do_ depend on the reading are the `unsigned_*` methods below.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Int128 {
    pub lo: u64,
    pub hi: u64,
}

const _: () = assert!(
    std::mem::size_of::<Int128>() == 2 * std::mem::size_of::<u64>()
        && std::mem::align_of::<Int128>() == std::mem::align_of::<u64>(),
    "Int128 must have the same cell layout as two u64 words"
);

impl Int128 {
    #[inline(always)]
    pub fn from_u128(value: u128) -> Self {
        Self {
            lo: value as u64,
            hi: (value >> 64) as u64,
        }
    }

    #[inline(always)]
    pub fn to_u128(self) -> u128 {
        (self.lo as u128) | ((self.hi as u128) << 64)
    }

    #[inline(always)]
    pub fn wrapping_add(self, rhs: Self) -> Self {
        Self::from_u128(self.to_u128().wrapping_add(rhs.to_u128()))
    }

    #[inline(always)]
    pub fn wrapping_sub(self, rhs: Self) -> Self {
        Self::from_u128(self.to_u128().wrapping_sub(rhs.to_u128()))
    }

    #[inline(always)]
    pub fn wrapping_mul(self, rhs: Self) -> Self {
        Self::from_u128(self.to_u128().wrapping_mul(rhs.to_u128()))
    }

    #[inline(always)]
    pub fn wrapping_shl(self, rhs: u32) -> Self {
        Self::from_u128(self.to_u128().wrapping_shl(rhs))
    }

    #[inline(always)]
    pub fn wrapping_shr(self, rhs: u32) -> Self {
        Self::from_u128(self.to_u128().wrapping_shr(rhs))
    }

    #[inline(always)]
    pub fn truncate(self, bits: u64) -> Self {
        Self::from_u128(self.to_u128() & int_mask(bits))
    }

    /// Order the pattern as an unsigned 128-bit integer.
    ///
    /// This and the two below are deliberately inherent methods rather than `Ord`/`Div`/`Rem`: they
    /// are the operations whose answer depends on how the bits are read, so each caller has to name
    /// the reading. An operator would let `a < b` mean "unsigned" by default, which is
    /// mistake-prone.
    #[inline(always)]
    pub fn unsigned_lt(self, rhs: Self) -> bool {
        self.to_u128() < rhs.to_u128()
    }

    /// Divide, reading both patterns as unsigned 128-bit integers.
    #[inline(always)]
    pub fn unsigned_div(self, rhs: Self) -> Self {
        let rhs = rhs.to_u128();
        if rhs == 0 {
            Self::default()
        } else {
            Self::from_u128(self.to_u128() / rhs)
        }
    }

    /// Remainder, reading both patterns as unsigned 128-bit integers. Total, as `unsigned_div` is.
    #[inline(always)]
    pub fn unsigned_rem(self, rhs: Self) -> Self {
        let rhs = rhs.to_u128();
        if rhs == 0 {
            Self::default()
        } else {
            Self::from_u128(self.to_u128() % rhs)
        }
    }
}

impl std::ops::BitAnd for Int128 {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self {
            lo: self.lo & rhs.lo,
            hi: self.hi & rhs.hi,
        }
    }
}

impl std::ops::BitOr for Int128 {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self {
            lo: self.lo | rhs.lo,
            hi: self.hi | rhs.hi,
        }
    }
}

impl std::ops::BitXor for Int128 {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self {
            lo: self.lo ^ rhs.lo,
            hi: self.hi ^ rhs.hi,
        }
    }
}

impl std::ops::Not for Int128 {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self {
            lo: !self.lo,
            hi: !self.hi,
        }
    }
}

unsafe fn for_each_array_leaf<F: FnMut(usize, *mut u64)>(
    array: BoxedValue,
    stride: usize,
    mut f: F,
) -> usize {
    unsafe fn go<F: FnMut(usize, *mut u64)>(
        array: BoxedValue,
        stride: usize,
        f: &mut F,
        idx: &mut usize,
    ) {
        let layout = array.layout();
        if layout.is_boxed_array() {
            let size = layout.array_size();
            for i in 0..size {
                let cell_ptr = array.array_idx(i, 1);
                let inner = unsafe { *(cell_ptr as *mut BoxedValue) };
                let inner_layout = inner.layout();
                if inner_layout.is_boxed_array() || inner_layout.is_prim_array() {
                    unsafe { go(inner, stride, f, idx) };
                } else {
                    f(*idx, cell_ptr);
                    *idx += 1;
                }
            }
        } else if layout.is_prim_array() {
            let n_elems = layout.array_size() / stride;
            for i in 0..n_elems {
                f(*idx, array.array_idx(i, stride));
                *idx += 1;
            }
        } else {
            panic!(
                "Unexpected array data type in lookup-table flatten: {:?}",
                layout.data_type()
            );
        }
    }
    let mut idx = 0;
    unsafe { go(array, stride, &mut f, &mut idx) };
    idx
}

#[derive(Clone, Copy)]
pub struct FramePosition(pub usize);

impl FramePosition {
    pub fn offset(&self, offset: isize) -> FramePosition {
        FramePosition(self.0.checked_add_signed(offset).unwrap())
    }
    pub fn return_data_ptr() -> FramePosition {
        FramePosition(0)
    }
    pub fn return_address_ptr() -> FramePosition {
        FramePosition(1)
    }
}

pub struct JumpTarget(pub isize);

#[derive(Clone)]
pub enum AllocationType {
    Stack,
    Heap,
}

#[derive(Clone)]
pub enum AlocationEvent {
    Alloc(AllocationType, usize),
    Free(AllocationType, usize),
}

#[derive(Clone)]
pub struct AllocationInstrumenter {
    pub events: Vec<AlocationEvent>,
}

impl AllocationInstrumenter {
    pub fn new() -> Self {
        Self { events: vec![] }
    }

    pub fn alloc(&mut self, ty: AllocationType, size: usize) {
        self.events.push(AlocationEvent::Alloc(ty, size));
    }

    pub fn free(&mut self, ty: AllocationType, size: usize) {
        self.events.push(AlocationEvent::Free(ty, size));
    }

    /// Returns the final memory usage in bytes (0 means no leak).
    pub fn final_memory_usage(&self) -> usize {
        let mut current_stack = 0usize;
        let mut current_heap = 0usize;

        for event in &self.events {
            match event {
                AlocationEvent::Alloc(AllocationType::Stack, size) => {
                    current_stack += size * 8;
                }
                AlocationEvent::Alloc(AllocationType::Heap, size) => {
                    current_heap += size * 8;
                }
                AlocationEvent::Free(AllocationType::Stack, size) => {
                    current_stack = current_stack.saturating_sub(*size * 8);
                }
                AlocationEvent::Free(AllocationType::Heap, size) => {
                    current_heap = current_heap.saturating_sub(*size * 8);
                }
            }
        }

        current_stack + current_heap
    }
}

#[derive(Clone)]
pub struct TableInfo {
    pub multiplicities_wit: *mut Field,
    pub num_indices: usize,
    pub kind: TableKind,
    pub length: usize,
    pub elem_inverses_witness_section_offset: usize,
    pub elem_inverses_constraint_section_offset: usize,
}

#[derive(Copy, Clone)]
pub struct FwdArrays {
    pub out_a: *mut Field,
    pub out_b: *mut Field,
    pub out_c: *mut Field,
    pub out_a_base: *mut Field,
    pub algebraic_witness: *mut Field,
    pub multiplicities_witness: *mut Field,
    pub lookups_a: *mut Field,
    pub lookups_b: *mut Field,
    pub lookups_c: *mut Field,
    pub elem_inverses_constraint_section_offset: usize,
    pub elem_inverses_witness_section_offset: usize,
}

#[derive(Copy, Clone)]
pub struct AdArrays {
    pub out_da: *mut Field,
    pub out_db: *mut Field,
    pub out_dc: *mut Field,
    pub ad_coeffs: *const Field,
    pub current_wit_off: usize,
    pub logup_wit_challenge_off: usize,
    pub current_wit_multiplicities_off: usize,
    pub current_wit_tables_off: usize,
    pub current_wit_lookups_off: usize,
    pub current_cnst_off: usize,
    pub current_cnst_tables_off: usize,
    pub current_cnst_lookups_off: usize,
}

pub union Arrays {
    pub as_forward: FwdArrays,
    pub as_ad: AdArrays,
}

/// Number of slots in the fixed-size table caches ([`VM::rgchk_tables`]/[`VM::spread_tables`]).
/// These are indexed by table size in bits, so this covers every width in `0..=32`.
pub const NUM_TABLE_SIZE_SLOTS: usize = 33;

struct InstructionProfile {
    profile: FlamegraphProfile,
    stack_ids: BTreeMap<Vec<usize>, FlamegraphStackId>,
    stack_scratch: Vec<usize>,
}

// Sentinel used only in cached profile stack keys. Real function indices address
// `DebugInfo::functions`, so they can never reach `usize::MAX`; callers must check
// this value before attempting a function-name lookup.
const UNKNOWN_FUNCTION_INDEX: usize = usize::MAX;

impl Default for InstructionProfile {
    fn default() -> Self {
        Self {
            profile: FlamegraphProfile::default(),
            stack_ids: BTreeMap::new(),
            stack_scratch: Vec::new(),
        }
    }
}

pub struct VM {
    pub data: Arrays,
    pub allocation_instrumenter: AllocationInstrumenter,
    pub tables: Vec<TableInfo>,
    /// Lazily-allocated rangecheck tables, indexed by table size in bits (a `2^bits`-row table).
    pub rgchk_tables: [Option<usize>; NUM_TABLE_SIZE_SLOTS],
    pub spread_tables: [Option<usize>; NUM_TABLE_SIZE_SLOTS],
    /// Indexed by `log2` of the shifted width, so only slots 0..=7 are ever used.
    pub pow2_tables: [Option<usize>; NUM_TABLE_SIZE_SLOTS],
    pub globals: *mut u64,
    pub struct_layouts: Vec<StructDescriptor>,
    /// Interned constant pool (flat words), read by the `mov_const_pool` opcode.
    pub constants: Vec<u64>,
    /// Set when the program executes a trap (e.g. a failed assertion). The
    /// interpreter checks this after dispatch returns to distinguish a clean
    /// halt from a trapped one.
    pub trapped: bool,
    program_base: *const u64,
    program_len: usize,
    debug_info: DebugInfo,
    pub(crate) stack_trace: Vec<StackFrame>,
    instruction_profile: Option<InstructionProfile>,
    /// Per-opcode `(invocation_count, accumulated_cycles)`, indexed by opcode
    /// discriminant. Written by generated handlers when `vm-profile` is enabled.
    #[cfg(feature = "vm-profile")]
    pub opcode_profile: Vec<(u64, u64)>,
}

impl VM {
    pub fn new_witgen(
        out_a: *mut Field,
        out_b: *mut Field,
        out_c: *mut Field,
        algebraic_witness: *mut Field,
        multiplicities_witness: *mut Field,
        lookups_a: *mut Field,
        lookups_b: *mut Field,
        lookups_c: *mut Field,
        elem_inverses_constraint_section_offset: usize,
        elem_inverses_witness_section_offset: usize,
        globals: *mut u64,
        struct_layouts: Vec<StructDescriptor>,
        constants: Vec<u64>,
    ) -> Self {
        Self {
            data: Arrays {
                as_forward: FwdArrays {
                    out_a,
                    out_b,
                    out_c,
                    out_a_base: out_a,
                    algebraic_witness,
                    multiplicities_witness,
                    lookups_b,
                    lookups_a,
                    lookups_c,
                    elem_inverses_constraint_section_offset,
                    elem_inverses_witness_section_offset,
                },
            },
            allocation_instrumenter: AllocationInstrumenter::new(),
            tables: vec![],
            rgchk_tables: [None; NUM_TABLE_SIZE_SLOTS],
            spread_tables: [None; NUM_TABLE_SIZE_SLOTS],
            pow2_tables: [None; NUM_TABLE_SIZE_SLOTS],
            globals,
            struct_layouts,
            constants,
            trapped: false,
            program_base: ptr::null(),
            program_len: 0,
            debug_info: DebugInfo::default(),
            stack_trace: Vec::new(),
            instruction_profile: None,
            #[cfg(feature = "vm-profile")]
            opcode_profile: vec![(0, 0); NUM_OPCODES],
        }
    }

    pub fn new_ad(
        out_da: *mut Field,
        out_db: *mut Field,
        out_dc: *mut Field,
        ad_coeffs: *const Field,

        witness_layout: WitnessLayout,
        constraints_layout: ConstraintsLayout,
        globals: *mut u64,
        struct_layouts: Vec<StructDescriptor>,
        constants: Vec<u64>,
    ) -> Self {
        Self {
            data: Arrays {
                as_ad: AdArrays {
                    out_da,
                    out_db,
                    out_dc,
                    ad_coeffs,
                    current_wit_off: 0,
                    // FIELD-ASSUMPTION: L4-logup-challenges (~9 sites)
                    // The AD pass reads the single challenge at `logup_wit_challenge_off`
                    // (alpha) and `+1` (beta). For K challenges these become
                    // `off + 2k`/`off + 2k + 1`, looped over the K inverse-witness copies.
                    logup_wit_challenge_off: witness_layout.challenges_start(),
                    current_wit_multiplicities_off: witness_layout.multiplicities_start(),
                    current_wit_tables_off: witness_layout.tables_data_start(),
                    current_wit_lookups_off: witness_layout.lookups_data_start(),
                    current_cnst_off: 0,
                    current_cnst_tables_off: constraints_layout.tables_data_start(),
                    current_cnst_lookups_off: constraints_layout.lookups_data_start(),
                },
            },
            allocation_instrumenter: AllocationInstrumenter::new(),
            tables: vec![],
            rgchk_tables: [None; NUM_TABLE_SIZE_SLOTS],
            spread_tables: [None; NUM_TABLE_SIZE_SLOTS],
            pow2_tables: [None; NUM_TABLE_SIZE_SLOTS],
            globals,
            struct_layouts,
            constants,
            trapped: false,
            program_base: ptr::null(),
            program_len: 0,
            debug_info: DebugInfo::default(),
            stack_trace: Vec::new(),
            instruction_profile: None,
            #[cfg(feature = "vm-profile")]
            opcode_profile: vec![(0, 0); NUM_OPCODES],
        }
    }

    pub fn set_debug_context(
        &mut self,
        program_base: *const u64,
        program_len: usize,
        debug_info: DebugInfo,
    ) {
        self.program_base = program_base;
        self.program_len = program_len;
        self.debug_info = debug_info;
    }

    fn program_offset(&self, pc: *const u64) -> Option<usize> {
        program_offset(self.program_base, self.program_len, pc)
    }

    pub fn enable_instruction_profile(&mut self) {
        self.instruction_profile = Some(InstructionProfile::default());
    }

    pub fn take_instruction_profile(&mut self) -> FlamegraphProfile {
        self.instruction_profile
            .take()
            .map(|profile| profile.profile)
            .unwrap_or_default()
    }

    #[inline(always)]
    pub fn instruction_profile_enabled(&self) -> bool {
        self.instruction_profile.is_some()
    }

    /// Count one simulated instruction against its root-first Noir call stack.
    #[cold]
    #[inline(never)]
    pub fn record_instruction(&mut self, pc: *const u64, frame: Frame) {
        let instruction_profile = self
            .instruction_profile
            .as_mut()
            .expect("instruction profiling is enabled before recording instructions");

        let program_base = self.program_base;
        let program_len = self.program_len;
        let debug_info = &self.debug_info;
        let stack = &mut instruction_profile.stack_scratch;
        stack.clear();

        if let Some(offset) = program_offset(program_base, program_len, pc)
            && let Some(function_index) = debug_info.function_index_at(offset)
        {
            stack.push(function_index);
        }

        let mut current = frame;
        while !current.data.is_null() {
            let parent_data = unsafe { *current.data.offset(-1) as *mut u64 };
            if parent_data.is_null() {
                break;
            }

            let return_pc = unsafe { *current.data.offset(1) as *const u64 };
            if let Some(return_offset) = program_offset(program_base, program_len, return_pc)
                && let Some(function_index) =
                    debug_info.function_index_at(return_offset.saturating_sub(1))
            {
                stack.push(function_index);
            }
            current = Frame { data: parent_data };
        }
        stack.reverse();
        if stack.is_empty() {
            stack.push(UNKNOWN_FUNCTION_INDEX);
        }

        let stack_id = if let Some(stack_id) = instruction_profile.stack_ids.get(stack.as_slice()) {
            *stack_id
        } else {
            let names = stack.iter().map(|function_index| {
                if *function_index == UNKNOWN_FUNCTION_INDEX {
                    "<unknown>".to_string()
                } else {
                    debug_info
                        .function_name(*function_index)
                        .unwrap_or("<unknown>")
                        .to_string()
                }
            });
            let stack_id = instruction_profile
                .profile
                .intern_stack(names)
                .expect("instruction profile stack is non-empty");
            instruction_profile
                .stack_ids
                .insert(stack.clone(), stack_id);
            stack_id
        };
        instruction_profile.profile.record_interned(stack_id, 1);
    }

    fn stack_frames_at(&self, pc: *const u64, frame: Frame) -> Vec<StackFrame> {
        let mut stack_trace = Vec::new();
        if let Some(offset) = self.program_offset(pc)
            && let Some(frame) = self.debug_info.stack_frame_at(offset)
        {
            stack_trace.push(frame);
        }

        let mut current = frame;
        while !current.data.is_null() {
            let parent_data = unsafe { *current.data.offset(-1) as *mut u64 };
            if parent_data.is_null() {
                break;
            }

            let return_pc = unsafe { *current.data.offset(1) as *const u64 };
            if let Some(return_offset) = self.program_offset(return_pc)
                && let Some(frame) = self
                    .debug_info
                    .stack_frame_at(return_offset.saturating_sub(1))
            {
                stack_trace.push(frame);
            }
            current = Frame { data: parent_data };
        }
        stack_trace
    }

    pub fn capture_trap(&mut self, pc: *const u64, frame: Frame) {
        self.trapped = true;
        self.stack_trace = self.stack_frames_at(pc, frame);
    }
}

fn program_offset(program_base: *const u64, program_len: usize, pc: *const u64) -> Option<usize> {
    if program_base.is_null() {
        return None;
    }
    let offset = unsafe { pc.offset_from(program_base) };
    (offset >= 0 && (offset as usize) < program_len).then_some(offset as usize)
}

/// The value column of a powers-of-two table: `2^n` as a field element, for `n` in `0..len`.
///
/// Shared by both VM fills — `run_phase2`'s and `dpow2_lookup_field`'s — so they cannot diverge.
/// Two more fills build the same column independently and must agree at every row or the backends
/// disagree about the table's contents: `hlssa_to_llssa::emit_pow2_ad_init_body` carries this same
/// accumulator through the loop it generates, and `hlssa_to_r1cs`'s `Table::Pow2` arm exponentiates
/// instead.
pub(crate) fn pow2_rows(len: usize) -> impl Iterator<Item = Field> {
    let mut acc = Field::from(1u64);
    (0..len).map(move |_| {
        let row = acc;
        acc += acc;
        row
    })
}

/// Compute spread of a u32: interleave zero bits between each bit.
pub(crate) fn spread_bits(v: u32) -> u64 {
    let mut x = v as u64;
    x = (x | (x << 16)) & 0x0000_FFFF_0000_FFFF;
    x = (x | (x << 8)) & 0x00FF_00FF_00FF_00FF;
    x = (x | (x << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333;
    x = (x | (x << 1)) & 0x5555_5555_5555_5555;
    x
}

/// Compact even-positioned bits into contiguous low bits.
fn compact_bits(mut x: u64) -> u32 {
    x &= 0x5555_5555_5555_5555;
    x = (x | (x >> 1)) & 0x3333_3333_3333_3333;
    x = (x | (x >> 2)) & 0x0F0F_0F0F_0F0F_0F0F;
    x = (x | (x >> 4)) & 0x00FF_00FF_00FF_00FF;
    x = (x | (x >> 8)) & 0x0000_FFFF_0000_FFFF;
    x = (x | (x >> 16)) & 0x0000_0000_FFFF_FFFF;
    x as u32
}

/// Extract even bits and odd bits from a spread sum. Returns (odd_bits, even_bits).
fn unspread_bits(v: u64) -> (u32, u32) {
    let even = compact_bits(v);
    let odd = compact_bits(v >> 1);
    (odd, even)
}

/// The row `key` addresses in a table of `length` entries, or `None` when it
/// addresses no row at all.
///
/// A lookup argument proves membership by bumping the multiplicity of the row it
/// landed on, so a key outside `0..length` is not a wider index — it is a
/// witness that cannot satisfy the lookup, and the caller must trap. Rejecting
/// on the high limbs matters as much as the bound: the bump is addressed by the
/// low limb alone, so a key just under the modulus would otherwise truncate to
/// an arbitrary `u64` and land outside the multiplicities buffer entirely.
#[inline(always)]
fn table_row_index(key: Field, length: usize) -> Option<u64> {
    let limbs = ark_ff::PrimeField::into_bigint(key).0;
    let (low, high) = limbs.split_first().expect("field bigint has no limbs");
    (high.iter().all(|limb| *limb == 0) && *low < length as u64).then_some(*low)
}

/// Emit a forward key-value lookup: bump multiplicity and write 2 lookup tape entries.
///
/// Returns `false`, having written nothing, when an active lookup's `key` is not
/// a row of the table; the caller must trap. An inactive lookup (`flag_u64 == 0`)
/// bumps no multiplicity and so is always emittable — a guarded-off lookup must
/// not be able to fail.
#[must_use]
unsafe fn forward_kv_lookup_emit(
    table_idx: usize,
    key: Field,
    result: Field,
    flag_u64: u64,
    vm: &mut VM,
) -> bool {
    let table_info = &vm.tables[table_idx];

    let key_row = if flag_u64 != 0 {
        match table_row_index(key, table_info.length) {
            Some(row) => Some(row),
            None => return false,
        }
    } else {
        None
    };

    // Entry 1 (x-constraint): table_id, result_value, 0
    unsafe {
        *(vm.data.as_forward.lookups_a as *mut u64) = table_idx as u64;
        *vm.data.as_forward.lookups_b = result;
        *(vm.data.as_forward.lookups_c as *mut u64) = 0;
        vm.data.as_forward.lookups_a = vm.data.as_forward.lookups_a.offset(1);
        vm.data.as_forward.lookups_b = vm.data.as_forward.lookups_b.offset(1);
        vm.data.as_forward.lookups_c = vm.data.as_forward.lookups_c.offset(1);
    }

    // Entry 2 (y-constraint): table_id, key, flag
    unsafe {
        *(vm.data.as_forward.lookups_a as *mut u64) = table_idx as u64;
        if let Some(key_u64) = key_row {
            let ptr = table_info.multiplicities_wit.offset(key_u64 as isize);
            *(ptr as *mut u64) += flag_u64;
            *(vm.data.as_forward.lookups_b as *mut u64) = key_u64;
        } else {
            *(vm.data.as_forward.lookups_b as *mut Field) = key;
        }
        *(vm.data.as_forward.lookups_c as *mut u64) = flag_u64;
        vm.data.as_forward.lookups_a = vm.data.as_forward.lookups_a.offset(1);
        vm.data.as_forward.lookups_b = vm.data.as_forward.lookups_b.offset(1);
        vm.data.as_forward.lookups_c = vm.data.as_forward.lookups_c.offset(1);
    }

    true
}

/// Emit AD bumps for a key-value lookup (x-constraint + y-constraint + sum).
unsafe fn ad_kv_lookup_emit(
    table_idx: usize,
    key: BoxedValue,
    result: BoxedValue,
    flag: BoxedValue,
    vm: &mut VM,
) {
    let table_info = &vm.tables[table_idx];
    let cnst_off = table_info.elem_inverses_constraint_section_offset;
    let length = table_info.length;
    // Sum constraint sits past the table's per-entry constraints: spread and powers-of-two
    // tables fold each entry into one constraint, arrays use two. Rangecheck tables
    // are key-only and never reach this key-value lookup path.
    let sum_off = match table_info.kind {
        TableKind::Spread | TableKind::Pow2 => cnst_off + length,
        TableKind::Array => cnst_off + 2 * length,
        TableKind::RangeCheck => panic!("ad_kv_lookup_emit called on a rangecheck table"),
    };

    let x_coeff = unsafe {
        let r = *vm
            .data
            .as_ad
            .ad_coeffs
            .add(vm.data.as_ad.current_cnst_lookups_off);
        vm.data.as_ad.current_cnst_lookups_off += 1;
        r
    };
    let x_wit_off = unsafe {
        let r = vm.data.as_ad.current_wit_lookups_off;
        vm.data.as_ad.current_wit_lookups_off += 1;
        r
    };
    let y_coeff = unsafe {
        let r = *vm
            .data
            .as_ad
            .ad_coeffs
            .add(vm.data.as_ad.current_cnst_lookups_off);
        vm.data.as_ad.current_cnst_lookups_off += 1;
        r
    };
    let y_wit_off = unsafe {
        let r = vm.data.as_ad.current_wit_lookups_off;
        vm.data.as_ad.current_wit_lookups_off += 1;
        r
    };
    let inv_sum_coeff = unsafe { *vm.data.as_ad.ad_coeffs.add(sum_off) };

    // x-constraint: beta * result - x_lookup = 0
    unsafe {
        *vm.data
            .as_ad
            .out_da
            .add(vm.data.as_ad.logup_wit_challenge_off + 1) += x_coeff;
    }
    result.bump_db(x_coeff, vm);
    unsafe {
        *vm.data.as_ad.out_dc.add(x_wit_off) -= x_coeff;
    }

    // y-constraint: y * (alpha - x_lookup - key) = flag
    unsafe {
        *vm.data.as_ad.out_da.add(y_wit_off) += y_coeff;
        *vm.data
            .as_ad
            .out_db
            .add(vm.data.as_ad.logup_wit_challenge_off) += y_coeff;
        *vm.data.as_ad.out_db.add(x_wit_off) -= y_coeff;
    }
    key.bump_db(-y_coeff, vm);
    flag.bump_dc(y_coeff, vm);

    // Sum constraint
    unsafe {
        *vm.data.as_ad.out_dc.add(y_wit_off) += inv_sum_coeff;
    }
}

/// The VM's opcode set.
///
/// An integer frame cell holds a value **masked to its declared width**: an `i8 -1` and a `u8 255`
/// are the same cell, `0x00000000000000FF`. Nothing about that pattern says how to read it, which
/// is the same position HLSSA's `TypeExpr::Int(bits)` and LLSSA's `Type::Int` take. So:
///
/// - **Reading from Opcode Name:** Where signed and unsigned disagree on the answer, there are two
///   opcodes and each names its reading: `udiv_int`/`sdiv_int`, `urem_int`/`srem_int`,
///   `ult_int`/`slt_int`, and `ushr_int`/`ashr_int`. Where they agree there is a single opcode and
///   no prefix. The pairs HLSSA keeps distinct for `Add`/`Sub`/`Mul`/`Shl` differ only in _when
///   they trap_, and the VM does not trap: overflow rejection is guard IR emitted before codegen.
///   See `BinaryArithOpKind`'s doc.
/// - **Suffix is the Lane**: `_int` is one 64-bit cell (widths 1..=64), `_int128` is two (widths
///   65..=128), `_intn` is `ceil(bits/64)` (every wider width), and `_field` is four.
/// - **`bits` is the Width:** It is never a signedness marker. There are two reasons an opcode
///   takes one:
///   - _Semantic_, where the answer depends on it: to re-mask an output that can exceed the width
///     (`add_int`, `sub_int`, `mul_int`, `shl_int`, `not_int`), to reduce a shift amount, or to
///     locate the sign bit at `bits - 1` (`sdiv_int`, `srem_int`, `slt_int`, `ashr_int`).
///     Operations correct on any two already-masked operands (`and_int`, `or_int`, `xor_int`,
///     `eq_int`, `ult_int`, `udiv_int`, `urem_int`, and their `_int128` counterparts) take no
///     width.
///   - _Structural_, where the operand's extent is not otherwise known. A cell-lane operand is one
///     cell and an `_int128` operand is two, so neither lane needs this; a wide operand is as many
///     cells as its width says, which is why `and_intn` carries a width it never reads.
#[interpreter]
mod def {
    #[raw_opcode]
    fn jmp(pc: *const u64, frame: Frame, vm: &mut VM, target: JumpTarget) -> (*const u64, Frame) {
        let pc = unsafe { pc.offset(target.0) };
        (pc, frame)
    }

    #[raw_opcode]
    fn jmp_if(
        pc: *const u64,
        frame: Frame,
        vm: &mut VM,
        #[frame] cond: u64,
        if_t: JumpTarget,
        if_f: JumpTarget,
    ) -> (*const u64, Frame) {
        let target = if cond != 0 { if_t } else { if_f };
        let pc = unsafe { pc.offset(target.0) };
        (pc, frame)
    }

    #[raw_opcode]
    fn call(
        pc: *const u64,
        frame: Frame,
        vm: &mut VM,
        func: JumpTarget,
        args: &[(usize, FramePosition)],
        ret: FramePosition,
    ) -> (*const u64, Frame) {
        let func_pc = unsafe { pc.offset(func.0) };
        let func_frame_size = unsafe { *func_pc.offset(-1) };
        let new_frame = Frame::push(func_frame_size, frame, vm);
        let ret_data_ptr = unsafe { frame.data.add(ret.0) };
        let ret_pc = unsafe { pc.offset(4 + 2 * args.len() as isize) };

        unsafe {
            *new_frame.data = ret_data_ptr as u64;
            *new_frame.data.offset(1) = ret_pc as u64;
        };

        let mut current_child = unsafe { new_frame.data.offset(2) };

        for (arg_size, arg_pos) in args {
            unsafe { frame.write_to(current_child, arg_pos.0 as isize, *arg_size) };
            current_child = unsafe { current_child.add(*arg_size) };
        }

        (func_pc, new_frame)
    }

    #[raw_opcode]
    fn ret(_pc: *const u64, frame: Frame, vm: &mut VM) -> (*const u64, Frame) {
        let ret_address = unsafe { *frame.data.offset(1) } as *mut u64;
        let new_frame = frame.pop(vm);
        if new_frame.data.is_null() {
            // Halt: returning a null pc tells `dispatch` to stop.
            return (std::ptr::null(), new_frame);
        }
        (ret_address, new_frame)
    }

    /// Halts execution and marks the VM as trapped. The interpreter analogue
    /// of the WASM target's `unreachable`: the assert-family opcodes delegate
    /// here when their check fails.
    #[raw_opcode]
    fn trap(pc: *const u64, frame: Frame, vm: &mut VM) -> (*const u64, Frame) {
        vm.capture_trap(pc, frame);
        (std::ptr::null(), frame)
    }

    #[raw_opcode]
    fn r1c(
        pc: *const u64,
        frame: Frame,
        vm: &mut VM,
        #[frame] a: Field,
        #[frame] b: Field,
        #[frame] c: Field,
    ) -> (*const u64, Frame) {
        unsafe {
            *vm.data.as_forward.out_a = a;
            *vm.data.as_forward.out_b = b;
            *vm.data.as_forward.out_c = c;
        }

        unsafe {
            vm.data.as_forward.out_a = vm.data.as_forward.out_a.offset(1);
            vm.data.as_forward.out_b = vm.data.as_forward.out_b.offset(1);
            vm.data.as_forward.out_c = vm.data.as_forward.out_c.offset(1);
        };
        let pc = unsafe { pc.offset(4) };
        (pc, frame)
    }

    #[raw_opcode]
    fn write_witness(
        pc: *const u64,
        frame: Frame,
        vm: &mut VM,
        #[frame] val: Field,
    ) -> (*const u64, Frame) {
        unsafe {
            *vm.data.as_forward.algebraic_witness = val;
            vm.data.as_forward.algebraic_witness = vm.data.as_forward.algebraic_witness.offset(1);
        };
        let pc = unsafe { pc.offset(2) };
        (pc, frame)
    }

    #[opcode]
    fn nop() {}

    #[opcode]
    fn mov_const(#[out] res: *mut u64, val: u64) {
        unsafe {
            *res = val;
        }
    }

    #[opcode]
    fn mov_frame(frame: Frame, target: FramePosition, source: FramePosition, size: usize) {
        frame.memcpy(target.0 as isize, source.0 as isize, size);
    }

    #[opcode]
    fn write_ptr(
        frame: Frame,
        #[frame] ptr: *mut u64,
        offset: isize,
        src: FramePosition,
        size: usize,
    ) {
        let ptr = unsafe { ptr.offset(offset) };
        unsafe { frame.write_to(ptr, src.0 as isize, size) };
    }

    #[opcode]
    fn add_int(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64, bits: u64) {
        unsafe {
            *res = cell_add(a, b, bits);
        }
    }

    #[opcode]
    fn sub_int(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64, bits: u64) {
        unsafe {
            *res = cell_sub(a, b, bits);
        }
    }

    #[opcode]
    fn mul_int(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64, bits: u64) {
        unsafe {
            *res = cell_mul(a, b, bits);
        }
    }

    #[opcode]
    fn add_int128(#[out] res: *mut Int128, #[frame] a: Int128, #[frame] b: Int128, bits: u64) {
        unsafe { *res = double_add(a, b, bits) };
    }

    #[opcode]
    fn sub_int128(#[out] res: *mut Int128, #[frame] a: Int128, #[frame] b: Int128, bits: u64) {
        unsafe { *res = double_sub(a, b, bits) };
    }

    #[opcode]
    fn mul_int128(#[out] res: *mut Int128, #[frame] a: Int128, #[frame] b: Int128, bits: u64) {
        unsafe { *res = double_mul(a, b, bits) };
    }

    /// Divide, reading both cells as unsigned. Total.
    #[opcode]
    fn udiv_int(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64) {
        unsafe {
            *res = cell_udiv(a, b);
        }
    }

    /// Remainder, reading both cells as unsigned. Total.
    #[opcode]
    fn urem_int(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64) {
        unsafe {
            *res = cell_urem(a, b);
        }
    }

    /// Divide, reading both cells as two's complement.
    ///
    /// Truncates toward zero, so the quotient's sign is the operands' xor. Total.
    #[opcode]
    fn sdiv_int(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64, bits: u64) {
        unsafe {
            *res = cell_sdiv(a, b, bits);
        }
    }

    /// Remainder, reading both cells as two's complement.
    ///
    /// Takes the dividend's sign, which is what truncation toward zero implies. Total.
    #[opcode]
    fn srem_int(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64, bits: u64) {
        unsafe {
            *res = cell_srem(a, b, bits);
        }
    }

    #[opcode]
    fn udiv_int128(#[out] res: *mut Int128, #[frame] a: Int128, #[frame] b: Int128) {
        unsafe { *res = a.unsigned_div(b) };
    }

    #[opcode]
    fn urem_int128(#[out] res: *mut Int128, #[frame] a: Int128, #[frame] b: Int128) {
        unsafe { *res = a.unsigned_rem(b) };
    }

    #[opcode]
    fn and_int(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64) {
        unsafe {
            *res = cell_and(a, b);
        }
    }

    #[opcode]
    fn or_int(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64) {
        unsafe {
            *res = cell_or(a, b);
        }
    }

    #[opcode]
    fn xor_int(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64) {
        unsafe {
            *res = cell_xor(a, b);
        }
    }

    #[opcode]
    fn and_int128(#[out] res: *mut Int128, #[frame] a: Int128, #[frame] b: Int128) {
        unsafe { *res = a & b };
    }

    #[opcode]
    fn or_int128(#[out] res: *mut Int128, #[frame] a: Int128, #[frame] b: Int128) {
        unsafe { *res = a | b };
    }

    #[opcode]
    fn xor_int128(#[out] res: *mut Int128, #[frame] a: Int128, #[frame] b: Int128) {
        unsafe { *res = a ^ b };
    }

    /// Left shift, wrapping at the operand width.
    ///
    /// A left shift is one map on the bit pattern, so there is no signed form: what the signed
    /// HLSSA `SShl` additionally rejects is a _negative amount_, and that rejection is guard IR
    /// (`pure_guards::emit_invalid_shift_cond`), not something this opcode can see.
    #[opcode]
    fn shl_int(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64, bits: u64) {
        unsafe {
            *res = cell_shl(a, b, bits);
        }
    }

    /// Logical right shift: zero-fill, the lowering for an unsigned `>>`.
    ///
    /// Takes `bits` only to mask the amount — the result of a zero-filling shift on an
    /// already-masked operand cannot exceed the width, so there is nothing to re-mask. `ashr_int`
    /// below needs `bits` for a second reason as well: to find the sign bit.
    #[opcode]
    fn ushr_int(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64, bits: u64) {
        unsafe {
            *res = cell_ushr(a, b, bits);
        }
    }

    /// Arithmetic right shift: the lowering for a signed `>>`.
    ///
    /// A signed value is held masked to `bits` inside a `u64`, so the sign has to be recovered
    /// before shifting — sign-extend to `i64`, shift there, then re-mask. That is the same
    /// `signed_cell` preamble `sdiv_int`/`srem_int`/`slt_int` use.
    ///
    /// The shift count is reduced modulo `bits` to match what the LLVM backend does for this op
    /// (`llssa_to_llvm::reduce_shift_count`), so an over-shift cannot make the two backends disagree
    /// — and cannot panic on the `i64` shift.
    #[opcode]
    fn ashr_int(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64, bits: u64) {
        unsafe {
            *res = cell_ashr(a, b, bits);
        }
    }

    /// Left shift in the 128-bit lane.
    ///
    /// The amount is masked by [`shift_amount_128`].
    #[opcode]
    fn shl_int128(#[out] res: *mut Int128, #[frame] a: Int128, #[frame] b: Int128, bits: u64) {
        unsafe { *res = double_shl(a, b, bits) };
    }

    /// Logical right shift in the 128-bit lane; zero-fill, masked as `shl_int128` is.
    #[opcode]
    fn ushr_int128(#[out] res: *mut Int128, #[frame] a: Int128, #[frame] b: Int128, bits: u64) {
        unsafe { *res = double_ushr(a, b, bits) };
    }

    /// Bitwise complement, re-masked to the operand width.
    ///
    /// Takes `bits` because `!a` sets every bit of the host `u64`, including the ones above the
    /// declared width — which would break the masked-cell invariant for every later reader.
    /// `lattice::eval_not` folds this as `!x & bit_mask(bits)` and LLVM's `not` is on an
    /// exact-width `iN`, so the mask is also what keeps the three implementations in agreement.
    ///
    /// **Currently unreachable**, and the mask is defence for when it stops being so:
    /// `LowerWitnessBitwiseOps::lower_not` rewrites _every_ `Not` — pure ones as well as witness
    /// ones, unlike the `And`/`Or`/`Xor` arm beside it — into `(2^bits - 1) - value` before
    /// codegen, so nothing emits this opcode today. Lowering a pure `Not` to it instead is the
    /// obvious optimisation, and that is the moment an unmasked `!a` would start returning wrong
    /// answers.
    #[opcode]
    fn not_int(#[out] res: *mut u64, #[frame] a: u64, bits: u64) {
        unsafe {
            *res = cell_complement(a, bits);
        }
    }

    /// Bitwise complement of a double-lane cell pair, re-masked to the operand width.
    ///
    /// Takes `bits` for the reason `not_int` does: the lane holds `65..=128`, so `!a` sets every
    /// bit of the pair including the ones above the declared width, and leaving them set would
    /// break the masked-cell invariant for every later reader. At the lane's widest width the mask
    /// is a no-op, which is why an unmasked complement passes a sweep that stops at 128 —
    /// `the_complement_opcode_agrees_with_the_model` covers both lanes at four narrower widths for
    /// that reason. Unreachable today for the same reason as `not_int`.
    #[opcode]
    fn not_int128(#[out] res: *mut Int128, #[frame] a: Int128, bits: u64) {
        unsafe { *res = double_complement(a, bits) };
    }

    #[opcode]
    fn eq_int(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64) {
        unsafe {
            *res = (a == b) as u64;
        }
    }

    #[opcode]
    fn ult_int(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64) {
        unsafe {
            *res = (a < b) as u64;
        }
    }

    #[opcode]
    fn eq_int128(#[out] res: *mut u64, #[frame] a: Int128, #[frame] b: Int128) {
        unsafe {
            *res = (a == b) as u64;
        }
    }

    #[opcode]
    fn eq_field(#[out] res: *mut u64, #[frame] a: Field, #[frame] b: Field) {
        unsafe {
            *res = (a == b) as u64;
        }
    }

    /// Order two field elements by their canonical integer representative.
    ///
    /// No `u`/`s` prefix, unlike `ult_int`: a field element has no two's-complement reading, so
    /// there is no second ordering for this one to be distinguished from. `hlssa_to_llssa` says
    /// the same thing by making an `SLt` over a `Field` an ICE rather than a lowering.
    #[opcode]
    fn lt_field(#[out] res: *mut u64, #[frame] a: Field, #[frame] b: Field) {
        unsafe {
            *res = (ark_ff::PrimeField::into_bigint(a) < ark_ff::PrimeField::into_bigint(b)) as u64;
        }
    }

    #[opcode]
    fn ult_int128(#[out] res: *mut u64, #[frame] a: Int128, #[frame] b: Int128) {
        unsafe {
            *res = a.unsigned_lt(b) as u64;
        }
    }

    #[opcode]
    fn slt_int(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64, bits: u64) {
        unsafe {
            *res = (signed_cell(a, bits) < signed_cell(b, bits)) as u64;
        }
    }

    #[opcode]
    fn truncate_int(#[out] res: *mut u64, #[frame] a: u64, to_bits: u64) {
        unsafe {
            let mask = cell_mask(to_bits);
            *res = a & mask;
        }
    }

    #[opcode]
    fn truncate_int128(#[out] res: *mut Int128, #[frame] a: Int128, to_bits: u64) {
        unsafe { *res = a.truncate(to_bits) };
    }

    // THE WIDE LANE
    // --------------------------------------------------------------------------------------
    //
    // One opcode per operation over a value of `ceil(bits/64)` frame cells, each body a call into
    // [`int_limbs`]. Every operand is one encoded word, a frame position, as in the narrow lanes,
    // while the width immediate is what says how many cells that position covers.
    //
    // That is also why nearly every opcode here carries a width where its narrower cousins do
    // not: this is the lane where the _structural_ reason in the opcode set's own doc applies, and
    // it is why `and_intn` takes a width its body never reads.

    #[opcode]
    fn add_intn(
        bits: u64,
        #[frame_slice(bits)] res: &mut [u64],
        #[frame_slice(bits)] a: &[u64],
        #[frame_slice(bits)] b: &[u64],
    ) {
        int_limbs::add(res, a, b, bits);
    }

    #[opcode]
    fn sub_intn(
        bits: u64,
        #[frame_slice(bits)] res: &mut [u64],
        #[frame_slice(bits)] a: &[u64],
        #[frame_slice(bits)] b: &[u64],
    ) {
        int_limbs::sub(res, a, b, bits);
    }

    #[opcode]
    fn mul_intn(
        bits: u64,
        #[frame_slice(bits)] res: &mut [u64],
        #[frame_slice(bits)] a: &[u64],
        #[frame_slice(bits)] b: &[u64],
    ) {
        int_limbs::mul(res, a, b, bits);
    }

    /// Divide, reading both operands as unsigned. Total, as the narrow lane's `cell_udiv` is.
    ///
    /// The scratch buffer is the remainder, which one long division produces alongside the
    /// quotient whether or not the opcode wants it. Allocating per division is a cost the narrow
    /// lanes do not pay; it is small beside the division itself, which is quadratic in limbs.
    #[opcode]
    fn udiv_intn(
        bits: u64,
        #[frame_slice(bits)] res: &mut [u64],
        #[frame_slice(bits)] a: &[u64],
        #[frame_slice(bits)] b: &[u64],
    ) {
        let mut discarded = vec![0u64; res.len()];
        int_limbs::udivrem(res, &mut discarded, a, b, bits);
    }

    /// Remainder, reading both operands as unsigned. Total, as [`udiv_intn`] is.
    #[opcode]
    fn urem_intn(
        bits: u64,
        #[frame_slice(bits)] res: &mut [u64],
        #[frame_slice(bits)] a: &[u64],
        #[frame_slice(bits)] b: &[u64],
    ) {
        let mut discarded = vec![0u64; res.len()];
        int_limbs::udivrem(&mut discarded, res, a, b, bits);
    }

    #[opcode]
    fn and_intn(
        bits: u64,
        #[frame_slice(bits)] res: &mut [u64],
        #[frame_slice(bits)] a: &[u64],
        #[frame_slice(bits)] b: &[u64],
    ) {
        int_limbs::and(res, a, b);
    }

    #[opcode]
    fn or_intn(
        bits: u64,
        #[frame_slice(bits)] res: &mut [u64],
        #[frame_slice(bits)] a: &[u64],
        #[frame_slice(bits)] b: &[u64],
    ) {
        int_limbs::or(res, a, b);
    }

    #[opcode]
    fn xor_intn(
        bits: u64,
        #[frame_slice(bits)] res: &mut [u64],
        #[frame_slice(bits)] a: &[u64],
        #[frame_slice(bits)] b: &[u64],
    ) {
        int_limbs::xor(res, a, b);
    }

    #[opcode]
    fn not_intn(bits: u64, #[frame_slice(bits)] res: &mut [u64], #[frame_slice(bits)] a: &[u64]) {
        int_limbs::not(res, a, bits);
    }

    /// Left shift, wrapping at the operand width, with the amount reduced modulo that width.
    ///
    /// The amount is a value of the same width as the operand, as it is in the narrow lanes: the
    /// elaborator unifies the two operands of an infix operator.
    #[opcode]
    fn shl_intn(
        bits: u64,
        #[frame_slice(bits)] res: &mut [u64],
        #[frame_slice(bits)] a: &[u64],
        #[frame_slice(bits)] b: &[u64],
    ) {
        int_limbs::shl_by(res, a, int_limbs::shift_amount(b, bits), bits);
    }

    /// Logical right shift: zero-fill, the lowering for an unsigned `>>`.
    #[opcode]
    fn ushr_intn(
        bits: u64,
        #[frame_slice(bits)] res: &mut [u64],
        #[frame_slice(bits)] a: &[u64],
        #[frame_slice(bits)] b: &[u64],
    ) {
        int_limbs::ushr_by(res, a, int_limbs::shift_amount(b, bits), bits);
    }

    #[opcode]
    fn eq_intn(
        bits: u64,
        #[out] res: *mut u64,
        #[frame_slice(bits)] a: &[u64],
        #[frame_slice(bits)] b: &[u64],
    ) {
        unsafe {
            *res = u64::from(int_limbs::eq(a, b));
        }
    }

    #[opcode]
    fn ult_intn(
        bits: u64,
        #[out] res: *mut u64,
        #[frame_slice(bits)] a: &[u64],
        #[frame_slice(bits)] b: &[u64],
    ) {
        unsafe {
            *res = u64::from(int_limbs::ult(a, b));
        }
    }

    /// Move a value between two integer widths, zero-filling above the source.
    ///
    /// One opcode for both directions: narrowing keeps the low `to_bits` bits and widening fills
    /// with the source's unsigned high bits, which are zero. There is no sign-extending form
    /// because bytecode never sees an `SExt`.
    #[opcode]
    fn cast_intn(
        from_bits: u64,
        to_bits: u64,
        #[frame_slice(to_bits)] res: &mut [u64],
        #[frame_slice(from_bits)] a: &[u64],
    ) {
        int_limbs::zero_extend(res, a, from_bits, to_bits);
    }

    #[opcode]
    fn add_field(#[out] res: *mut Field, #[frame] a: Field, #[frame] b: Field) {
        unsafe {
            *res = a + b;
        }
    }

    #[opcode]
    fn sub_field(#[out] res: *mut Field, #[frame] a: Field, #[frame] b: Field) {
        unsafe {
            *res = a - b;
        }
    }

    #[opcode]
    #[inline(never)]
    // FIELD-ASSUMPTION: L4-inverse
    fn div_field(#[out] res: *mut Field, #[frame] a: Field, #[frame] b: Field) {
        unsafe {
            *res = if b == Field::ZERO { Field::ZERO } else { a / b };
        }
    }

    #[opcode]
    fn mul_field(#[out] res: *mut Field, #[frame] a: Field, #[frame] b: Field) {
        unsafe {
            *res = a * b;
        }
    }

    #[opcode]
    fn cast_field_to_int(#[out] res: *mut u64, #[frame] a: Field) {
        unsafe {
            *res = ark_ff::PrimeField::into_bigint(a).0[0];
        }
    }

    #[opcode]
    fn cast_field_to_int128(#[out] res: *mut Int128, #[frame] a: Field) {
        let limbs = ark_ff::PrimeField::into_bigint(a).0;
        unsafe {
            *res = Int128 {
                lo: limbs[0],
                hi: limbs[1],
            }
        };
    }

    #[opcode]
    fn cast_int_to_field(#[out] res: *mut Field, #[frame] a: u64) {
        unsafe {
            *res = From::from(a);
        }
    }

    #[opcode]
    fn cast_int128_to_field(#[out] res: *mut Field, #[frame] a: Int128) {
        unsafe {
            *res = Field::from(a.to_u128());
        }
    }

    /// The element's canonical limbs, read as a `bits`-wide integer.
    ///
    /// Total at every width: an element has no bits above the ones [`FELT_LIMBS`] carries, so a
    /// wider target is zero-filled and a narrower one keeps its low `bits`, which is the
    /// truncating reading `docs/int-semantics.md` gives the `Field -> uN` cast. It masks its own
    /// top cell, unlike its cell-lane sibling, because it has the width to mask to.
    #[opcode]
    fn cast_field_to_intn(bits: u64, #[frame_slice(bits)] res: &mut [u64], #[frame] a: Field) {
        let limbs = ark_ff::PrimeField::into_bigint(a).0;
        int_limbs::zero_extend(res, &limbs, (FELT_LIMBS * 64) as u64, bits);
    }

    /// A `bits`-wide integer moved into a field element.
    ///
    /// The limbs are the element's, so this needs every one of them to be a limb the element has:
    /// past that there is nowhere for the value to go. The compiler never asks for one, because
    /// width validation refuses a cast to `Field` above the widest width the modulus carries
    /// injectively, which is narrower still — so both the assertion and the `expect` below are
    /// backstops on that rule rather than reachable refusals.
    #[opcode]
    fn cast_intn_to_field(bits: u64, #[out] res: *mut Field, #[frame_slice(bits)] a: &[u64]) {
        assert!(
            a.len() <= FELT_LIMBS,
            "an integer crosses the field boundary through {FELT_LIMBS} limbs, \
             so an int{bits} has no element"
        );
        let mut limbs = [0u64; FELT_LIMBS];
        limbs[..a.len()].copy_from_slice(a);
        unsafe {
            *res = <Field as ark_ff::PrimeField>::from_bigint(ark_ff::BigInt(limbs))
                .expect("a width the modulus carries injectively has an element for every value");
        }
    }

    #[opcode]
    fn array_alloc(
        #[out] res: *mut BoxedValue,
        stride: usize,
        meta: BoxedLayout,
        items: &[FramePosition],
        frame: Frame,
        vm: &mut VM,
    ) {
        let array = BoxedValue::alloc(meta, vm);
        // println!(
        //     "array_alloc: size={} stride={} has_ptr_elems={} @ {:?}",
        //     meta.size(),
        //     stride,
        //     meta.ptr_elems(),
        //     array.0
        // );
        for (i, item) in items.iter().enumerate() {
            let tgt = array.array_idx(i, stride);
            unsafe {
                frame.write_to(tgt, item.0 as isize, stride);
            }
        }
        // println!(
        //     "array_alloc: array={:?} stride={} size={} storage_size={}",
        //     array.0,
        //     stride,
        //     array.layout().array_size(),
        //     array.layout().underlying_array_size()
        // );
        unsafe {
            *res = array;
        }
    }

    #[opcode]
    fn array_alloc_from_frame(
        #[out] res: *mut BoxedValue,
        stride: usize,
        meta: BoxedLayout,
        count: usize,
        source: FramePosition,
        frame: Frame,
        vm: &mut VM,
    ) {
        let array = BoxedValue::alloc(meta, vm);
        unsafe {
            frame.write_to(array.data(), source.0 as isize, count * stride);
            *res = array;
        }
    }

    #[opcode]
    fn array_alloc_repeated(
        #[out] res: *mut BoxedValue,
        stride: usize,
        meta: BoxedLayout,
        count: usize,
        item: FramePosition,
        frame: Frame,
        vm: &mut VM,
    ) {
        let array = BoxedValue::alloc(meta, vm);
        for i in 0..count {
            let tgt = array.array_idx(i, stride);
            unsafe {
                frame.write_to(tgt, item.0 as isize, stride);
            }
        }
        unsafe {
            *res = array;
        }
    }

    #[opcode]
    #[inline(never)]
    fn tuple_alloc(
        #[out] res: *mut BoxedValue,
        meta: BoxedLayout,
        fields: &[FramePosition],
        frame: Frame,
        vm: &mut VM,
    ) {
        let tuple = BoxedValue::alloc(meta, vm);
        let view = meta.as_struct(&vm.struct_layouts);
        let mut field_offset = 0;
        for (i, field) in fields.iter().enumerate() {
            let size = view.field_size(i);
            let tgt = unsafe { tuple.data().add(field_offset) };
            unsafe {
                frame.write_to(tgt, field.0 as isize, size);
            }
            field_offset += size;
        }
        unsafe {
            *res = tuple;
        }
    }

    #[opcode]
    fn ref_alloc(#[out] res: *mut BoxedValue, meta: BoxedLayout, vm: &mut VM) {
        let cell = BoxedValue::alloc(meta, vm);
        unsafe {
            ptr::write_bytes(cell.data(), 0, meta.ref_cell_elem_size());
            *res = cell;
        }
    }

    #[opcode]
    #[inline(never)]
    fn ref_store(
        #[frame] cell: BoxedValue,
        source: FramePosition,
        stride: usize,
        elem_rc: usize,
        frame: Frame,
        vm: &mut VM,
    ) {
        if elem_rc != 0 {
            let old = unsafe { *(cell.data() as *mut BoxedValue) };
            if !old.0.is_null() {
                old.dec_rc(vm);
            }
        }
        unsafe {
            frame.write_to(cell.data(), source.0 as isize, stride);
        }
    }

    #[opcode]
    fn ref_load(#[out] res: *mut u64, #[frame] cell: BoxedValue, stride: usize) {
        unsafe {
            ptr::copy_nonoverlapping(cell.data(), res, stride);
        }
    }

    #[raw_opcode]
    fn array_get(
        pc: *const u64,
        frame: Frame,
        vm: &mut VM,
        #[out] res: *mut u64,
        #[frame] array: BoxedValue,
        #[frame] index: u64,
        stride: usize,
    ) -> (*const u64, Frame) {
        // Check the element index before multiplying by stride: a large index can wrap.
        if stride == 0 || index as usize >= array.layout().array_size() / stride {
            return trap(pc, frame, vm);
        }
        let src = array.array_idx(index as usize, stride);
        unsafe {
            ptr::copy_nonoverlapping(src, res, stride);
        }
        (unsafe { pc.add(5) }, frame)
    }

    /// Read an element out of a blob: a raw sequence of `len` elements of
    /// `stride` cells each, stored inline in the frame starting at `source`.
    #[raw_opcode]
    fn blob_get(
        pc: *const u64,
        frame: Frame,
        vm: &mut VM,
        #[out] res: *mut u64,
        source: FramePosition,
        #[frame] index: u64,
        stride: usize,
        len: usize,
    ) -> (*const u64, Frame) {
        if index as usize >= len {
            return trap(pc, frame, vm);
        }
        unsafe {
            frame.write_to(res, (source.0 + (index as usize) * stride) as isize, stride);
        }
        (unsafe { pc.add(6) }, frame)
    }

    #[opcode]
    fn tuple_proj(
        #[out] res: *mut u64,
        #[frame] tuple: BoxedValue,
        field_offset: usize,
        field_size: usize,
        vm: &mut VM,
    ) {
        let src = unsafe { tuple.data().add(field_offset) };
        unsafe {
            ptr::copy_nonoverlapping(src, res, field_size);
        }
    }

    #[raw_opcode]
    #[inline(never)]
    fn array_set(
        pc: *const u64,
        frame: Frame,
        vm: &mut VM,
        #[out] res: *mut BoxedValue,
        #[frame] array: BoxedValue,
        #[frame] index: u64,
        source: FramePosition,
        stride: usize,
    ) -> (*const u64, Frame) {
        if stride == 0 || index as usize >= array.layout().array_size() / stride {
            return trap(pc, frame, vm);
        }
        let new_array = array.copy_if_reused(vm);
        let target = new_array.array_idx(index as usize, stride);
        if new_array.layout().data_type() == DataType::BoxedArray {
            if new_array.0 == array.0 {
                // if we're reusing the array, the old element needs to be garbage collected
                let old_elem = unsafe { *(target as *mut BoxedValue) };
                old_elem.dec_rc(vm);
            } else {
                // if we're not reusing the array, we need to bump RC of all _other_ elements,
                // because they're now aliased in the new array.
                for i in 0..new_array.layout().array_size() {
                    if i != index as usize {
                        let elem = unsafe { *(new_array.array_idx(i, stride) as *mut BoxedValue) };
                        elem.inc_rc(1);
                    }
                }
            }
        }
        unsafe {
            frame.write_to(target, source.0 as isize, stride);
            *res = new_array;
        }
        (unsafe { pc.add(6) }, frame)
    }

    #[opcode]
    fn slice_len(#[out] res: *mut u64, #[frame] array: BoxedValue, stride: usize) {
        let len = array.layout().array_size() / stride;
        unsafe {
            *res = len as u64;
        }
    }

    #[opcode]
    fn slice_push(
        #[out] res: *mut BoxedValue,
        #[frame] slice: BoxedValue,
        stride: usize,
        is_push_front: usize,
        values: &[FramePosition],
        frame: Frame,
        vm: &mut VM,
    ) {
        let extra_space_needed = values.len() * stride;
        let new_array = slice.alloc_grown_slice(extra_space_needed, is_push_front, vm);
        let pushed_data_offset = if is_push_front != 0 {
            0
        } else {
            new_array.layout().array_size() - extra_space_needed
        };
        for (i, item) in values.iter().enumerate() {
            let tgt = unsafe { new_array.data().add(pushed_data_offset + i * stride) };
            unsafe {
                frame.write_to(tgt, item.0 as isize, stride);
            }
        }
        unsafe {
            *res = new_array;
        }
    }

    #[opcode]
    fn inc_rc(#[frame] array: BoxedValue, amount: u64) {
        // println!("inc_array_rc_intro");
        array.inc_rc(amount);
        // println!("inc_array_rc_outro");
    }

    #[opcode]
    #[inline(never)]
    fn dec_rc(#[frame] array: BoxedValue, vm: &mut VM) {
        // println!("dec_array_rc_intro");
        array.dec_rc(vm);
        // println!("dec_array_rc_outro");
    }

    #[opcode]
    fn witness_ref_alloc(#[out] res: *mut BoxedValue, data: Field, vm: &mut VM) {
        let val = BoxedValue::alloc(BoxedLayout::ad_const(), vm);
        let d = val.as_ad_const();
        unsafe {
            (*d).value = data;
            *res = val;
        };
    }

    #[opcode]
    fn bump_da(#[frame] v: BoxedValue, #[frame] coeff: Field, vm: &mut VM) {
        v.bump_da(coeff, vm);
    }

    #[opcode]
    fn bump_db(#[frame] v: BoxedValue, #[frame] coeff: Field, vm: &mut VM) {
        v.bump_db(coeff, vm);
    }

    #[opcode]
    fn bump_dc(#[frame] v: BoxedValue, #[frame] coeff: Field, vm: &mut VM) {
        v.bump_dc(coeff, vm);
    }

    #[opcode]
    fn next_d_coeff(#[out] v: *mut Field, vm: &mut VM) {
        unsafe {
            *v = *vm.data.as_ad.ad_coeffs.add(vm.data.as_ad.current_cnst_off);
            vm.data.as_ad.current_cnst_off += 1;
        };
    }

    #[opcode]
    fn fresh_witness(#[out] res: *mut BoxedValue, vm: &mut VM) {
        let index = unsafe { vm.data.as_ad.current_wit_off as u64 };
        unsafe { vm.data.as_ad.current_wit_off += 1 };
        let val = BoxedValue::alloc(BoxedLayout::ad_witness(), vm);
        let d = val.as_ad_witness();
        unsafe {
            (*d).index = index;
            *res = val;
        }
    }

    #[opcode]
    fn pure_to_witness_ref(#[out] res: *mut BoxedValue, #[frame] v: Field, vm: &mut VM) {
        let val = BoxedValue::alloc(BoxedLayout::ad_const(), vm);
        let d = val.as_ad_const();
        unsafe {
            (*d).value = v;
            *res = val;
        }
    }

    #[opcode]
    fn unbox_field(#[out] res: *mut Field, #[frame] v: BoxedValue) {
        let d = v.as_ad_const();
        unsafe {
            *res = (*d).value;
        }
    }

    #[opcode]
    fn mul_const(
        #[out] res: *mut BoxedValue,
        #[frame] coeff: Field,
        #[frame] v: BoxedValue,
        vm: &mut VM,
    ) {
        let val = BoxedValue::alloc(BoxedLayout::mul_const(), vm);
        let d = val.as_mul_const();
        unsafe {
            (*d).coeff = coeff;
            (*d).value = v;
            (*d).da = Field::ZERO;
            (*d).db = Field::ZERO;
            (*d).dc = Field::ZERO;
            *res = val;
        }
    }

    #[opcode]
    fn add_boxed(
        #[out] res: *mut BoxedValue,
        #[frame] a: BoxedValue,
        #[frame] b: BoxedValue,
        vm: &mut VM,
    ) {
        let val = BoxedValue::alloc(BoxedLayout::ad_sum(), vm);
        let d = val.as_ad_sum();
        unsafe {
            (*d).a = a;
            (*d).b = b;
            (*d).da = Field::ZERO;
            (*d).db = Field::ZERO;
            (*d).dc = Field::ZERO;
            *res = val;
        }
    }

    #[raw_opcode]
    fn assert_eq_int(
        pc: *const u64,
        frame: Frame,
        vm: &mut VM,
        #[frame] a: u64,
        #[frame] b: u64,
    ) -> (*const u64, Frame) {
        if a != b {
            return trap(pc, frame, vm);
        }
        (unsafe { pc.offset(3) }, frame)
    }

    #[raw_opcode]
    fn assert_eq_int128(
        pc: *const u64,
        frame: Frame,
        vm: &mut VM,
        #[frame] a: Int128,
        #[frame] b: Int128,
    ) -> (*const u64, Frame) {
        if a != b {
            return trap(pc, frame, vm);
        }
        (unsafe { pc.offset(3) }, frame)
    }

    /// Assert two wide integers are equal, trapping if they are not.
    ///
    /// Raw, as the other assert-family opcodes are: a failing check has to return a null `pc` to
    /// halt dispatch rather than fall through to the next instruction.
    #[raw_opcode]
    fn assert_eq_intn(
        pc: *const u64,
        frame: Frame,
        vm: &mut VM,
        bits: u64,
        #[frame_slice(bits)] a: &[u64],
        #[frame_slice(bits)] b: &[u64],
    ) -> (*const u64, Frame) {
        if !int_limbs::eq(a, b) {
            return trap(pc, frame, vm);
        }
        (unsafe { pc.offset(4) }, frame)
    }

    #[raw_opcode]
    fn assert_eq_field(
        pc: *const u64,
        frame: Frame,
        vm: &mut VM,
        #[frame] a: Field,
        #[frame] b: Field,
    ) -> (*const u64, Frame) {
        if a != b {
            return trap(pc, frame, vm);
        }
        (unsafe { pc.offset(3) }, frame)
    }

    #[raw_opcode]
    fn assert_r1c(
        pc: *const u64,
        frame: Frame,
        vm: &mut VM,
        #[frame] a: Field,
        #[frame] b: Field,
        #[frame] c: Field,
    ) -> (*const u64, Frame) {
        if a * b != c {
            return trap(pc, frame, vm);
        }
        (unsafe { pc.offset(4) }, frame)
    }

    #[raw_opcode]
    #[inline(never)] // TODO better impl
    fn rangecheck(
        pc: *const u64,
        frame: Frame,
        vm: &mut VM,
        #[frame] val: Field,
        max_bits: usize,
    ) -> (*const u64, Frame) {
        // Convert field to bigint and check if it fits in max_bits
        // FIELD-ASSUMPTION: L4-decompose
        let bigint = ark_ff::PrimeField::into_bigint(val);
        let check = bigint.to_bits_le().iter().skip(max_bits).all(|b| !b);
        if !check {
            return trap(pc, frame, vm);
        }
        (unsafe { pc.offset(3) }, frame)
    }

    #[opcode]
    fn to_bytes_be(#[frame] val: Field, count: u64, #[out] res: *mut BoxedValue, vm: &mut VM) {
        let val = ark_ff::PrimeField::into_bigint(val);
        let r = BoxedValue::alloc(BoxedLayout::array(count as usize, false), vm);
        unsafe {
            for i in 0..count {
                // Each limb in val.0 is a u64 (8 bytes), little-endian limb order
                let byte_idx = i as usize; // byte index from LSB
                let limb_idx = byte_idx / 8;
                let byte_in_limb = byte_idx % 8;
                let byte_val = if limb_idx < val.0.len() {
                    (val.0[limb_idx] >> (byte_in_limb * 8)) & 0xFF
                } else {
                    0
                };
                *r.array_idx((count - i - 1) as usize, 1) = byte_val;
            }
            *res = r;
        }
    }

    #[opcode]
    fn to_bytes_le(#[frame] val: Field, count: u64, #[out] res: *mut BoxedValue, vm: &mut VM) {
        let val = ark_ff::PrimeField::into_bigint(val);
        let r = BoxedValue::alloc(BoxedLayout::array(count as usize, false), vm);
        unsafe {
            for i in 0..count {
                // Each limb in val.0 is a u64 (8 bytes), little-endian limb order.
                let byte_idx = i as usize;
                let limb_idx = byte_idx / 8;
                let byte_in_limb = byte_idx % 8;
                let byte_val = if limb_idx < val.0.len() {
                    (val.0[limb_idx] >> (byte_in_limb * 8)) & 0xFF
                } else {
                    0
                };
                *r.array_idx(i as usize, 1) = byte_val;
            }
            *res = r;
        }
    }

    #[opcode]
    fn to_bits_le(#[out] res: *mut BoxedValue, #[frame] val: Field, count: u64, vm: &mut VM) {
        let val = ark_ff::PrimeField::into_bigint(val);
        let r = BoxedValue::alloc(BoxedLayout::array(count as usize, false), vm);
        unsafe {
            for i in 0..count {
                let bit_idx = i as usize;
                let limb_idx = bit_idx / 64;
                let bit_in_limb = bit_idx % 64;
                let bit = if limb_idx < val.0.len() {
                    (val.0[limb_idx] >> bit_in_limb) & 1
                } else {
                    0
                };
                *r.array_idx(bit_idx, 1) = bit;
            }
            *res = r;
        }
    }

    #[opcode]
    fn to_bits_be(#[out] res: *mut BoxedValue, #[frame] val: Field, count: u64, vm: &mut VM) {
        let val = ark_ff::PrimeField::into_bigint(val);
        let r = BoxedValue::alloc(BoxedLayout::array(count as usize, false), vm);
        unsafe {
            for i in 0..count {
                let bit_idx = i as usize;
                let limb_idx = bit_idx / 64;
                let bit_in_limb = bit_idx % 64;
                let bit = if limb_idx < val.0.len() {
                    (val.0[limb_idx] >> bit_in_limb) & 1
                } else {
                    0
                };
                *r.array_idx((count - i - 1) as usize, 1) = bit;
            }
            *res = r;
        }
    }

    /// Interleave the low 32 bits of `val` with zeros, giving a 64-bit spread pattern.
    #[opcode]
    fn spread_u32_to_u64(#[out] res: *mut u64, #[frame] val: u64) {
        let result = spread_bits(val as u32);
        unsafe {
            *res = result;
        }
    }

    /// Split a 64-bit spread pattern into its two 32-bit bit streams, odd-indexed and even.
    #[opcode]
    fn unspread_u64_to_u32(#[out] res_and: *mut u64, #[out] res_xor: *mut u64, #[frame] val: u64) {
        let (and_val, xor_val) = unspread_bits(val);
        unsafe {
            *res_and = and_val as u64;
            *res_xor = xor_val as u64;
        }
    }

    #[raw_opcode]
    fn spread_lookup_field(
        pc: *const u64,
        frame: Frame,
        vm: &mut VM,
        #[frame] val: Field,
        #[frame] result: Field,
        #[frame] flag: Field,
        bits: usize,
    ) -> (*const u64, Frame) {
        // Initialize spread table for this bit-width on first call.
        //
        // Spread tables use the folded single-constraint allocation (`TableKind::Spread`): both
        // operands of each entry (key=i, value=spread(i)) are constants, so each entry is just one
        // `y·(α-i+β·spread(i))=m` constraint/witness instead of the generic two-constraint
        // key-value form. Phase 2 recomputes `spread(i)` itself, so there is nothing to dump here.
        if vm.spread_tables[bits].is_none() {
            let length = 1usize << bits;
            let table_info = TableInfo {
                multiplicities_wit: unsafe { vm.data.as_forward.multiplicities_witness },
                num_indices: 1,
                kind: TableKind::Spread,
                length,
                elem_inverses_constraint_section_offset: unsafe {
                    vm.data.as_forward.elem_inverses_constraint_section_offset
                },
                elem_inverses_witness_section_offset: unsafe {
                    vm.data.as_forward.elem_inverses_witness_section_offset
                },
            };
            vm.spread_tables[bits] = Some(vm.tables.len());
            vm.tables.push(table_info);

            unsafe {
                vm.data.as_forward.multiplicities_witness =
                    vm.data.as_forward.multiplicities_witness.add(length);
                // One constraint per entry + one sum constraint; one witness per entry.
                vm.data.as_forward.elem_inverses_constraint_section_offset += length + 1;
                vm.data.as_forward.elem_inverses_witness_section_offset += length;
            }
        }

        let table_idx = vm.spread_tables[bits].unwrap();
        let flag_u64 = ark_ff::PrimeField::into_bigint(flag).0[0];
        // The table's keys are `0..2^bits`, so a wider `val` has nothing to look up.
        if !unsafe { forward_kv_lookup_emit(table_idx, val, result, flag_u64, vm) } {
            return trap(pc, frame, vm);
        }

        (unsafe { pc.offset(5) }, frame)
    }

    #[opcode]
    fn dspread_lookup_field(
        #[frame] val: BoxedValue,
        #[frame] result: BoxedValue,
        #[frame] flag: BoxedValue,
        bits: usize,
        vm: &mut VM,
    ) {
        if vm.spread_tables[bits].is_none() {
            let length = 1usize << bits;
            let inverses_constraint_section_offset =
                unsafe { vm.data.as_ad.current_cnst_tables_off };
            let inverses_witness_section_offset = unsafe { vm.data.as_ad.current_wit_tables_off };
            let multiplicities_wit_offset = unsafe { vm.data.as_ad.current_wit_multiplicities_off };
            let table_info = TableInfo {
                multiplicities_wit: ptr::null_mut(),
                num_indices: 1,
                kind: TableKind::Spread,
                length,
                elem_inverses_witness_section_offset: inverses_witness_section_offset,
                elem_inverses_constraint_section_offset: inverses_constraint_section_offset,
            };
            vm.spread_tables[bits] = Some(vm.tables.len());
            vm.tables.push(table_info);
            unsafe {
                // Folded allocation: one constraint per entry + one sum
                // constraint; one witness per entry.
                vm.data.as_ad.current_wit_multiplicities_off += length;
                vm.data.as_ad.current_wit_tables_off += length;
                vm.data.as_ad.current_cnst_tables_off += length + 1;
            }

            let inv_sum_coeff = unsafe {
                *vm.data
                    .as_ad
                    .ad_coeffs
                    .offset(inverses_constraint_section_offset as isize + length as isize)
            };

            for i in 0..length {
                // Single folded constraint: y · (α - i + β·spread(i)) - m = 0
                //   A = (y), B = (α) + (w0, -i) + (β, spread(i)), C = (m)
                let coeff = unsafe {
                    *vm.data
                        .as_ad
                        .ad_coeffs
                        .offset(inverses_constraint_section_offset as isize + i as isize)
                };
                unsafe {
                    // da[y_wit] += coeff
                    *vm.data
                        .as_ad
                        .out_da
                        .offset(inverses_witness_section_offset as isize + i as isize) += coeff;

                    // db[α] += coeff
                    *vm.data
                        .as_ad
                        .out_db
                        .add(vm.data.as_ad.logup_wit_challenge_off) += coeff;
                    // db[w0] -= coeff * i
                    *vm.data.as_ad.out_db -= coeff * Field::from(i as u64);
                    // db[β] += coeff * spread(i)
                    *vm.data
                        .as_ad
                        .out_db
                        .offset(vm.data.as_ad.logup_wit_challenge_off as isize + 1) +=
                        coeff * Field::from(spread_bits(i as u32));

                    // dc[m] += coeff
                    *vm.data
                        .as_ad
                        .out_dc
                        .offset(multiplicities_wit_offset as isize + i as isize) += coeff;

                    // Sum: inv goes into A position
                    *vm.data
                        .as_ad
                        .out_da
                        .offset(inverses_witness_section_offset as isize + i as isize) +=
                        inv_sum_coeff;
                }
            }

            unsafe {
                *vm.data.as_ad.out_db += inv_sum_coeff;
            }
        }

        let table_idx = vm.spread_tables[bits].unwrap();
        unsafe { ad_kv_lookup_emit(table_idx, val, result, flag, vm) };
    }

    /// Prove `factor == 2^amount` with `amount` below the shifted operand's width, by bumping
    /// the matching row's multiplicity in the powers-of-two table.
    ///
    /// `size` is `log2` of that width, so the table has `1 << size == bits` rows keyed by the
    /// legal amounts. An amount at or past the width -- including a negative one, whose raw
    /// encoding is far larger -- has no row and traps, which is how a witness-amount shift gets
    /// the rejection that `shift_guard` builds out of a comparison on the pure path.
    #[raw_opcode]
    fn pow2_lookup_field(
        pc: *const u64,
        frame: Frame,
        vm: &mut VM,
        #[frame] amount: Field,
        #[frame] factor: Field,
        #[frame] flag: Field,
        size: usize,
    ) -> (*const u64, Frame) {
        // Initialize the powers-of-two table for this width on first call. Laid out exactly like
        // a spread table -- both operands of every entry are constants, so each entry folds to
        // one constraint/witness. Phase 2 recomputes 2^n from the row index, so nothing is
        // dumped here.
        if vm.pow2_tables[size].is_none() {
            let length = 1usize << size;
            let table_info = TableInfo {
                multiplicities_wit: unsafe { vm.data.as_forward.multiplicities_witness },
                num_indices: 1,
                kind: TableKind::Pow2,
                length,
                elem_inverses_constraint_section_offset: unsafe {
                    vm.data.as_forward.elem_inverses_constraint_section_offset
                },
                elem_inverses_witness_section_offset: unsafe {
                    vm.data.as_forward.elem_inverses_witness_section_offset
                },
            };
            vm.pow2_tables[size] = Some(vm.tables.len());
            vm.tables.push(table_info);

            unsafe {
                vm.data.as_forward.multiplicities_witness =
                    vm.data.as_forward.multiplicities_witness.add(length);
                vm.data.as_forward.elem_inverses_constraint_section_offset += length + 1;
                vm.data.as_forward.elem_inverses_witness_section_offset += length;
            }
        }

        let table_idx = vm.pow2_tables[size].unwrap();
        let flag_u64 = ark_ff::PrimeField::into_bigint(flag).0[0];
        // The table's keys are `0..2^size`, so an out-of-range amount has nothing to look up.
        if !unsafe { forward_kv_lookup_emit(table_idx, amount, factor, flag_u64, vm) } {
            return trap(pc, frame, vm);
        }

        (unsafe { pc.offset(5) }, frame)
    }

    /// The AD twin of [`Self::pow2_lookup_field`].
    #[opcode]
    fn dpow2_lookup_field(
        #[frame] amount: BoxedValue,
        #[frame] factor: BoxedValue,
        #[frame] flag: BoxedValue,
        size: usize,
        vm: &mut VM,
    ) {
        if vm.pow2_tables[size].is_none() {
            let length = 1usize << size;
            let inverses_constraint_section_offset =
                unsafe { vm.data.as_ad.current_cnst_tables_off };
            let inverses_witness_section_offset = unsafe { vm.data.as_ad.current_wit_tables_off };
            let multiplicities_wit_offset = unsafe { vm.data.as_ad.current_wit_multiplicities_off };
            let table_info = TableInfo {
                multiplicities_wit: ptr::null_mut(),
                num_indices: 1,
                kind: TableKind::Pow2,
                length,
                elem_inverses_witness_section_offset: inverses_witness_section_offset,
                elem_inverses_constraint_section_offset: inverses_constraint_section_offset,
            };
            vm.pow2_tables[size] = Some(vm.tables.len());
            vm.tables.push(table_info);
            unsafe {
                // Folded allocation: one constraint per entry + one sum
                // constraint; one witness per entry.
                vm.data.as_ad.current_wit_multiplicities_off += length;
                vm.data.as_ad.current_wit_tables_off += length;
                vm.data.as_ad.current_cnst_tables_off += length + 1;
            }

            let inv_sum_coeff = unsafe {
                *vm.data
                    .as_ad
                    .ad_coeffs
                    .offset(inverses_constraint_section_offset as isize + length as isize)
            };

            for (i, pow) in pow2_rows(length).enumerate() {
                // Single folded constraint: y · (α - i + β·2^i) - m = 0
                //   A = (y), B = (α) + (w0, -i) + (β, 2^i), C = (m)
                let coeff = unsafe {
                    *vm.data
                        .as_ad
                        .ad_coeffs
                        .offset(inverses_constraint_section_offset as isize + i as isize)
                };
                unsafe {
                    // da[y_wit] += coeff
                    *vm.data
                        .as_ad
                        .out_da
                        .offset(inverses_witness_section_offset as isize + i as isize) += coeff;

                    // db[α] += coeff
                    *vm.data
                        .as_ad
                        .out_db
                        .add(vm.data.as_ad.logup_wit_challenge_off) += coeff;
                    // db[w0] -= coeff * i
                    *vm.data.as_ad.out_db -= coeff * Field::from(i as u64);
                    // db[β] += coeff * 2^i
                    *vm.data
                        .as_ad
                        .out_db
                        .offset(vm.data.as_ad.logup_wit_challenge_off as isize + 1) += coeff * pow;

                    // dc[m] += coeff
                    *vm.data
                        .as_ad
                        .out_dc
                        .offset(multiplicities_wit_offset as isize + i as isize) += coeff;

                    // Sum: inv goes into A position
                    *vm.data
                        .as_ad
                        .out_da
                        .offset(inverses_witness_section_offset as isize + i as isize) +=
                        inv_sum_coeff;
                }
            }

            unsafe {
                *vm.data.as_ad.out_db += inv_sum_coeff;
            }
        }

        let table_idx = vm.pow2_tables[size].unwrap();
        unsafe { ad_kv_lookup_emit(table_idx, amount, factor, flag, vm) };
    }

    /// The lookup-argument form of [`Self::rangecheck`]: instead of decomposing
    /// `val` inline, prove `val < 2^bits` by bumping its multiplicity in the
    /// `bits`-wide rangecheck table. A value with no row in that table fails the
    /// check, so this traps for exactly the inputs `rangecheck` traps on.
    #[raw_opcode]
    fn rngchk_field(
        pc: *const u64,
        frame: Frame,
        vm: &mut VM,
        #[frame] val: Field,
        #[frame] flag: Field,
        bits: usize,
    ) -> (*const u64, Frame) {
        if vm.rgchk_tables[bits].is_none() {
            let length = 1usize << bits;
            let table_info = TableInfo {
                multiplicities_wit: unsafe { vm.data.as_forward.multiplicities_witness },
                num_indices: 1,
                kind: TableKind::RangeCheck,
                length,
                elem_inverses_constraint_section_offset: unsafe {
                    vm.data.as_forward.elem_inverses_constraint_section_offset
                },
                elem_inverses_witness_section_offset: unsafe {
                    vm.data.as_forward.elem_inverses_witness_section_offset
                },
            };
            vm.rgchk_tables[bits] = Some(vm.tables.len());
            vm.tables.push(table_info);
            unsafe {
                vm.data.as_forward.multiplicities_witness =
                    vm.data.as_forward.multiplicities_witness.add(length);
                vm.data.as_forward.elem_inverses_constraint_section_offset += length + 1;
                vm.data.as_forward.elem_inverses_witness_section_offset += length;
            }
        }
        let flag_u64 = ark_ff::PrimeField::into_bigint(flag).0[0];
        let table_idx = vm.rgchk_tables[bits].unwrap();
        let table_info = &vm.tables[table_idx];

        // An inactive check bumps no multiplicity, so it stays emittable whatever
        // `val` is: a guarded-off rangecheck must not be able to fail.
        let val_row = if flag_u64 != 0 {
            match table_row_index(val, table_info.length) {
                Some(row) => Some(row),
                None => return trap(pc, frame, vm),
            }
        } else {
            None
        };

        unsafe {
            if let Some(val_u64) = val_row {
                let ptr = table_info.multiplicities_wit.offset(val_u64 as isize);
                *(ptr as *mut u64) += flag_u64;
                *(vm.data.as_forward.lookups_a as *mut u64) = table_idx as u64;
                vm.data.as_forward.lookups_a = vm.data.as_forward.lookups_a.offset(1);
                *(vm.data.as_forward.lookups_b as *mut u64) = val_u64;
                vm.data.as_forward.lookups_b = vm.data.as_forward.lookups_b.offset(1);
            } else {
                *(vm.data.as_forward.lookups_a as *mut u64) = table_idx as u64;
                vm.data.as_forward.lookups_a = vm.data.as_forward.lookups_a.offset(1);
                *(vm.data.as_forward.lookups_b as *mut Field) = val;
                vm.data.as_forward.lookups_b = vm.data.as_forward.lookups_b.offset(1);
            }
            *(vm.data.as_forward.lookups_c as *mut u64) = flag_u64;
            vm.data.as_forward.lookups_c = vm.data.as_forward.lookups_c.offset(1);
        }

        (unsafe { pc.offset(4) }, frame)
    }

    #[raw_opcode]
    fn array_lookup_field(
        pc: *const u64,
        frame: Frame,
        vm: &mut VM,
        #[frame] array: BoxedValue,
        #[frame] index: Field,
        #[frame] result: Field,
        #[frame] flag: Field,
        stride: usize,
        elem_kind: usize,
    ) -> (*const u64, Frame) {
        let table_id_ptr = array.table_id();
        let table_idx = unsafe { *table_id_ptr };

        let table_idx = if table_idx == u64::MAX {
            // First lookup on this array: create a new table
            let (cnst_off, wit_off, mult_wit) = unsafe {
                (
                    vm.data.as_forward.elem_inverses_constraint_section_offset,
                    vm.data.as_forward.elem_inverses_witness_section_offset,
                    vm.data.as_forward.multiplicities_witness,
                )
            };

            // Dump array element values into the x-slots (even offsets) of the table section
            let length = unsafe {
                for_each_array_leaf(array, stride, |i, elem_ptr| {
                    let elem_field = read_pure_elem_as_field(elem_ptr, elem_kind);
                    // Write it into the x-slot (even offset: 2*i) of the constraint section
                    *vm.data.as_forward.out_a_base.add(cnst_off + 2 * i) = elem_field;
                })
            };

            let table_info = TableInfo {
                multiplicities_wit: mult_wit,
                num_indices: 1,
                kind: TableKind::Array,
                length,
                elem_inverses_constraint_section_offset: cnst_off,
                elem_inverses_witness_section_offset: wit_off,
            };
            let new_table_idx = vm.tables.len();
            vm.tables.push(table_info);

            unsafe {
                vm.data.as_forward.multiplicities_witness = mult_wit.add(length);
                // 2 constraints per element + 1 sum constraint
                vm.data.as_forward.elem_inverses_constraint_section_offset += 2 * length + 1;
                // 2 witness slots per element
                vm.data.as_forward.elem_inverses_witness_section_offset += 2 * length;
            }

            // Store table index on the array
            unsafe { *table_id_ptr = new_table_idx as u64 };

            new_table_idx
        } else {
            table_idx as usize
        };

        let flag_u64 = ark_ff::PrimeField::into_bigint(flag).0[0];
        // The table's keys are the element indices, so an out-of-bounds `index`
        // has nothing to look up — the bounds check the compiler emits alongside
        // this read is what normally rules that out.
        if !unsafe { forward_kv_lookup_emit(table_idx, index, result, flag_u64, vm) } {
            return trap(pc, frame, vm);
        }

        (unsafe { pc.offset(7) }, frame)
    }

    #[opcode]
    fn drngchk_field(
        #[frame] val: BoxedValue,
        #[frame] flag: BoxedValue,
        bits: usize,
        vm: &mut VM,
    ) {
        let length = 1usize << bits;
        if vm.rgchk_tables[bits].is_none() {
            let inverses_constraint_section_offset =
                unsafe { vm.data.as_ad.current_cnst_tables_off };
            let inverses_witness_section_offset = unsafe { vm.data.as_ad.current_wit_tables_off };
            let multiplicities_wit_offset = unsafe { vm.data.as_ad.current_wit_multiplicities_off };
            let table_info = TableInfo {
                multiplicities_wit: ptr::null_mut(),
                num_indices: 1,
                kind: TableKind::RangeCheck,
                length,
                elem_inverses_witness_section_offset: inverses_witness_section_offset,
                elem_inverses_constraint_section_offset: inverses_constraint_section_offset,
            };
            vm.rgchk_tables[bits] = Some(vm.tables.len());
            vm.tables.push(table_info);
            unsafe {
                vm.data.as_ad.current_wit_multiplicities_off += length;
                vm.data.as_ad.current_wit_tables_off += length;
                vm.data.as_ad.current_cnst_tables_off += length + 1;
            }
            let inv_sum_coeff = unsafe {
                *vm.data
                    .as_ad
                    .ad_coeffs
                    .offset(inverses_constraint_section_offset as isize + length as isize)
            };

            for i in 0..length as isize {
                // For each element in the table, we have constraint `elem_inv_witness * (alpha - i) - multiplicity_witness = 0`
                let coeff = unsafe {
                    *vm.data
                        .as_ad
                        .ad_coeffs
                        .offset(inverses_constraint_section_offset as isize + i)
                };
                unsafe {
                    *vm.data
                        .as_ad
                        .out_da
                        .offset(inverses_witness_section_offset as isize + i) += coeff;
                    // if i == 0 {
                    //     println!("bump da at {} from inv by {coeff}", inverses_witness_section_offset as isize + i);
                    // }

                    *vm.data
                        .as_ad
                        .out_db
                        .add(vm.data.as_ad.logup_wit_challenge_off) += coeff;
                    *vm.data.as_ad.out_db -= coeff * Field::from(i as u64);

                    *vm.data
                        .as_ad
                        .out_dc
                        .offset(multiplicities_wit_offset as isize + i) += coeff;
                }

                // Also each inv goes into the A position of the total sum
                unsafe {
                    *vm.data
                        .as_ad
                        .out_da
                        .offset(inverses_witness_section_offset as isize + i) += inv_sum_coeff;
                }
            }

            // The coeff at B on the sum constraint is just `1` so we bump it.
            unsafe {
                *vm.data.as_ad.out_db += inv_sum_coeff;
            }
        }
        let table_idx = vm.rgchk_tables[bits].unwrap();
        let table_info = &vm.tables[table_idx];

        let inv_coeff = unsafe {
            let r = *vm
                .data
                .as_ad
                .ad_coeffs
                .add(vm.data.as_ad.current_cnst_lookups_off);
            vm.data.as_ad.current_cnst_lookups_off += 1;
            r
        };

        let inv_sum_coeff = unsafe {
            *vm.data
                .as_ad
                .ad_coeffs
                .add(table_info.elem_inverses_constraint_section_offset + length)
        };

        let current_inv_wit_offset = unsafe {
            let r = vm.data.as_ad.current_wit_lookups_off;
            vm.data.as_ad.current_wit_lookups_off += 1;
            r
        };

        unsafe {
            // bump for the RHS of the sum
            *vm.data.as_ad.out_dc.add(current_inv_wit_offset) += inv_sum_coeff;

            // bumps for the inversion assert: y*(α-key) = flag
            // da[y] += inv_coeff
            *vm.data.as_ad.out_da.add(current_inv_wit_offset) += inv_coeff;

            // db[α] += inv_coeff
            *vm.data
                .as_ad
                .out_db
                .add(vm.data.as_ad.logup_wit_challenge_off) += inv_coeff;
            // db[key] -= inv_coeff
            val.bump_db(-inv_coeff, vm);

            // dc[flag] += inv_coeff  (RHS is flag, not constant 1)
            flag.bump_dc(inv_coeff, vm);
        }
    }

    #[opcode]
    fn darray_lookup_field(
        #[frame] array: BoxedValue,
        #[frame] index: BoxedValue,
        #[frame] result: BoxedValue,
        #[frame] flag: BoxedValue,
        stride: usize,
        elem_kind: usize,
        vm: &mut VM,
    ) {
        let table_id_ptr = array.table_id();
        let table_idx = unsafe { *table_id_ptr };

        let table_idx = if table_idx == u64::MAX {
            // First AD call on this array: create table and process table constraints
            let inverses_constraint_section_offset =
                unsafe { vm.data.as_ad.current_cnst_tables_off };
            let inverses_witness_section_offset = unsafe { vm.data.as_ad.current_wit_tables_off };
            let multiplicities_wit_offset = unsafe { vm.data.as_ad.current_wit_multiplicities_off };

            let length =
                unsafe {
                    for_each_array_leaf(array, stride, |i, elem_ptr| {
                        // x-constraint at base + 2*i: A=[(beta,1)], B=v_i, C=[(x,-1)]
                        let x_coeff =
                            *vm.data.as_ad.ad_coeffs.offset(
                                inverses_constraint_section_offset as isize + 2 * i as isize,
                            );
                        // da[beta] += x_coeff (A entry: (beta, 1))
                        *vm.data
                            .as_ad
                            .out_da
                            .offset(vm.data.as_ad.logup_wit_challenge_off as isize + 1) += x_coeff;
                        // db[v_i] += x_coeff (B entry: element value)
                        lookup_elem_bump_db(elem_ptr, elem_kind, x_coeff, vm);
                        // dc[x_wit] -= x_coeff (C entry: (x, -1))
                        *vm.data
                            .as_ad
                            .out_dc
                            .offset(inverses_witness_section_offset as isize + 2 * i as isize) -=
                            x_coeff;

                        // y-constraint at base + 2*i + 1: A=y_i, B=(alpha - i - x_i), C=mult_i
                        let y_coeff = *vm.data.as_ad.ad_coeffs.offset(
                            inverses_constraint_section_offset as isize + 2 * i as isize + 1,
                        );
                        // dA[y_witness] += y_coeff
                        *vm.data.as_ad.out_da.offset(
                            inverses_witness_section_offset as isize + 2 * i as isize + 1,
                        ) += y_coeff;
                        // dB[alpha] += y_coeff
                        *vm.data
                            .as_ad
                            .out_db
                            .add(vm.data.as_ad.logup_wit_challenge_off) += y_coeff;
                        // dB -= y_coeff * i (constant part)
                        *vm.data.as_ad.out_db -= y_coeff * Field::from(i as u64);
                        // dB[x_witness] -= y_coeff (x_i appears negated in B)
                        *vm.data
                            .as_ad
                            .out_db
                            .add(inverses_witness_section_offset + 2 * i) -= y_coeff;
                        // dC[mult_witness] += y_coeff
                        *vm.data.as_ad.out_dc.add(multiplicities_wit_offset + i) += y_coeff;
                    })
                };

            let sum_coeff = unsafe {
                *vm.data
                    .as_ad
                    .ad_coeffs
                    .offset(inverses_constraint_section_offset as isize + 2 * length as isize)
            };

            // Sum constraint: y_i goes into A position
            for i in 0..length {
                unsafe {
                    *vm.data
                        .as_ad
                        .out_da
                        .add(inverses_witness_section_offset + 2 * i + 1) += sum_coeff;
                }
            }

            // Sum constraint B=1: bump out_db by sum_coeff
            unsafe {
                *vm.data.as_ad.out_db += sum_coeff;
            }

            let table_info = TableInfo {
                multiplicities_wit: ptr::null_mut(),
                num_indices: 1,
                kind: TableKind::Array,
                length,
                elem_inverses_witness_section_offset: inverses_witness_section_offset,
                elem_inverses_constraint_section_offset: inverses_constraint_section_offset,
            };
            let new_table_idx = vm.tables.len();
            vm.tables.push(table_info);
            unsafe {
                vm.data.as_ad.current_wit_multiplicities_off += length;
                vm.data.as_ad.current_wit_tables_off += 2 * length;
                vm.data.as_ad.current_cnst_tables_off += 2 * length + 1;
            }

            unsafe { *table_id_ptr = new_table_idx as u64 };
            new_table_idx
        } else {
            table_idx as usize
        };

        unsafe { ad_kv_lookup_emit(table_idx, index, result, flag, vm) };
    }

    #[opcode]
    fn init_global(
        vm: &mut VM,
        frame: Frame,
        src: FramePosition,
        global_offset: usize,
        size: usize,
    ) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                frame.data.add(src.0),
                vm.globals.add(global_offset),
                size,
            );
        }
    }

    #[opcode]
    fn read_global(#[out] res: *mut u64, vm: &mut VM, global_offset: usize, size: usize) {
        unsafe {
            std::ptr::copy_nonoverlapping(vm.globals.add(global_offset), res, size);
        }
    }

    #[opcode]
    #[inline(never)]
    fn drop_global(vm: &mut VM, global_offset: usize) {
        unsafe {
            let boxed = *(vm.globals.add(global_offset) as *mut BoxedValue);
            boxed.dec_rc(vm);
        }
    }

    #[opcode]
    fn mov_const_pool(#[out] res: *mut u64, vm: &mut VM, pool_offset: usize, size: usize) {
        unsafe {
            std::ptr::copy_nonoverlapping(vm.constants.as_ptr().add(pool_offset), res, size);
        }
    }
}

pub struct Function {
    pub name: String,
    pub frame_size: usize,
    pub code: Vec<OpCode>,
    /// One location per opcode in `code`.
    pub source_locations: Vec<SourceLocation>,
}

impl Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "fn {} (frame_size = {}):", self.name, self.frame_size)?;
        for op in &self.code {
            writeln!(f, "  {}", op)?;
        }
        Ok(())
    }
}

/// Index of the witness-generation entry point in a program's entry table.
pub const ENTRY_WITGEN: usize = 0;
/// Index of the AD entry point in a program's entry table.
pub const ENTRY_AD: usize = 1;

pub struct Program {
    pub functions: Vec<Function>,
    /// Indices into `functions` of the program's entry points, in entry-table order
    /// ([`ENTRY_WITGEN`], [`ENTRY_AD`], ...).
    pub entry_points: Vec<usize>,
    /// Flattened field count of the witgen entry blob
    pub entry_blob_field_count: usize,
    pub global_frame_size: usize,
    pub struct_layouts: Vec<StructDescriptor>,
    /// Interned constant pool: a flat word buffer holding each distinct multi-cell constant
    /// once. Referenced by `MovConstPool { pool_offset, size }` instead of being re-spilled
    /// into every function's frame.
    pub constant_pool: Vec<u64>,
}

impl Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let max_line_number: usize = self.functions.iter().map(|f| f.code.len()).sum::<usize>() - 1;
        let max_line_number_digits = max_line_number.to_string().len();
        let max_function_idx = self.functions.len().to_string().len() - 1;
        let max_function_idx_digits = max_function_idx.to_string().len();
        let mut line = 0;
        for (i, function) in self.functions.iter().enumerate() {
            writeln!(
                f,
                "{: >max_function_idx_digits$}: fn {} (frame_size = {})",
                i, function.name, function.frame_size
            )?;
            for op in &function.code {
                writeln!(f, "  {: >max_line_number_digits$}: {}", line, op)?;
                line += 1;
            }
        }
        Ok(())
    }
}

/// Encode a single field of a `StructDescriptor` as one `u64`:
/// high bit = refcounted flag, low 32 bits = field size in u64 words.
#[inline(always)]
fn encode_struct_field(size: u32, refcounted: bool) -> u64 {
    (refcounted as u64) << 63 | (size as u64)
}

#[inline(always)]
fn decode_struct_field(word: u64) -> (u32, bool) {
    let refcounted = (word >> 63) != 0;
    let size = (word & 0xFFFF_FFFF) as u32;
    (size, refcounted)
}

impl Program {
    /// Serialize executable VM bytecode. Debug information is never embedded in this output.
    pub fn to_binary(&self) -> Vec<u64> {
        self.to_binary_impl(false).0
    }

    /// Backwards-compatible alias for [`Program::to_binary`].
    pub fn to_binary_without_debug_info(&self) -> Vec<u64> {
        self.to_binary()
    }

    /// Serialize compact executable bytecode and construct its standalone source map.
    pub fn to_binary_and_debug_info(&self) -> (Vec<u64>, DebugInfo) {
        let (binary, debug_info) = self.to_binary_impl(true);
        (
            binary,
            debug_info.expect("debug info was requested while serializing VM bytecode"),
        )
    }

    fn to_binary_impl(&self, include_debug_info: bool) -> (Vec<u64>, Option<DebugInfo>) {
        let mut binary = Vec::new();
        // Layout-table header: [num_descriptors, ...descriptors...].
        // Each descriptor: [num_fields, field_0_packed, field_1_packed, ...].
        binary.push(self.struct_layouts.len() as u64);
        for desc in &self.struct_layouts {
            let fields = desc.fields();
            binary.push(fields.len() as u64);
            for &(size, refcounted) in fields {
                binary.push(encode_struct_field(size, refcounted));
            }
        }

        // Constant pool: [pool_len, ...words...].
        binary.push(self.constant_pool.len() as u64);
        binary.extend_from_slice(&self.constant_pool);

        binary.push(self.global_frame_size as u64);
        binary.push(self.entry_blob_field_count as u64);

        // Entry-point table: [num_entries, marker_offset_0, ...]. The offsets are absolute word
        // indices of each entry function's `u64::MAX` marker; they are patched in once function
        // positions are known.
        binary.push(self.entry_points.len() as u64);
        let entry_table_start = binary.len();
        binary.extend(std::iter::repeat(0u64).take(self.entry_points.len()));

        let mut positions = vec![];
        let mut jumps_to_fix: Vec<(usize, isize)> = vec![];
        let mut function_markers = vec![];
        let mut debug_info = include_debug_info.then(|| DebugInfo {
            functions: Vec::with_capacity(self.functions.len()),
            ..DebugInfo::default()
        });

        for function in &self.functions {
            if include_debug_info {
                assert_eq!(
                    function.code.len(),
                    function.source_locations.len(),
                    "every VM opcode must have a source location"
                );
            }
            // Function marker
            function_markers.push(binary.len());
            let function_offset = binary.len();
            binary.push(u64::MAX);
            binary.push(function.frame_size as u64);

            let mut locations = debug_info.as_ref().map(|_| Vec::new());
            for (opcode_index, op) in function.code.iter().enumerate() {
                if let Some(locations) = &mut locations {
                    let location = &function.source_locations[opcode_index];
                    if opcode_index == 0 || function.source_locations[opcode_index - 1] != *location
                    {
                        let debug_info = debug_info.as_mut().unwrap();
                        let file_index = debug_info
                            .files
                            .iter()
                            .position(|file| file == &location.file)
                            .unwrap_or_else(|| {
                                debug_info.files.push(location.file.clone());
                                debug_info.files.len() - 1
                            });
                        locations.push(DebugLocation {
                            code_offset: binary.len(),
                            file_index,
                            line: location.line,
                            column: location.column,
                        });
                    }
                }
                positions.push(binary.len());
                op.to_binary(&mut binary, &mut jumps_to_fix);
            }
            if let Some(locations) = locations {
                debug_info.as_mut().unwrap().functions.push(DebugFunction {
                    name: function.name.clone(),
                    code_offset: function_offset,
                    locations,
                });
            }
        }

        for (slot, fn_idx) in self.entry_points.iter().enumerate() {
            binary[entry_table_start + slot] = function_markers[*fn_idx] as u64;
        }
        for (jump_position, add_offset) in jumps_to_fix {
            let target = binary[jump_position];
            let target_pos = positions[target as usize];
            binary[jump_position] =
                (target_pos as isize - (jump_position as isize + add_offset)) as u64;
        }
        (binary, debug_info)
    }
}

/// The decoded header of a program binary.
pub struct ProgramHeader {
    pub struct_layouts: Vec<StructDescriptor>,
    /// Interned constant pool (flat words), read at runtime by `mov_const_pool`.
    pub constant_pool: Vec<u64>,
    pub global_frame_size: usize,
    /// The exact number of input fields the caller must supply.
    pub entry_blob_field_count: usize,
    /// Absolute word offsets of each entry point's function marker, in entry-table order
    /// ([`ENTRY_WITGEN`], [`ENTRY_AD`], ...). The entry's frame size lives at `offset + 1` and
    /// its first opcode at `offset + 2`.
    pub entry_points: Vec<usize>,
    /// Word offset of the first function marker, i.e. where the opcode stream begins.
    pub code_start: usize,
}

/// Decode a program binary's header: struct layouts, global frame size and the entry-point
/// table.
pub fn parse_program_header(program: &[u64]) -> ProgramHeader {
    let (struct_layouts, off) = parse_struct_layouts(program);
    let pool_len = program[off] as usize;
    let constant_pool = program[off + 1..off + 1 + pool_len].to_vec();
    let off = off + 1 + pool_len;
    let global_frame_size = program[off] as usize;
    let entry_blob_field_count = program[off + 1] as usize;
    let num_entries = program[off + 2] as usize;
    let entry_points: Vec<usize> = (0..num_entries)
        .map(|i| program[off + 3 + i] as usize)
        .collect();
    let code_start = off + 3 + num_entries;
    ProgramHeader {
        struct_layouts,
        constant_pool,
        global_frame_size,
        entry_blob_field_count,
        entry_points,
        code_start,
    }
}

/// Read the struct-layout table from the binary header and return both the
/// descriptors and the offset (in u64 words) at which the rest of the program
/// (starting with `global_frame_size`) begins.
pub fn parse_struct_layouts(program: &[u64]) -> (Vec<StructDescriptor>, usize) {
    let num_descriptors = program[0] as usize;
    let mut layouts = Vec::with_capacity(num_descriptors);
    let mut off = 1usize;
    for _ in 0..num_descriptors {
        let n = program[off] as usize;
        off += 1;
        let mut fields = Vec::with_capacity(n);
        for _ in 0..n {
            fields.push(decode_struct_field(program[off]));
            off += 1;
        }
        layouts.push(StructDescriptor::new(fields));
    }
    (layouts, off)
}

/// Every width the `_int128` lane holds that is worth sweeping, narrowest first.
///
/// The interior widths are the range's corners: two non-powers of two, where masking an amount by
/// `bits - 1` and reducing it modulo `bits` stop agreeing, and one below the top.
#[cfg(test)]
fn double_lane_widths() -> Vec<u64> {
    use mavros_int_semantics::int_bits::{HOST_LIMB_BITS, HOST_WORD_BITS};

    let narrowest = HOST_LIMB_BITS as u64 + 1;
    let widest = HOST_WORD_BITS as u64;
    vec![narrowest, 96, 100, widest - 1, widest]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn location(function: &str, line: u64) -> SourceLocation {
        SourceLocation::new(format!("src/{function}.nr"), line, 7)
    }

    fn empty_witgen_vm() -> VM {
        VM::new_witgen(
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            0,
            0,
            ptr::null_mut(),
            Vec::new(),
            Vec::new(),
        )
    }

    /// Run the wide lane through real dispatch.
    ///
    /// The conformance sweeps in `int_limbs` call the bodies directly, so they say nothing about
    /// whether an operand survives being written to the binary and read back: a slice takes its
    /// length from a width immediate the same instruction carries, and a writer and a reader that
    /// disagreed about the operand order would still compute the right answer on the wrong cells.
    ///
    /// The program ends in `trap` rather than `ret` because `ret` deallocates the base frame, and
    /// the results are read out of that frame afterwards.
    #[test]
    fn the_wide_lane_survives_the_round_trip_through_dispatch() {
        const BITS: u64 = 192;
        let (a, b, sum, product, less, narrowed) = (2usize, 5, 8, 11, 14, 15);

        let code = vec![
            OpCode::AddIntn {
                bits: BITS,
                res: FramePosition(sum),
                a: FramePosition(a),
                b: FramePosition(b),
            },
            OpCode::MulIntn {
                bits: BITS,
                res: FramePosition(product),
                a: FramePosition(a),
                b: FramePosition(b),
            },
            OpCode::UltIntn {
                bits: BITS,
                res: FramePosition(less),
                a: FramePosition(a),
                b: FramePosition(b),
            },
            OpCode::CastIntn {
                from_bits: BITS,
                to_bits: 64,
                res: FramePosition(narrowed),
                a: FramePosition(a),
            },
            OpCode::Trap {},
        ];
        let locations = vec![location("main", 1); code.len()];
        let program = Program {
            functions: vec![Function {
                name: "main".to_string(),
                frame_size: 20,
                code,
                source_locations: locations,
            }],
            entry_points: vec![0],
            entry_blob_field_count: 0,
            global_frame_size: 0,
            struct_layouts: Vec::new(),
            constant_pool: Vec::new(),
        };

        let (mut binary, debug_info) = program.to_binary_and_debug_info();
        let header = parse_program_header(&binary);
        crate::interpreter::prepare_dispatch(&mut binary, header.code_start);

        let mut vm = empty_witgen_vm();
        vm.set_debug_context(binary.as_ptr(), binary.len(), debug_info);
        let frame = Frame::base_frame(20, &mut vm);

        // Every cell either operand covers is written below, which a pushed frame requires: it is
        // `alloc`ated rather than zeroed.
        //
        // `a` is the largest value one limb holds, so every answer below crosses a limb boundary.
        for (i, word) in [u64::MAX, 0, 0].into_iter().enumerate() {
            frame.write_u64((a + i) as isize, word);
        }
        for (i, word) in [3u64, 0, 0].into_iter().enumerate() {
            frame.write_u64((b + i) as isize, word);
        }

        let entry = header.entry_points[0];
        unsafe { crate::interpreter::dispatch(binary.as_ptr().add(entry + 2), frame, &mut vm) };

        let cells =
            |at: usize| -> Vec<u64> { (0..3).map(|i| frame.read_u64((at + i) as isize)).collect() };
        // (2^64 - 1) + 3 carries into the second limb.
        assert_eq!(cells(sum), vec![2, 1, 0], "add_intn");
        // 3 * (2^64 - 1) is 2*2^64 + (2^64 - 3).
        assert_eq!(cells(product), vec![u64::MAX - 2, 2, 0], "mul_intn");
        assert_eq!(
            frame.read_u64(less as isize),
            0,
            "ult_intn: 2^64 - 1 is not below 3"
        );
        assert_eq!(
            frame.read_u64(narrowed as isize),
            u64::MAX,
            "cast_intn to 64 bits"
        );

        frame.pop(&mut vm);
    }

    /// The other half of the overlap check: a scalar result inside a wide operand.
    ///
    /// `eq_intn` has no result _slice_, so the slice-against-slice check never runs for it. What
    /// would corrupt it is a write through its `#[out]` into cells one of its operands is reading,
    /// which is undefined behavior for the shared slice however the pointer was obtained.
    #[test]
    #[should_panic(expected = "overlap")]
    fn a_scalar_result_inside_a_wide_operand_is_caught() {
        let program = Program {
            functions: vec![Function {
                name: "main".to_string(),
                frame_size: 12,
                // `res` is one cell at 1, inside `a`'s three cells at 0.
                code: vec![
                    OpCode::EqIntn {
                        bits: 192,
                        res: FramePosition(1),
                        a: FramePosition(0),
                        b: FramePosition(4),
                    },
                    OpCode::Trap {},
                ],
                source_locations: vec![location("main", 1); 2],
            }],
            entry_points: vec![0],
            entry_blob_field_count: 0,
            global_frame_size: 0,
            struct_layouts: Vec::new(),
            constant_pool: Vec::new(),
        };
        let (mut binary, debug_info) = program.to_binary_and_debug_info();
        let header = parse_program_header(&binary);
        crate::interpreter::prepare_dispatch(&mut binary, header.code_start);
        let mut vm = empty_witgen_vm();
        vm.set_debug_context(binary.as_ptr(), binary.len(), debug_info);
        let frame = Frame::base_frame(12, &mut vm);
        let entry = header.entry_points[0];
        unsafe { crate::interpreter::dispatch(binary.as_ptr().add(entry + 2), frame, &mut vm) };
    }

    /// The two wide shapes the round trip above does not reach.
    ///
    /// `not_intn` is the only `_intn` opcode with one operand, and `assert_eq_intn` is the only one
    /// whose instruction length is written by hand rather than derived by the macro — its
    /// `pc.offset(4)`. A wrong length there does not compute a wrong answer, it resumes dispatch on
    /// an operand word, so the check is that the instruction _after_ the assertion runs at all.
    #[test]
    fn the_remaining_wide_shapes_survive_dispatch() {
        const BITS: u64 = 192;
        let (a, b, complement) = (0usize, 3, 6);

        let code = vec![
            OpCode::AssertEqIntn {
                bits: BITS,
                a: FramePosition(a),
                b: FramePosition(b),
            },
            OpCode::NotIntn {
                bits: BITS,
                res: FramePosition(complement),
                a: FramePosition(a),
            },
            OpCode::Trap {},
        ];
        let program = Program {
            functions: vec![Function {
                name: "main".to_string(),
                frame_size: 12,
                source_locations: vec![location("main", 1); code.len()],
                code,
            }],
            entry_points: vec![0],
            entry_blob_field_count: 0,
            global_frame_size: 0,
            struct_layouts: Vec::new(),
            constant_pool: Vec::new(),
        };

        let (mut binary, debug_info) = program.to_binary_and_debug_info();
        let header = parse_program_header(&binary);
        crate::interpreter::prepare_dispatch(&mut binary, header.code_start);

        let mut vm = empty_witgen_vm();
        vm.set_debug_context(binary.as_ptr(), binary.len(), debug_info);
        let frame = Frame::base_frame(12, &mut vm);
        // A pushed frame is `alloc`ated and not zeroed, so every cell an operand covers is written.
        for cell in 0..12 {
            frame.write_u64(cell, 0);
        }
        for (i, word) in [7u64, 0, u64::MAX].into_iter().enumerate() {
            frame.write_u64((a + i) as isize, word);
            frame.write_u64((b + i) as isize, word);
        }

        let entry = header.entry_points[0];
        unsafe { crate::interpreter::dispatch(binary.as_ptr().add(entry + 2), frame, &mut vm) };

        let got: Vec<u64> = (0..3)
            .map(|i| frame.read_u64((complement + i) as isize))
            .collect();
        assert_eq!(
            got,
            vec![!7u64, u64::MAX, 0],
            "the assertion passed but did not resume on the instruction after it"
        );

        frame.pop(&mut vm);
    }

    /// The double lane wraps at its operand's width, not at the width of its two cells.
    #[test]
    fn the_double_lane_wraps_at_the_operand_width() {
        const BITS: u64 = 96;
        let (a, b, sum) = (2usize, 4, 6);

        let program = Program {
            functions: vec![Function {
                name: "main".to_string(),
                frame_size: 12,
                code: vec![
                    OpCode::AddInt128 {
                        res: FramePosition(sum),
                        a: FramePosition(a),
                        b: FramePosition(b),
                        bits: BITS,
                    },
                    OpCode::Trap {},
                ],
                source_locations: vec![location("main", 1); 2],
            }],
            entry_points: vec![0],
            entry_blob_field_count: 0,
            global_frame_size: 0,
            struct_layouts: Vec::new(),
            constant_pool: Vec::new(),
        };

        let (mut binary, debug_info) = program.to_binary_and_debug_info();
        let header = parse_program_header(&binary);
        crate::interpreter::prepare_dispatch(&mut binary, header.code_start);

        let mut vm = empty_witgen_vm();
        vm.set_debug_context(binary.as_ptr(), binary.len(), debug_info);
        let frame = Frame::base_frame(12, &mut vm);

        // A pushed frame is `alloc`ated and not zeroed, so every cell an opcode reads has to be
        // written here, including the high cell of a small addend, which is otherwise whatever the
        // allocator last left there.
        for cell in 0..12 {
            frame.write_u64(cell, 0);
        }

        // The largest value 96 bits holds, plus one: zero at 96 bits, and `2^96` at 128.
        frame.write_u64(a as isize, u64::MAX);
        frame.write_u64(a as isize + 1, 0xFFFF_FFFF);
        frame.write_u64(b as isize, 1);

        let entry = header.entry_points[0];
        unsafe { crate::interpreter::dispatch(binary.as_ptr().add(entry + 2), frame, &mut vm) };

        assert_eq!(
            (
                frame.read_u64(sum as isize),
                frame.read_u64(sum as isize + 1)
            ),
            (0, 0),
            "the sum wrapped at 128 bits rather than at 96"
        );

        frame.pop(&mut vm);
    }

    /// The overlap check exists because SSA is what keeps a result off its own operands: a property
    /// of the frame layouter in another crate.
    #[test]
    #[should_panic(expected = "overlap")]
    fn a_result_sharing_cells_with_an_operand_is_caught() {
        const BITS: u64 = 192;
        let program = Program {
            functions: vec![Function {
                name: "main".to_string(),
                frame_size: 12,
                // `res` at cell 3 covers 3..6 and `b` at cell 5 covers 5..8.
                code: vec![
                    OpCode::AddIntn {
                        bits: BITS,
                        res: FramePosition(3),
                        a: FramePosition(0),
                        b: FramePosition(5),
                    },
                    OpCode::Trap {},
                ],
                source_locations: vec![location("main", 1); 2],
            }],
            entry_points: vec![0],
            entry_blob_field_count: 0,
            global_frame_size: 0,
            struct_layouts: Vec::new(),
            constant_pool: Vec::new(),
        };

        let (mut binary, debug_info) = program.to_binary_and_debug_info();
        let header = parse_program_header(&binary);
        crate::interpreter::prepare_dispatch(&mut binary, header.code_start);

        let mut vm = empty_witgen_vm();
        vm.set_debug_context(binary.as_ptr(), binary.len(), debug_info);
        let frame = Frame::base_frame(12, &mut vm);
        let entry = header.entry_points[0];
        unsafe { crate::interpreter::dispatch(binary.as_ptr().add(entry + 2), frame, &mut vm) };
    }

    #[test]
    fn indexed_accesses_trap_or_advance() {
        // Two four-word elements: index 1 is valid, 2 is out of bounds, and
        // 1 << 62 would wrap to zero if multiplied by the stride before checking.
        for index in [1, 2, 1 << 62] {
            for opcode in [
                OpCode::ArrayGet {
                    res: FramePosition(2),
                    array: FramePosition(6),
                    index: FramePosition(7),
                    stride: 4,
                },
                OpCode::ArraySet {
                    res: FramePosition(2),
                    array: FramePosition(6),
                    index: FramePosition(7),
                    source: FramePosition(8),
                    stride: 4,
                },
                OpCode::BlobGet {
                    res: FramePosition(2),
                    source: FramePosition(8),
                    index: FramePosition(7),
                    stride: 4,
                    len: 2,
                },
            ] {
                let mut vm = empty_witgen_vm();
                let frame = Frame::base_frame(16, &mut vm);
                let array = BoxedValue::alloc(BoxedLayout::array(8, false), &mut vm);
                let initial = [10, 11, 12, 13, 14, 15, 16, 17];
                unsafe {
                    ptr::copy_nonoverlapping(initial.as_ptr(), array.data(), 8);
                    ptr::write_bytes(frame.data, 0, 16);
                    *frame.data.add(6) = array.0 as u64;
                    *frame.data.add(7) = index;
                    ptr::copy_nonoverlapping(initial.as_ptr(), frame.data.add(8), 8);
                }
                let mut binary = Vec::new();
                opcode.to_binary(&mut binary, &mut Vec::new());
                let (next_pc, _) = DISPATCH[binary[0] as usize](binary.as_ptr(), frame, &mut vm);
                let output = unsafe { std::slice::from_raw_parts(frame.data.add(2), 4) };
                let contents = unsafe { std::slice::from_raw_parts(array.data(), 8) };
                if index == 1 {
                    assert!(!vm.trapped, "{opcode}");
                    assert_eq!(next_pc, unsafe { binary.as_ptr().add(binary.len()) });
                    match opcode {
                        OpCode::ArraySet { .. } => {
                            assert_eq!(output[0], array.0 as u64);
                            assert_eq!(contents, &[10, 11, 12, 13, 10, 11, 12, 13]);
                        }
                        _ => assert_eq!(output, &[14, 15, 16, 17]),
                    }
                } else {
                    assert!(vm.trapped, "{opcode}, index={index}");
                    assert!(next_pc.is_null());
                    assert_eq!(output, &[0; 4]);
                    assert_eq!(contents, initial);
                }
                array.dec_rc(&mut vm);
                frame.pop(&mut vm);
            }
        }
    }

    #[test]
    fn source_map_drives_deep_instruction_profile_and_trap_stack_trace() {
        let caller_location = location("main", 10);
        let middle_location = location("middle", 17);
        let callee_location = location("helper", 24);
        let program = Program {
            functions: vec![
                Function {
                    name: "main".to_string(),
                    frame_size: 3,
                    code: vec![OpCode::Nop {}, OpCode::Ret {}],
                    source_locations: vec![caller_location.clone(), caller_location.clone()],
                },
                Function {
                    name: "middle".to_string(),
                    frame_size: 3,
                    code: vec![OpCode::Nop {}, OpCode::Ret {}],
                    source_locations: vec![middle_location.clone(), middle_location.clone()],
                },
                Function {
                    name: "helper".to_string(),
                    frame_size: 3,
                    code: vec![OpCode::Nop {}, OpCode::Ret {}],
                    source_locations: vec![callee_location.clone(), callee_location.clone()],
                },
            ],
            entry_points: vec![0],
            entry_blob_field_count: 0,
            global_frame_size: 0,
            struct_layouts: Vec::new(),
            constant_pool: Vec::new(),
        };
        let (binary, debug_info) = program.to_binary_and_debug_info();
        let main_opcode = debug_info.functions[0].locations[0].code_offset;
        let middle_opcode = debug_info.functions[1].locations[0].code_offset;
        let helper_opcode = debug_info.functions[2].locations[0].code_offset;

        let mut vm = empty_witgen_vm();
        vm.set_debug_context(binary.as_ptr(), binary.len(), debug_info);

        let caller = Frame::base_frame(3, &mut vm);
        let middle = Frame::push(3, caller, &mut vm);
        let callee = Frame::push(3, middle, &mut vm);
        unsafe {
            *middle.data.offset(1) = binary.as_ptr().add(main_opcode + 1) as u64;
            *callee.data.offset(1) = binary.as_ptr().add(middle_opcode + 1) as u64;
        }
        vm.enable_instruction_profile();
        vm.record_instruction(unsafe { binary.as_ptr().add(helper_opcode) }, callee);
        assert_eq!(
            vm.take_instruction_profile().to_folded(),
            "main;middle;helper 1\n"
        );
        vm.capture_trap(unsafe { binary.as_ptr().add(helper_opcode) }, callee);

        assert_eq!(
            vm.stack_trace,
            vec![
                StackFrame {
                    function: "helper".to_string(),
                    location: callee_location,
                },
                StackFrame {
                    function: "middle".to_string(),
                    location: middle_location,
                },
                StackFrame {
                    function: "main".to_string(),
                    location: caller_location,
                },
            ]
        );

        let middle = callee.pop(&mut vm);
        let caller = middle.pop(&mut vm);
        let root = caller.pop(&mut vm);
        assert!(root.data.is_null());
    }

    #[test]
    fn instruction_profile_uses_unknown_without_debug_context() {
        let mut vm = empty_witgen_vm();
        vm.enable_instruction_profile();
        vm.record_instruction(
            ptr::null(),
            Frame {
                data: ptr::null_mut(),
            },
        );

        assert_eq!(vm.take_instruction_profile().to_folded(), "<unknown> 1\n");
    }

    #[test]
    fn compact_binary_header_parses_without_debug_metadata() {
        // Empty header: no struct layouts, constants, globals, entry blob fields, or
        // entries, then code starts.
        let binary = [0, 0, 0, 0, 0, u64::MAX, 0];
        let header = parse_program_header(&binary);

        assert_eq!(header.code_start, 5);
        assert!(header.constant_pool.is_empty());
    }

    #[test]
    fn bytecode_debug_info_is_always_standalone() {
        let source_location = location("main", 10);
        let program = Program {
            functions: vec![Function {
                name: "main".to_string(),
                frame_size: 3,
                code: vec![OpCode::Nop {}, OpCode::Ret {}],
                source_locations: vec![source_location.clone(), source_location],
            }],
            entry_points: vec![0],
            entry_blob_field_count: 0,
            global_frame_size: 0,
            struct_layouts: Vec::new(),
            constant_pool: Vec::new(),
        };

        let (binary, debug_info) = program.to_binary_and_debug_info();

        assert_eq!(binary, program.to_binary_without_debug_info());
        assert_eq!(binary, program.to_binary());
        assert_eq!(debug_info.format_version(), DEBUG_INFO_FORMAT_VERSION);
        assert_eq!(debug_info.files, vec!["src/main.nr"]);
        assert_eq!(debug_info.functions.len(), 1);
        assert_eq!(debug_info.functions[0].name, "main");
        assert_eq!(debug_info.functions[0].locations.len(), 1);
        assert_eq!(debug_info.functions[0].locations[0].file_index, 0);
        assert_eq!(parse_program_header(&binary).entry_points.len(), 1);
    }

    #[test]
    fn debug_info_serialization_interns_files_and_validates_the_version() {
        let source_location = location("shared", 10);
        let program = Program {
            functions: vec![Function {
                name: "main".to_string(),
                frame_size: 3,
                code: vec![OpCode::Nop {}, OpCode::Ret {}],
                source_locations: vec![
                    source_location.clone(),
                    SourceLocation::new(source_location.file.clone(), 11, 7),
                ],
            }],
            entry_points: vec![0],
            entry_blob_field_count: 0,
            global_frame_size: 0,
            struct_layouts: Vec::new(),
            constant_pool: Vec::new(),
        };

        let (_, debug_info) = program.to_binary_and_debug_info();
        assert_eq!(debug_info.files, vec!["src/shared.nr"]);
        assert!(
            debug_info.functions[0]
                .locations
                .iter()
                .all(|location| location.file_index == 0)
        );

        let json = serde_json::to_string(&debug_info).unwrap();
        assert_eq!(json.matches("src/shared.nr").count(), 1);
        let unsupported = json.replace(
            &format!("\"formatVersion\":{DEBUG_INFO_FORMAT_VERSION}"),
            "\"formatVersion\":2",
        );
        assert!(serde_json::from_str::<DebugInfo>(&unsupported).is_err());
        assert_eq!(
            DebugInfo::default().format_version(),
            DEBUG_INFO_FORMAT_VERSION
        );
    }

    /// A key names its own row while it is inside the table and no row at all from the end onwards.
    ///
    /// The bound is what stops [`table_row_index`]'s callers from bumping a multiplicity outside
    /// the buffer: every one of them addresses `multiplicities_wit` with the returned row through
    /// raw pointer arithmetic, which no bounds check stands behind.
    #[test]
    fn table_row_index_accepts_only_keys_inside_the_table() {
        assert_eq!(table_row_index(Field::from(0u64), 8), Some(0));
        assert_eq!(table_row_index(Field::from(7u64), 8), Some(7));
        assert_eq!(table_row_index(Field::from(8u64), 8), None);
        assert_eq!(table_row_index(Field::from(9u64), 8), None);
        // A degenerate table has no rows to land on, so nothing is a member of it.
        assert_eq!(table_row_index(Field::from(0u64), 0), None);
    }

    /// The bound alone is not enough. The row is addressed by the key's _low limb_, so a key whose
    /// high limbs are set has to be rejected even when that low limb names a valid row — otherwise
    /// `2^64 + 3` reads as row 3 of an 8-row table, and the lookup it cannot satisfy silently
    /// succeeds against a row it does not name.
    #[test]
    fn table_row_index_rejects_a_key_whose_high_limbs_are_set() {
        let low_limb_in_range = Field::from((1u128 << 64) + 3);
        assert_eq!(
            ark_ff::PrimeField::into_bigint(low_limb_in_range).0[0],
            3,
            "the point of this case is that the low limb alone looks like a valid row"
        );
        assert_eq!(table_row_index(low_limb_in_range, 8), None);

        // The same shape at the top of the field: `-1` truncates to an arbitrary `u64`, which is
        // the case that would land outside the multiplicities buffer entirely if it were taken.
        assert_eq!(table_row_index(-Field::from(1u64), 8), None);
    }

    #[test]
    fn cell_mask_holds_a_value_to_its_declared_width() {
        assert_eq!(cell_mask(1), 0x1);
        assert_eq!(cell_mask(8), 0xFF);
        assert_eq!(cell_mask(32), 0xFFFF_FFFF);
        // At and past the cell width the mask is a no-op, and must not be computed as
        // `(1 << bits) - 1` -- that shift is an overflow at 64.
        assert_eq!(cell_mask(64), u64::MAX);
        assert_eq!(cell_mask(128), u64::MAX);
    }

    #[test]
    fn a_pow2_table_row_is_two_to_its_index() {
        let rows: Vec<Field> = pow2_rows(128).collect();
        assert_eq!(rows.len(), 128);
        assert_eq!(rows[0], Field::from(1u64));
        assert_eq!(rows[1], Field::from(2u64));

        // Every row expressible as a `u64`, pinned against the host shift.
        for n in 0..64 {
            assert_eq!(rows[n], Field::from(1u64 << n), "row {n}");
        }

        // Past a `u64`, which the widest table -- a 128-bit shift -- reaches. `pow2_rows` doubles
        // rather than shifting for exactly this reason. Checked against square-and-multiply rather
        // than against doubling, which would only restate the implementation: this is the row a
        // second algorithm agrees on, and `emit_pow2_ad_init_body` must produce the same one or the
        // backends disagree about the table's contents.
        for (n, row) in rows.iter().enumerate() {
            let by_exponentiation = <Field as ark_ff::Field>::pow(&Field::from(2u64), [n as u64]);
            assert_eq!(*row, by_exponentiation, "row {n}");
        }

        // And the widest row is a genuine power of two rather than a residue: `MAX_POW2_TABLE_SIZE`
        // is set so the table clears the modulus, and a table that wrapped would still pass every
        // assertion above, since both algorithms wrap alike.
        let widest = ark_ff::PrimeField::into_bigint(rows[127]).0;
        assert_eq!(widest, [0, 1u64 << 63, 0, 0], "row 127 is not 2^127");
    }

    #[test]
    fn a_shift_amount_is_reduced_to_the_operand_width() {
        // In range: the amount passes through untouched.
        assert_eq!(shift_amount(0, 8), 0);
        assert_eq!(shift_amount(7, 8), 7);
        assert_eq!(shift_amount(63, 64), 63);

        // Out of range: reduced modulo the width, which is what LLVM does with the same operands
        // (`llssa_to_llvm::reduce_shift_count`). Every shift opcode routes its amount through here
        // rather than using it raw, which is what keeps the two backends agreeing.
        assert_eq!(shift_amount(8, 8), 0);
        assert_eq!(shift_amount(9, 8), 1);
        assert_eq!(shift_amount(64, 64), 0);
        assert_eq!(shift_amount(65, 64), 1);

        // At a non-power-of-two width a mask and a modulo part company, and only the modulo leaves
        // an in-range amount alone. A mask by `bits - 1 == 6` would answer `0` for both of these.
        assert_eq!(shift_amount(1, 7), 1);
        assert_eq!(shift_amount(8, 7), 1);
        assert_eq!(
            shift_amount(6, 7),
            6,
            "the largest legal amount at seven bits"
        );
    }

    #[test]
    fn a_shift_amount_lands_below_the_width_at_every_width() {
        // The property the backstop rests on, and the reason it is safe to apply blanket rather
        // than only where `bits` makes the mask a modulo: `b & (bits - 1)` is a submask of
        // `bits - 1`, so it cannot exceed it. Checked at the non-powers of two too, which
        // `BitRange` can mint even though no shift reaches one today.
        for bits in [1u64, 3, 8, 17, 24, 32, 63, 64] {
            for b in [0u64, 1, 7, 23, 63, 64, 200, u64::MAX] {
                let amount = u64::from(shift_amount(b, bits));
                assert!(
                    amount < bits,
                    "amount {amount} escaped a {bits}-bit width from {b}"
                );
            }
        }
    }

    #[test]
    fn an_over_shift_reduces_rather_than_panicking() {
        // The whole point of routing the amount through `shift_amount`: a bare `a << b` panics in
        // a debug build for `b >= 64` and is silently `b & 63` in release, which is the wrong
        // answer for every width below 64. Both directions, through the opcode bodies themselves.
        let a = 0xABu64;
        assert_eq!(cell_shl(a, 8, 8), a, "an 8-bit `<<` by 8 reduces to `<< 0`");
        assert_eq!(
            cell_ushr(a, 8, 8),
            a,
            "an 8-bit `>>` by 8 reduces to `>> 0`"
        );

        // An amount past the _host_ width too, which is where a bare `a << b` panics in debug.
        assert_eq!(cell_shl(a, 200, 8), a);
        assert_eq!(cell_ushr(a, 200, 8), a);
        assert_eq!(cell_ashr(a, 200, 8), a);
        // At the host width the mask is `& 63`, so the largest amount there is a shift by 63
        // rather than a panic.
        assert_eq!(cell_shl(a, u64::MAX, 64), a << 63);
    }

    #[test]
    fn a_shift_result_stays_inside_its_declared_width() {
        // `shl_int` re-masks because bits pushed off the top are discarded rather than kept, which
        // is Noir's semantics for a `<<` that overflows -- `200u8 << 1` is 144, not 400.
        assert_eq!(cell_shl(200, 1, 8), 144);
        assert_eq!(cell_shl(1, 7, 8), 128);
        assert_eq!(cell_shl(1, 0, 8), 1);
        // Every result is inside the width, whatever the operands.
        for bits in [1u64, 8, 32, 64] {
            for a in [0u64, 1, 0x5A, u64::MAX] {
                for b in [0u64, 1, 7, 63, 200] {
                    let a = a & cell_mask(bits);
                    for out in [
                        cell_shl(a, b, bits),
                        cell_ushr(a, b, bits),
                        cell_ashr(a, b, bits),
                        cell_complement(a, bits),
                    ] {
                        assert_eq!(
                            out & !cell_mask(bits),
                            0,
                            "a {bits}-bit result escaped its width from a={a} b={b}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_right_shift_fills_by_its_reading() {
        // The one pair where the reading changes the answer, which is why there are two opcodes.
        // `0xF0` is `-16` at eight bits, so an arithmetic `>> 2` keeps the sign and a logical one
        // does not.
        assert_eq!(cell_ashr(0xF0, 2, 8), 0xFC, "sign-filled: -16 >> 2 == -4");
        assert_eq!(cell_ushr(0xF0, 2, 8), 0x3C, "zero-filled: 240 >> 2 == 60");
        // A non-negative operand is the case where they agree.
        assert_eq!(cell_ashr(0x70, 2, 8), 0x1C);
        assert_eq!(cell_ushr(0x70, 2, 8), 0x1C);
        // At the host width, where the sign-extend preamble is the identity.
        assert_eq!(cell_ashr(u64::MAX, 3, 64), u64::MAX);
        assert_eq!(cell_ushr(u64::MAX, 3, 64), u64::MAX >> 3);
    }

    #[test]
    fn the_double_lane_reduces_its_shift_amount_like_llvm() {
        // At a power-of-two width the mask reaches only the low limb, so discarding `b.hi` is
        // exact rather than a check going missing: LLVM masks the whole pattern with
        // `bit_width - 1`, and those bits live entirely in the low limb.
        let hi_set = Int128 { lo: 3, hi: 1 };
        assert_eq!(
            shift_amount_128(hi_set, 128),
            3,
            "a high limb cannot change the low seven bits"
        );
        assert_eq!(shift_amount_128(Int128 { lo: 1 << 32, hi: 0 }, 128), 0);
        assert_eq!(
            shift_amount_128(Int128 { lo: 128, hi: 0 }, 128),
            0,
            "128 reduces to a shift by zero"
        );

        // Spelled out independently at 128, where LLVM's reduction is a mask.
        for lo in [0u64, 1, 63, 64, 127, 128, 255, 1 << 32, u64::MAX] {
            for hi in [0u64, 1, u64::MAX] {
                let b = Int128 { lo, hi };
                let llvm = (b.to_u128() & 127) as u32;
                assert_eq!(
                    shift_amount_128(b, 128),
                    llvm,
                    "disagreed with LLVM's mask at {b:?}"
                );
            }
        }

        // Below the lane's top width the reduction is a real modulo, and the high limb carries
        // bits the modulus can see. The top width is the exhaustive block above, where it is a
        // mask instead, so it is the one width this loop drops.
        let top = *double_lane_widths().last().expect("the lane holds a width");
        for bits in double_lane_widths().into_iter().filter(|bits| *bits < top) {
            for lo in [0u64, 1, 63, 64, 95, 96, 1 << 32, u64::MAX] {
                for hi in [0u64, 1, u64::MAX] {
                    let b = Int128 { lo, hi };
                    assert_eq!(
                        u128::from(shift_amount_128(b, bits)),
                        b.to_u128() % u128::from(bits),
                        "the reduction at {bits} bits is not the model's at {b:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_zero_width_shift_answers_rather_than_panicking() {
        // A width of zero must not reach the host shift unreduced -- and `bits - 1` would
        // underflow to `u64::MAX`. A zero-width cell holds nothing, so zero is the only answer.
        assert_eq!(shift_amount(u64::MAX, 0), 0);
        assert_eq!(cell_shl(0, u64::MAX, 0), 0);
        assert_eq!(cell_ushr(0, u64::MAX, 0), 0);
    }

    #[test]
    fn division_is_total_at_every_width() {
        // The VM reports a failed execution through `trap`; aborting the host process instead
        // turns a rejectable user program into an undiagnosable witgen crash. `divmod_guard`
        // should mean these are never reached.
        assert_eq!(cell_udiv(7, 0), 0);
        assert_eq!(cell_urem(7, 0), 0);
        assert_eq!(cell_sdiv(7, 0, 8), 0);
        assert_eq!(cell_srem(7, 0, 8), 0);
        assert_eq!(
            Int128::from_u128(7).unsigned_div(Int128::default()),
            Int128::default()
        );
        assert_eq!(
            Int128::from_u128(7).unsigned_rem(Int128::default()),
            Int128::default()
        );

        // `INT_MIN / -1` is the other input the host aborts on, and only at the host's own width:
        // below 64 the `i64` division succeeds and the mask brings it back, and the 64-bit case
        // is held to that same wrapping rather than to the host's abort.
        assert_eq!(
            cell_sdiv(0x80, 0xFF, 8),
            0x80,
            "-128i8 / -1 wraps back to -128"
        );
        assert_eq!(cell_srem(0x80, 0xFF, 8), 0);
        let min64 = i64::MIN as u64;
        assert_eq!(
            cell_sdiv(min64, u64::MAX, 64),
            min64,
            "i64::MIN / -1 wraps rather than aborting"
        );
        assert_eq!(cell_srem(min64, u64::MAX, 64), 0);
    }

    #[test]
    fn division_rounds_toward_zero_and_the_remainder_takes_the_dividend() {
        // Noir's rule, via `expand_signed_math`: the quotient truncates toward zero, so the rem
        // carries the dividend's sign rather than the divisor's. `-7 / 2` is -3 and not -4, which
        // is what separates it from the arithmetic `>>` in `signed_shift`.
        let neg7 = 0xF9u64; // -7 at eight bits
        assert_eq!(signed_cell(cell_sdiv(neg7, 2, 8), 8), -3);
        assert_eq!(signed_cell(cell_srem(neg7, 2, 8), 8), -1);
        let neg2 = 0xFEu64; // -2 at eight bits
        assert_eq!(signed_cell(cell_sdiv(7, neg2, 8), 8), -3);
        assert_eq!(
            signed_cell(cell_srem(7, neg2, 8), 8),
            1,
            "the sign follows the dividend"
        );
        // Unsigned reads the same patterns as magnitudes, which is why there are two opcodes.
        assert_eq!(cell_udiv(neg7, 2), 0xF9 / 2);
    }

    #[test]
    fn a_signed_cell_is_read_at_its_own_width() {
        // The preamble `sdiv_int`, `srem_int`, `slt_int` and `ashr_int` share.
        assert_eq!(signed_cell(0xFF, 8), -1);
        assert_eq!(signed_cell(0x80, 8), -128);
        assert_eq!(signed_cell(0x7F, 8), 127);
        assert_eq!(signed_cell(0, 1), 0);
        assert_eq!(signed_cell(1, 1), -1, "the only negative a `u1` cell holds");
        assert_eq!(signed_cell(u64::MAX, 64), -1);
        // Dirty bits above the width are discarded by the up-shift, which is what makes the
        // signed opcodes robust to a cell the masked-cell invariant did not reach.
        assert_eq!(signed_cell(0xFFFF_FFFF_FFFF_FF7F, 8), 127);
    }

    #[test]
    fn a_complement_stays_inside_its_declared_width() {
        // An unmasked `!a` sets all 64 host bits, so an 8-bit `!0` would leave
        // `0xFFFF_FFFF_FFFF_FFFF` in a cell every later reader treats as 8 bits wide -- and, at
        // one bit, would make `!1` a _truthy_ value where the answer is `0`.
        assert_eq!(cell_complement(0x0F, 8), 0xF0);
        assert_eq!(cell_complement(0x00, 8), 0xFF);
        assert_eq!(cell_complement(0, 1), 1, "!false is true");
        assert_eq!(
            cell_complement(1, 1),
            0,
            "!true is false, not a large truthy value"
        );
        assert_eq!(cell_complement(0, 64), u64::MAX);

        // Agrees with the constant folder, `click_cooper::lattice::eval_not`, which is
        // `!x & bit_mask(bits)`. Complementing without re-masking would disagree with it at every
        // width below 64.
        for bits in [1u64, 8, 32, 64] {
            for a in [0u64, 1, 0x5A, u64::MAX] {
                let a = a & cell_mask(bits);
                assert_eq!(
                    cell_complement(a, bits),
                    !a & cell_mask(bits),
                    "complement escaped its width at {bits} bits"
                );
            }
        }
    }
}

/// The VM's conformance relation to the normative model in `mavros-int-semantics`.
///
/// The VM cannot _delegate_ to the model as it dispatches over `u64` frame cells with a separate
/// 128-bit lane, where the model is `u128` throughout and returns an [`Outcome`] the VM would
/// only compute in order to discard. The relation is thus checked instead of enforced, and it is:
///
/// 1. **Equal to [`residue`] wherever the model has an opinion.** `residue` is what a _total_
///    evaluator must produce, including on inputs Noir rejects.
/// 2. **Total.** No panic, no process abort, on any input at any width. A failed execution is
///    reported through the VM's own `trap`, never by killing the host.
/// 3. **Inside the width.** Every answer satisfies the masked-cell invariant, so the next opcode
///    to read the cell sees nothing above bit `bits - 1`.
///
/// Rule 1 is vacuous on the two inputs [`residue`] returns [`None`] for, a zero divisor and a
/// signed `INT_MIN / -1`: LLVM calls both undefined while the VM answers zero and wraps
/// respectively, so there is no agreed answer to hold either to. Rules 2 and 3 still bind there,
/// which is what makes those inputs safe rather than merely unspecified.
#[cfg(test)]
mod int_semantics_conformance {
    use mavros_int_semantics::{
        CmpOp, IntBits, IntOp, Sign, corners, int_bits::HOST_LIMB_BITS, residue,
    };

    use super::*;

    /// Every width the `_int` lane can hold, including the non-powers of two.
    ///
    /// The odd widths are here because these bodies are width-generic in both the VM and the
    /// model: they take `bits` and mask by it, so nothing about them is specific to the five
    /// widths Noir can name. Shifts share the set: `shift_amount` reduces the amount modulo the
    /// width rather than masking it by `bits - 1`, which is the model's reduction at every width
    /// and not only at a power of two.
    fn lane_widths(sign: Sign) -> Vec<u64> {
        corners::widths_for(sign.is_signed())
            .iter()
            .copied()
            .filter(|bits| *bits <= HOST_LIMB_BITS)
            .map(|bits| bits as u64)
            .collect()
    }

    /// Run one operation through the `_int` lane's opcode bodies.
    ///
    /// This is the dispatch `bytecode/mod.rs` performs when it picks a `BinaryArithOp`, written out
    /// so the conformance sweep exercises the same bodies the interpreter does without going
    /// through the dispatch loop.
    fn int_lane(op: IntOp, bits: u64, a: u64, b: u64) -> u64 {
        match op {
            // Add, sub and mul come in a signed pair in the model because they differ in _when they
            // fail_. By the time an opcode runs the guard IR has already decided. So both members
            // map to one body here, which is the same reason `residue` and not `eval` is what this
            // lane is held to.
            IntOp::UAdd | IntOp::SAdd => cell_add(a, b, bits),
            IntOp::USub | IntOp::SSub => cell_sub(a, b, bits),
            IntOp::UMul | IntOp::SMul => cell_mul(a, b, bits),
            IntOp::UDiv => cell_udiv(a, b),
            IntOp::SDiv => cell_sdiv(a, b, bits),
            IntOp::URem => cell_urem(a, b),
            IntOp::SRem => cell_srem(a, b, bits),
            IntOp::And => cell_and(a, b),
            IntOp::Or => cell_or(a, b),
            IntOp::Xor => cell_xor(a, b),
            // A left shift is one map on the bit pattern, so there is no signed form to dispatch
            // to; what a signed `<<` additionally rejects is a negative amount, and that rejection
            // is guard IR's rather than an opcode's. The model has one `Shl` for the same reason.
            IntOp::Shl => cell_shl(a, b, bits),
            IntOp::UShr => cell_ushr(a, b, bits),
            IntOp::SShr => cell_ashr(a, b, bits),
        }
    }

    /// Run one operation through the 128-bit lane's opcode bodies.
    ///
    /// Unsigned only, and that is the lane's contract rather than a gap in the sweep:
    /// `MAX_LOWERED_SIGNED_BITS` is 64, so no signed opcode ever reads a pattern this wide, which
    /// is why there is no `ashr_int128` or `sdiv_int128` to call.
    fn int128_lane(op: IntOp, bits: u64, a: Int128, b: Int128) -> Int128 {
        match op {
            IntOp::UAdd => double_add(a, b, bits),
            IntOp::USub => double_sub(a, b, bits),
            IntOp::UMul => double_mul(a, b, bits),
            // Division and the bitwise three carry no width, and that is the masked-cell
            // invariant being used: a quotient cannot exceed its dividend, and neither can a
            // bitwise result set a bit its operands did not have.
            IntOp::UDiv => a.unsigned_div(b),
            IntOp::URem => a.unsigned_rem(b),
            IntOp::And => a & b,
            IntOp::Or => a | b,
            IntOp::Xor => a ^ b,
            IntOp::Shl => double_shl(a, b, bits),
            IntOp::UShr => double_ushr(a, b, bits),
            // The signed forms have no 128-bit opcode to sweep, which the caller enforces by
            // sweeping this lane unsigned; naming them here keeps a new variant a compile error.
            IntOp::SAdd | IntOp::SSub | IntOp::SMul | IntOp::SDiv | IntOp::SRem | IntOp::SShr => {
                unreachable!("{op:?} has no 128-bit lane: MAX_LOWERED_SIGNED_BITS is 64")
            }
        }
    }

    /// The operand pairs to sweep for one operation at one width.
    ///
    /// A shift takes its right operand from the amount axis rather than the value one, because the
    /// interesting amounts are the ones around `bits` and a corner _value_ never lands there.
    ///
    /// Both operands are read at the same width, which is the case that has to be right: Noir's
    /// elaborator unifies the two operands of an infix operator, shifts included, so `bits` is also
    /// the amount's width for anything the frontend produces. It is also the only case the VM can
    /// distinguish.
    fn operand_pairs(op: IntOp, bits: usize) -> Vec<(u128, u128)> {
        let rhs = if op.is_shift() {
            corners::shift_amounts(bits, bits)
        } else {
            corners::values(bits)
        };
        corners::values(bits)
            .into_iter()
            .flat_map(|a| rhs.iter().map(move |b| (a, *b)))
            .collect()
    }

    /// The host word a model answer denotes.
    ///
    /// The VM holds its integers in `u64` frame cells and this sweep compares cells, so a pattern
    /// exists only for the length of a call into the model. It is the one evaluator whose value
    /// domain _is_ the machine's, which is why it carries no `IntBits` of its own.
    fn host(value: &IntBits) -> u128 {
        u128::try_from(value).expect("a pattern this sweep builds fits a host word")
    }

    /// `residue` on the host words either side of it.
    fn model(op: IntOp, bits: usize, a: u128, rhs_bits: usize, b: u128) -> Option<u128> {
        residue(
            op,
            &IntBits::from_u128(bits, a),
            &IntBits::from_u128(rhs_bits, b),
        )
        .map(|v| host(&v))
    }

    /// `IntBits::compare` on host words.
    fn compare(op: CmpOp, bits: usize, a: u128, b: u128) -> bool {
        IntBits::from_u128(bits, a).compare(op, &IntBits::from_u128(bits, b))
    }

    #[test]
    fn the_int_lane_agrees_with_the_model() {
        let mut checked = 0usize;

        for op in IntOp::ALL {
            // The reading is the operation's own, so the width set it admits comes off it rather
            // than off a sign swept beside it. One set covers the shifts too — see `lane_widths`.
            let sign = op.sign().unwrap_or(Sign::Unsigned);
            let widths = lane_widths(sign);

            for bits in widths {
                for (a, b) in operand_pairs(op, bits as usize) {
                    let (a, b) = (a as u64, b as u64);
                    let got = int_lane(op, bits, a, b);

                    // Rule 3, which binds on every input including the unspecified ones.
                    assert_eq!(
                        got & !cell_mask(bits),
                        0,
                        "{op:?} at {bits} bits left {a:#x} {b:#x} outside the width: {got:#x}"
                    );

                    // Rule 1, where the model has an opinion.
                    if let Some(want) = model(
                        op,
                        bits as usize,
                        u128::from(a),
                        bits as usize,
                        u128::from(b),
                    ) {
                        assert_eq!(
                            u128::from(got),
                            want,
                            "{op:?} at {bits} bits: {a:#x} {b:#x} gave {got:#x}, \
                             model says {want:#x}"
                        );
                        checked += 1;
                    }
                }
            }
        }

        // A sweep that agreed with the model on nothing would satisfy every assertion above, so the
        // count is part of the test rather than a diagnostic.
        assert!(
            checked > 25_000,
            "the sweep only reached {checked} specified points"
        );
    }

    #[test]
    fn the_int128_lane_agrees_with_the_model() {
        let mut checked = 0usize;

        // Unsigned only: `MAX_LOWERED_SIGNED_BITS` is 64, so no signed opcode reads a pattern
        // this wide and there is none to sweep. The filter is what the lane's own `unreachable!`
        // arms rely on.
        for op in IntOp::ALL.into_iter().filter(|op| !op.is_signed()) {
            // Every width the lane holds, not only its widest — see `double_lane_widths`.
            for bits in double_lane_widths() {
                for (a, b) in operand_pairs(op, bits as usize) {
                    let got = int128_lane(op, bits, Int128::from_u128(a), Int128::from_u128(b));

                    // Rule 3: whatever the answer is, it is inside the width.
                    assert_eq!(
                        got.to_u128() & !int_mask(bits),
                        0,
                        "{op:?} at {bits} bits left {a:#x} {b:#x} outside the width"
                    );

                    if let Some(want) = model(op, bits as usize, a, bits as usize, b) {
                        assert_eq!(
                            got.to_u128(),
                            want,
                            "{op:?} at {bits} bits: {a:#x} {b:#x} gave {:#x}, model says {want:#x}",
                            got.to_u128()
                        );
                        checked += 1;
                    }
                }
            }
        }

        assert!(
            checked > 2_000,
            "the sweep only reached {checked} specified points"
        );
    }

    #[test]
    fn the_comparison_opcodes_agree_with_the_model() {
        // `eq_int` and `ult_int` take no width — they read the raw cell — so the model is asked at
        // the width the operands were masked to, which is the same question.
        for bits in lane_widths(Sign::Unsigned) {
            for a in corners::values(bits as usize) {
                for b in corners::values(bits as usize) {
                    let (au, bu) = (a as u64, b as u64);
                    let at = bits as usize;

                    assert_eq!(au == bu, compare(CmpOp::Eq, at, a, b));
                    assert_eq!(au < bu, compare(CmpOp::ULt, at, a, b));

                    if corners::signed_width_ok(at) {
                        assert_eq!(
                            signed_cell(au, bits) < signed_cell(bu, bits),
                            compare(CmpOp::SLt, at, a, b),
                            "slt_int disagreed at {bits} bits on {a:#x} {b:#x}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_complement_opcode_agrees_with_the_model() {
        for bits in lane_widths(Sign::Unsigned) {
            for a in corners::values(bits as usize) {
                assert_eq!(
                    u128::from(cell_complement(a as u64, bits)),
                    host(&IntBits::from_u128(bits as usize, a).complement()),
                    "not_int disagreed at {bits} bits on {a:#x}"
                );
            }
        }

        // The double lane's complement, which the sweep above cannot reach: `IntOp` has no
        // complement variant, so `int128_lane` never sees one and `not_int128` would otherwise
        // have no conformance test in either lane.
        for bits in double_lane_widths() {
            for a in corners::values(bits as usize) {
                assert_eq!(
                    double_complement(Int128::from_u128(a), bits).to_u128(),
                    host(&IntBits::from_u128(bits as usize, a).complement()),
                    "not_int128 disagreed at {bits} bits on {a:#x}"
                );
            }
        }
    }
}
