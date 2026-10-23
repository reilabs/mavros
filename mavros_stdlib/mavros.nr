#[foreign(unsafe_cast)]
pub fn unsafe_cast<T>(x: Field) -> T {}

#[foreign(spread_inner)]
fn spread_inner(value: u32, bits: u32) -> u64 {}

/// Interleaves the low `N` bits of `value` with zeros, so that bit `i` lands on bit `2i`. `N` is
/// 1 to 32.
///
/// On a witness, a set bit above the `N` read fails the proof, which makes the call a range check.
/// A value known at compile time is masked to its low `N` bits instead, so a call the proof
/// rejects for a witness can be accepted once the optimiser has folded that witness to a constant.
pub fn spread<let N: u32>(value: u32) -> u64 {
    spread_inner(value, N)
}

#[foreign(unspread_inner)]
fn unspread_inner(value: u64, bits: u32) -> (u32, u32) {}

/// Splits the low `2N` bits of `value` into its odd-indexed and even-indexed bits, each an `N`-bit
/// stream: `unspread::<N>(spread::<N>(x))` is `(0, x)`. `N` is 1 to 32.
///
/// On a witness, a set bit above the `2N` read fails the proof. A value known at compile time is
/// masked to its low `2N` bits instead, as for [`spread`].
pub fn unspread<let N: u32>(value: u64) -> (u32, u32) {
    unspread_inner(value, N)
}

pub mod replacements;
pub mod vendor;
