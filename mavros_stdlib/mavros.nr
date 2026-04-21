#[foreign(unsafe_cast)]
pub fn unsafe_cast<T>(x: Field) -> T {}

#[foreign(spread_inner)]
fn spread_inner(value: u32, bits: u32) -> u64 {}

pub fn spread<let N: u32>(value: u32) -> u64 {
    spread_inner(value, N)
}

#[foreign(unspread_inner)]
fn unspread_inner(value: u64, bits: u32) -> (u32, u32) {}

pub fn unspread<let N: u32>(value: u64) -> (u32, u32) {
    unspread_inner(value, N)
}
