unconstrained fn down(n: u32) -> u32 { if n == 0 { 7 } else { down(n - 1) } }
fn main(n: u32) { let result = unsafe { down(n) }; assert(result == 7); }
