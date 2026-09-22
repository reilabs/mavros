// Known gap: return reachability accepts this because one path returns.
// Keep the non-returning path inactive during execution.
unconstrained fn spin() { spin(); }
fn main(x: Field) { if x == 0 { unsafe { spin(); } } assert(x != 5); }
