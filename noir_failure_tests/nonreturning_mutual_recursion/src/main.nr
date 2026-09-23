unconstrained fn first() { second(); }
unconstrained fn second() { first(); }
fn main() { unsafe { first(); } }
