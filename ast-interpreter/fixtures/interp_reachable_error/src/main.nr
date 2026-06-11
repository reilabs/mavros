// Negative fixture for the validation frontend's error boundary. `main`'s body has a
// type error in REACHABLE code (a `u64` bound to a `bool`). The validation frontend
// tolerates elaboration errors confined to *unreached* code (the bn254 crypto stdlib under
// Goldilocks), but a reachable error like this must NOT yield a clean mono-AST — otherwise
// the interpreter would validate a silently-miscompiled program. See
// `rejects_reachable_type_error` in ast_interpreter/tests.rs.
fn main(x: u64) -> pub u64 {
    let _b: bool = x; // type error: u64 is not bool, and main reaches it
    x
}
