// Exercises SCS's reclamation of blocks orphaned by its own conditional `JmpIf` folds, at the
// pre-WTI run (the one place a leftover orphan reaches an all-blocks consumer: the ArraySroa
// cluster's PointsTo rebuild types every block against reachable-only type info).
//
// The fold must first become possible at the pre-WTI SCS and no earlier: conditional facts track
// raw value pairs, so while `c` reaches the branch comparison and the establishing assert through
// two separate loads of a mutable local, the post-dominating assert says nothing about the
// comparison feeding the branch. The initial Mem2Reg promotes the local and unifies the loads;
// the SCS run directly after it then folds the comparison to true through the anticipated-equal
// channel (both comparison sides must be runtime values: a constant side lands in the
// anticipated-const channel, which only rewrites operands and leaves the terminator fold to the
// next run, whose solver prunes cleanly). The `JmpIf` becomes a `Jmp` and the else arm is
// orphaned in the same run. The casts and the array access in that arm are exactly the shapes
// the PointsTo builder asks the type table about.
fn main(a: Field, b: pub Field) {
    let mut c = a;
    c = c + b;
    let d = a * 2;
    let mut out = a;
    if c == d {
        out = a + 1;
    } else {
        let t = (a as u32 + 1) as Field;
        out = [t, b][0] * 2;
    }
    assert(c == d);
    assert(out == a + 1);
}
