// PRE binding stability over the invariance closure, det-call form: t is the result of a
// deterministic call recomputed in the OUTER loop's body on every iteration, always to the
// same value. The call itself can never move (calls are not motion candidates), so no
// cross-site hoist cascade can ever rewrite t to an acyclically defined value -- under the
// old acyclic-definition gate the inner t * t was refused at every site, paying one witness
// product per INNER iteration. The invariance closure admits t (det call over stable args),
// and t * t hoists to the inner preheader: one product per OUTER iteration.
fn combine(x: Field, y: Field) -> Field {
    x * y + 1
}

fn main(x: Field, y: Field) {
    let mut acc: Field = 0;
    for _j in 0..10 {
        let t = combine(x, y);
        for _k in 0..20 {
            acc += t * t;
        }
    }
    assert(acc == 51200);
}
