struct P {}

fn main(idx: u32) {
    let mut p = P {};
    let refs: [&mut P] = @[&mut p, &mut p, &mut p];
    let _ref = refs[idx]; // in bounds
    let (refs, _last_ref) = refs.pop_back(); // 2
    assert(refs.len() == 2);
}
