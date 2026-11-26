// Guarded pushes of a multi-leaf struct element onto a witness-length slice: purify batches
// one pushed value per leaf, so each push writes both leaves at the witness cursor (back) or
// the physical front. Expected values are public inputs so the circuit is world-parametric.
struct Pt {
    x: Field,
    y: Field,
}

fn main(w: u8, x0: pub u8, y0: pub u8, x1: pub u8, y1: pub u8, elen: pub u8) {
    let w = w as Field;
    let x0 = x0 as Field;
    let y0 = y0 as Field;
    let x1 = x1 as Field;
    let y1 = y1 as Field;
    let elen = elen as Field;

    let mut s: [Pt] = [Pt { x: 10, y: 11 }];
    if w == 1 {
        s = s.push_back(Pt { x: 20, y: 21 });
        s = s.push_front(Pt { x: 1, y: 2 });
    }
    assert_eq(s.len() as Field, elen);
    assert_eq(s[0].x, x0);
    assert_eq(s[0].y, y0);
    assert_eq(s[1].x, x1);
    assert_eq(s[1].y, y1);
}
