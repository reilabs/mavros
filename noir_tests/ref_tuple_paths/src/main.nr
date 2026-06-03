pub struct Leaf {
    pub a: Field,
    pub b: Field,
}

pub struct Pair {
    pub left: Leaf,
    pub right: Leaf,
}

fn owned_mut_tuple_path(a: Field, b: Field, c: Field, d: Field) -> Field {
    let mut tuple = (Leaf { a, b }, Leaf { a: c, b: d });

    let before = tuple.0.a + tuple.1.b;
    tuple.0.b = tuple.0.a + tuple.1.a;
    tuple.1 = Leaf { a: tuple.0.b, b: tuple.1.b + 1 };

    before + tuple.0.b + tuple.1.a + tuple.1.b
}

fn explicit_mut_ref_path(pair: &mut Pair) -> Field {
    let value = (*pair).left.a + (*pair).right.b;
    (*pair).right.a = value;
    (*pair).right.a
}

fn implicit_mut_ref_path(pair: &mut Pair) -> Field {
    pair.left.a + pair.right.b
}

fn implicit_immut_ref_path(pair: &Pair) -> Field {
    pair.left.b + pair.right.a
}

fn explicit_tuple_of_refs_path(refs: (&Leaf, &mut Leaf)) -> Field {
    (*refs.0).a + (*refs.1).b
}

fn implicit_tuple_of_refs_path(refs: (&Leaf, &mut Leaf)) -> Field {
    refs.0.a + refs.1.b
}

fn explicit_ref_to_tuple_of_refs_path(refs: &(&Leaf, &Leaf)) -> Field {
    (*(*refs).0).a + (*(*refs).1).b
}

fn implicit_ref_to_tuple_of_refs_path(refs: &(&Leaf, &Leaf)) -> Field {
    refs.0.a + refs.1.b
}

fn read_leaf_by_ref(leaf: &Leaf) -> Field {
    leaf.a + leaf.b
}

fn mutate_leaf_by_ref(leaf: &mut Leaf, delta: Field) -> Field {
    leaf.b = leaf.a + delta;
    leaf.b
}

fn main(a: Field, b: Field, c: Field, d: Field) {
    let owned = owned_mut_tuple_path(a, b, c, d);
    assert_eq(owned, 18);

    let mut pair = Pair {
        left: Leaf { a, b },
        right: Leaf { a: c, b: d },
    };

    let direct_implicit_ref = (&mut pair).left.a;
    assert_eq(direct_implicit_ref, 1);

    let explicit_mut = explicit_mut_ref_path(&mut pair);
    assert_eq(explicit_mut, 5);

    let implicit_mut = implicit_mut_ref_path(&mut pair);
    assert_eq(implicit_mut, 5);

    let implicit_immut = implicit_immut_ref_path(&pair);
    assert_eq(implicit_immut, 7);

    let mut left = Leaf { a, b };
    let mut right = Leaf { a: c, b: d };
    let tuple_refs = (&left, &mut right);

    let explicit_tuple_refs = explicit_tuple_of_refs_path(tuple_refs);
    assert_eq(explicit_tuple_refs, 5);

    let implicit_tuple_refs = implicit_tuple_of_refs_path(tuple_refs);
    assert_eq(implicit_tuple_refs, 5);

    let immut_tuple_refs = (&left, &right);
    let explicit_ref_tuple_refs = explicit_ref_to_tuple_of_refs_path(&immut_tuple_refs);
    assert_eq(explicit_ref_tuple_refs, 5);

    let implicit_ref_tuple_refs = implicit_ref_to_tuple_of_refs_path(&immut_tuple_refs);
    assert_eq(implicit_ref_tuple_refs, 5);

    let left_sum = read_leaf_by_ref(&pair.left);
    assert_eq(left_sum, 3);

    let right_sum = mutate_leaf_by_ref(&mut pair.right, left_sum);
    assert_eq(right_sum, 8);
}
