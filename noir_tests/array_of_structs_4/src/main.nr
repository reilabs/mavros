pub struct G {
    val: Field,
}

pub struct F {
    pub g: G,
}

// E 
pub struct E {
    pub f: F,
}

// D is the deepest level (level 4)
pub struct D {
    pub d_field: Field,
    pub e_arr: [E; 2],
}

// C contains D (level 3)
pub struct C {
    pub d: D,
    pub c_field: Field,
}

// B contains an ARRAY of C structs (level 2)
// This tests arrays of structs within structs
pub struct B {
    pub c_arr: [C; 2],
    pub b_field: Field,
}

// A is the top level (level 1)
// Contains B and also has an array of D structs
pub struct A {
    pub a_field: Field,
    pub b: B,
    pub d_arr: [D; 2],
}

// Structure summary:
// - 7 struct types with deep nesting (A -> B -> C -> D -> E -> F -> G)
// - Arrays of structs: B.c_arr: [C; 2], A.d_arr: [D; 2], D.e_arr: [E; 2]
// - 16 fields per A element, 32 total with [A; 2]

fn main(x: [A; 2]) {
    // Test first A element (x[0]) - values 0-15
    assert_eq(x[0].a_field, 0);

    // x[0].b.c_arr[0] (first C in B's array)
    assert_eq(x[0].b.c_arr[0].d.d_field, 1);
    assert_eq(x[0].b.c_arr[0].d.e_arr[0].f.g.val, 2);
    assert_eq(x[0].b.c_arr[0].d.e_arr[1].f.g.val, 3);
    assert_eq(x[0].b.c_arr[0].c_field, 4);

    // x[0].b.c_arr[1] (second C in B's array)
    assert_eq(x[0].b.c_arr[1].d.d_field, 5);
    assert_eq(x[0].b.c_arr[1].d.e_arr[0].f.g.val, 6);
    assert_eq(x[0].b.c_arr[1].d.e_arr[1].f.g.val, 7);
    assert_eq(x[0].b.c_arr[1].c_field, 8);

    assert_eq(x[0].b.b_field, 9);

    // x[0].d_arr (array of D structs directly in A)
    assert_eq(x[0].d_arr[0].d_field, 10);
    assert_eq(x[0].d_arr[0].e_arr[0].f.g.val, 11);
    assert_eq(x[0].d_arr[0].e_arr[1].f.g.val, 12);
    assert_eq(x[0].d_arr[1].d_field, 13);
    assert_eq(x[0].d_arr[1].e_arr[0].f.g.val, 14);
    assert_eq(x[0].d_arr[1].e_arr[1].f.g.val, 15);

    // Test second A element (x[1]) - values 16-31
    assert_eq(x[1].a_field, 16);

    // x[1].b.c_arr[0]
    assert_eq(x[1].b.c_arr[0].d.d_field, 17);
    assert_eq(x[1].b.c_arr[0].d.e_arr[0].f.g.val, 18);
    assert_eq(x[1].b.c_arr[0].d.e_arr[1].f.g.val, 19);
    assert_eq(x[1].b.c_arr[0].c_field, 20);

    // x[1].b.c_arr[1]
    assert_eq(x[1].b.c_arr[1].d.d_field, 21);
    assert_eq(x[1].b.c_arr[1].d.e_arr[0].f.g.val, 22);
    assert_eq(x[1].b.c_arr[1].d.e_arr[1].f.g.val, 23);
    assert_eq(x[1].b.c_arr[1].c_field, 24);

    assert_eq(x[1].b.b_field, 25);

    // x[1].d_arr
    assert_eq(x[1].d_arr[0].d_field, 26);
    assert_eq(x[1].d_arr[0].e_arr[0].f.g.val, 27);
    assert_eq(x[1].d_arr[0].e_arr[1].f.g.val, 28);
    assert_eq(x[1].d_arr[1].d_field, 29);
    assert_eq(x[1].d_arr[1].e_arr[0].f.g.val, 30);
    assert_eq(x[1].d_arr[1].e_arr[1].f.g.val, 31);
}
