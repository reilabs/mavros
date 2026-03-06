// Tests conditional value selection (Select instruction) inside a loop.
// Needed by: find_subarray_index pattern in passport verification -
//            searching for hash in SOD data with conditional mutation.
// The Select instruction is generated when mutable variables are
// conditionally updated inside loops. Currently unsupported in codegen.
fn main(arr: [Field; 8], target: Field) {
    let mut found_index: Field = 0;
    let mut found: bool = false;
    for i in 0..8 {
        if (!found) {
            if (arr[i] == target) {
                found_index = i as Field;
                found = true;
            }
        }
    }
    assert(found);
    assert_eq(found_index, 3);
}
