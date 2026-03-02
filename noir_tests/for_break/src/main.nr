// Tests conditional mutation inside a for loop (find-first pattern).
// Needed by: find_subarray_index() in passport utils - searches for
//            a hash pattern in SOD/eContent data.
// Generates Select instruction which is unsupported in codegen.
fn main(haystack: [u8; 16], needle: u8) {
    let mut found_index: u32 = 0;
    let mut found: bool = false;
    for i in 0..16 {
        if (!found) {
            if (haystack[i] == needle) {
                found_index = i;
                found = true;
            }
        }
    }
    assert(found);
    assert_eq(found_index, 5);
}
