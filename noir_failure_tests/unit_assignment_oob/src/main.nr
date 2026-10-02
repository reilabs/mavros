fn main(index: u32) {
    let mut array = [()];
    // A statement-shaped RHS must materialize unit before checking the store.
    array[index] = { assert(index != 0); };
}
