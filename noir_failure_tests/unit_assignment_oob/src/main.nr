fn main(index: u32) {
    let mut array = [()];
    // Storing a zero-sized value must still check the index.
    array[index] = ();
}
