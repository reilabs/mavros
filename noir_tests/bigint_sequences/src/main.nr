fn twice<let N: u32>(x: u<N>) -> u<N> { x + x }

fn main(x: u64, index: u32) -> pub [u64; 3] {
    let high: u257 = 1 << 200;
    let a = high + (x as u257);
    let mut values = [a, twice(a)];
    values[index] = values[index] + 1;
    let vector = values.as_vector().push_back(a + 7);
    let (rest, last) = vector.pop_back();
    let selected = rest[index];
    [(selected >> 200) as u64, selected as u64, last as u64]
}
