fn add1(x: Field) -> Field { x + 1 }
fn double(x: Field) -> Field { x * 2 }
fn square(x: Field) -> Field { x * x }

fn add1_u32(x: u32) -> u32 { x + 1 }

fn apply(f: fn(Field) -> Field, x: Field) -> Field { f(x) }
fn compose(f: fn(Field) -> Field, g: fn(Field) -> Field, x: Field) -> Field { f(g(x)) }

fn apply_twice(f: fn(Field) -> Field, x: Field) -> Field { f(f(x)) }
fn apply_n_times(f: fn(Field) -> Field, x: Field, n: u32) -> Field {
    let mut acc = x;
    for _ in 0..n {
        acc = f(acc);
    }
    acc
}

fn main(x: Field, y: Field, expected_dot: pub Field) {
    test_basic_map();
    test_map_with_capture(x);
    test_chained_maps();
    test_fold_sum();
    test_fold_product();
    test_fold_with_capture(y);
    test_reduce();
    test_function_pointers();
    test_composition();
    test_apply_n();
    test_nested_hof();
    test_closure_capture_multiple(x, y);
    test_mapi();
    test_map_fold_pipeline();
    test_closure_pipeline(x);
}

fn test_basic_map() {
    let arr: [Field; 4] = [1, 2, 3, 4];
    let doubled = arr.map(|n| n * 2);
    assert(doubled == [2, 4, 6, 8]);

    let incremented = arr.map(|n| n + 10);
    assert(incremented == [11, 12, 13, 14]);

    let squared = arr.map(|n| n * n);
    assert(squared == [1, 4, 9, 16]);
}

fn test_map_with_capture(x: Field) {
    let arr: [Field; 3] = [10, 20, 30];
    let offset = x;
    let shifted = arr.map(|n| n + offset);
    let shifted_sum = shifted.fold(0, |acc, n| acc + n);
    assert(shifted_sum == 60 + 3 * x);

    let scaled = arr.map(|n| n * x);
    let scaled_sum = scaled.fold(0, |acc, n| acc + n);
    assert(scaled_sum == 60 * x);
}

fn test_chained_maps() {
    let arr: [Field; 3] = [1, 2, 3];
    let result = arr.map(|n| n * 2).map(|n| n + 1);
    assert(result == [3, 5, 7]);

    let result2 = arr.map(|n| n + 1).map(|n| n * n).map(|n| n + 10);
    assert(result2 == [14, 19, 26]);
}

fn test_fold_sum() {
    let arr: [Field; 5] = [1, 2, 3, 4, 5];
    let sum = arr.fold(0, |acc, n| acc + n);
    assert(sum == 15);

    let arr2: [u32; 4] = [10, 20, 30, 40];
    let sum2 = arr2.fold(0, |acc: u32, n: u32| acc + n);
    assert(sum2 == 100);
}

fn test_fold_product() {
    let arr: [Field; 4] = [1, 2, 3, 4];
    let product = arr.fold(1, |acc, n| acc * n);
    assert(product == 24);

    let arr2: [Field; 5] = [1, 2, 3, 4, 5];
    let product2 = arr2.fold(1, |acc, n| acc * n);
    assert(product2 == 120);
}

fn test_fold_with_capture(y: Field) {
    let arr: [Field; 3] = [1, 2, 3];
    let bonus = y;
    let result = arr.fold(0, |acc, n| acc + n + bonus);
    assert(result == 6 + 3 * y);
}

fn test_reduce() {
    let arr: [Field; 4] = [1, 2, 3, 4];
    let sum = arr.reduce(|a, b| a + b);
    assert(sum == 10);

    let product = arr.reduce(|a, b| a * b);
    assert(product == 24);

    let arr2: [Field; 3] = [5, 3, 8];
    let sum_sq = arr2.map(|n| n * n).reduce(|a, b| a + b);
    assert(sum_sq == 98);
}

fn test_function_pointers() {
    assert(apply(add1, 10) == 11);
    assert(apply(double, 10) == 20);
    assert(apply(square, 5) == 25);

    let arr: [Field; 3] = [1, 2, 3];
    let incremented = arr.map(add1);
    assert(incremented == [2, 3, 4]);

    let doubled = arr.map(double);
    assert(doubled == [2, 4, 6]);

    let uarr: [u32; 3] = [10, 20, 30];
    let uinc = uarr.map(add1_u32);
    assert(uinc == [11, 21, 31]);
}

fn test_composition() {
    assert(compose(double, add1, 5) == 12);
    assert(compose(add1, double, 5) == 11);
    assert(compose(square, add1, 2) == 9);
    assert(compose(add1, square, 3) == 10);
}

fn test_apply_n() {
    assert(apply_n_times(add1, 10, 0) == 10);
    assert(apply_n_times(add1, 10, 5) == 15);
    assert(apply_n_times(double, 2, 3) == 16);

    assert(apply_twice(add1, 0) == 2);
    assert(apply_twice(double, 3) == 12);
    assert(apply_twice(square, 2) == 16);
}

fn test_nested_hof() {
    let arr: [Field; 4] = [1, 2, 3, 4];

    let sum_doubled = arr.map(|n| n * 2).fold(0, |acc, n| acc + n);
    assert(sum_doubled == 20);

    let product_incremented = arr.map(|n| n + 1).reduce(|a, b| a * b);
    assert(product_incremented == 120);

    let sum_x10 = arr.map(|n| n * 10).fold(0, |acc, n| acc + n);
    assert(sum_x10 == 100);

    let dot = arr.fold(0, |acc, n| acc + n * n);
    assert(dot == 30);
}

fn test_closure_capture_multiple(x: Field, y: Field) {
    let a = x;
    let b = y;
    let arr: [Field; 3] = [1, 2, 3];

    let result = arr.map(|n| n * a + b);
    let result_sum = result.fold(0, |acc, n| acc + n);
    assert(result_sum == 6 * a + 3 * b);

    let folded = arr.fold(a, |acc, n| acc + n * b);
    assert(folded == a + 6 * b);
}

fn test_mapi() {
    let arr: [Field; 4] = [10, 20, 30, 40];

    let indexed = arr.mapi(|i: u32, val: Field| val + i as Field);
    assert(indexed == [10, 21, 32, 43]);

    let weighted = arr.mapi(|i: u32, val: Field| val * (i as Field + 1));
    assert(weighted == [10, 40, 90, 160]);

    let weighted_sum = weighted.fold(0, |acc, n| acc + n);
    assert(weighted_sum == 300);
}

fn test_map_fold_pipeline() {
    let arr: [Field; 5] = [1, 2, 3, 4, 5];
    let result = arr
        .map(|n| n * 2)
        .map(|n| n + 3)
        .fold(0, |acc, n| acc + n);
    assert(result == 45);

    let arr2: [u32; 4] = [1, 2, 3, 4];
    let squares = arr2.map(|n| n * n);
    let sq_sum = squares.fold(0, |acc: u32, n: u32| acc + n);
    assert(sq_sum == 30);

    let uarr: [u32; 3] = [1, 2, 3];
    let incremented = uarr.map(add1_u32);
    assert(incremented == [2, 3, 4]);
    let u_sum = incremented.fold(0, |acc: u32, n: u32| acc + n);
    assert(u_sum == 9);
}

fn test_closure_pipeline(x: Field) {
    let arr: [Field; 4] = [1, 2, 3, 4];
    let factor = x;

    let result = arr
        .map(|n| n * factor)
        .map(|n| n + 1)
        .fold(factor, |acc, n| acc + n);
    assert(result == 11 * x + 4);

    let indexed = arr.mapi(|i: u32, val: Field| val * factor + i as Field);
    let idx_sum = indexed.fold(0, |acc, n| acc + n);
    assert(idx_sum == 10 * x + 6);
}
