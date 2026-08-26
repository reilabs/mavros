// Adapted compatibility subset of zkpassport/noir-bignum v0.10.0-1 and
// zkpassport/noir_bigcurve v0.14.0-1. Kept inside the injected std crate so low-level foreign
// replacements follow Mavros' existing pure-Noir path. See THIRD_PARTY_NOTICES.md.

pub mod bignum {

    use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
    pub mod constants {
        use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
        pub(crate) global TWO_POW_60: u128 = 0x1000000000000000;
        pub(crate) global TWO_POW_120: u128 = 0x1000000000000000000000000000000;
        pub(crate) global TWO_POW_126: Field = 0x40000000000000000000000000000000;
        pub(crate) global TWO_POW_240: Field =
            0x1000000000000000000000000000000000000000000000000000000000000;
        pub(crate) global TWO_POW_246: Field =
            0x40000000000000000000000000000000000000000000000000000000000000;
        pub(crate) global GRUMPKIN_MODULUS: [u128; 3] =
            [0x33e84879b9709143e1f593f0000001, 0x4e72e131a029b85045b68181585d28, 0x3064];
    }

    pub mod params {

        use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
        use crate::mavros::vendor::bignum::constants::TWO_POW_120;

        pub struct BigNumParams<let N: u32, let MOD_BITS: u32> {
            pub has_multiplicative_inverse: bool,

            // `modulus`: all `BigNum` operations are evaluated modulo this value
            pub modulus: [u128; N],

            // `double_modulus`: used when performing negations and subtractions
            // Note: we borrow 1 from the last limb. This is for easing up the operations that might underflow
            pub double_modulus: [u128; N],

            // `redc_param` used for __barrett_reduction. See https://en.wikipedia.org/wiki/Barrett_reduction
            // and `__barrett_reduction` in `fns/unconstrained_helpers.nr`
            pub redc_param: [u128; N],
        }

        impl<let N: u32, let MOD_BITS: u32> BigNumParams<N, MOD_BITS> {
            pub fn new(
                has_multiplicative_inverse: bool,
                modulus: [u128; N],
                redc_param: [u128; N],
            ) -> Self {
                Self {
                    has_multiplicative_inverse,
                    modulus,
                    double_modulus: get_double_modulus(modulus),
                    redc_param,
                }
            }
        }

        impl<let N: u32, let MOD_BITS: u32> crate::cmp::Eq for BigNumParams<N, MOD_BITS> {
            fn eq(self, other: Self) -> bool {
                (self.has_multiplicative_inverse == other.has_multiplicative_inverse)
                    & (self.modulus == other.modulus)
                    & (self.double_modulus == other.double_modulus)
                    & (self.redc_param == other.redc_param)
            }
        }

        fn get_double_modulus<let N: u32>(modulus: [u128; N]) -> [u128; N] {
            let mut result: [u128; N] = [0; N];
            let mut carry: u128 = 0;
            for i in 0..N {
                let mut add_term = (modulus[i] + modulus[i] + carry);
                carry = add_term >> 120;
                add_term = add_term - (carry << 120);
                result[i] = add_term;
            }

            result[0] += TWO_POW_120;
            for i in 1..N - 1 {
                result[i] += (TWO_POW_120 - 1);
            }
            result[N - 1] -= 1;
            result
        }
    }

    pub mod utils {

        use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
        pub mod split_bits {
            use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
            use crate::mavros::vendor::bignum::constants::TWO_POW_120;

            /// Split the Field value into two 120-bit limbs (unconstrained)
            ///
            /// Here we're taking advantage of truncating 120 bit limbs from the input field
            /// and then subtracting them from the input such that the field division is equivalent
            /// to integer division.
            ///
            /// We return the lower 120-bit limb as a `u128` value,
            /// and the upper limbs as a `Field`, to avoid unnecessary conversions
            /// and potential overflows
            pub(crate) unconstrained fn __split_120_bits(mut x: Field) -> (u128, Field) {
                let low: u128 = (x as u128) % TWO_POW_120;
                let high: Field = ((x - low as Field) / TWO_POW_120 as Field);
                (low, high)
            }

            /// Normalize an array of Field values into 120-bit limbs (unconstrained)
            ///
            /// Each Field element is split into two parts modulo 2^{120}
            /// The overflow from the lower limbs is carried into the higher limbs
            pub(crate) unconstrained fn __normalize_limbs<let N: u32>(input: [Field; N]) -> [u128; N] {
                let mut normalized: [u128; N] = [0; N];
                let mut next: Field = input[0];
                for i in 0..(N - 1) {
                    let (lo, hi): (u128, Field) = __split_120_bits(next);
                    normalized[i] = lo;
                    next = input[i + 1] + hi;
                }
                let (lo, hi): (u128, Field) = __split_120_bits(next);
                normalized[N - 1] = lo;

                // non-zero final carry <=> normalized value overflows the array length
                assert(hi == 0);

                normalized
            }
        }

        pub mod msb {

            use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
            use crate::ops::WrappingMul;

            global MUL_DE_BRUIJN_BIT: [u32; 128] = [
                1, 14, 2, 15, 26, 20, 3, 16, 68, 80, 27, 21, 56, 50, 4, 17, 65, 96, 69, 81, 105, 99, 28, 22, 86,
                90, 57, 51, 72, 42, 5, 126, 18, 66, 48, 94, 97, 84, 70, 124, 82, 122, 106, 100, 114, 108, 29,
                23, 77, 102, 87, 91, 119, 116, 58, 52, 61, 110, 73, 37, 43, 31, 6, 127, 13, 25, 19, 67, 79, 55,
                49, 64, 95, 104, 98, 85, 89, 71, 41, 125, 47, 93, 83, 123, 121, 113, 107, 76, 101, 118, 115, 60,
                109, 36, 30, 12, 24, 78, 54, 63, 103, 88, 40, 46, 92, 120, 112, 75, 117, 59, 35, 11, 53, 62, 39,
                45, 111, 74, 34, 10, 38, 44, 33, 9, 32, 8, 7, 128,
            ];

            /// Get the most significant bit position of a `val` (unconstrained)
            ///
            /// Bit hack that uses De Bruijn sequence to calculate msb position in log(N)
            /// See [IntegerLog2DeBruijn](https://graphics.stanford.edu/~seander/bithacks.html#IntegerLogDeBruijn)
            pub(crate) unconstrained fn get_msb(x: u128) -> u32 {
                let result: u32 = if x == 0 {
                    0
                } else {
                    let mut v: u128 = x;
                    v |= v >> 1;
                    v |= v >> 2;
                    v |= v >> 4;
                    v |= v >> 8;
                    v |= v >> 16;
                    v |= v >> 32;
                    v |= v >> 64;
                    let index: u128 = (v.wrapping_mul(0x1FC10C2FBCF471B913B14CD2595D6D5)) >> 121;
                    MUL_DE_BRUIJN_BIT[index as u32]
                };
                result
            }
        }

        pub mod map {

            use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
            // Copied from crate::array, because I couldn't figure out how to import the `map` method of the weird trait for an array.
            // And the reason I wanted direct access to it, is because I couldn't figure out how to implement a double map.
            pub(crate) fn map<T, let N: u32, U, Env>(arr: [T; N], f: fn[Env](T) -> U) -> [U; N] {
                let mut ret: [U; N] = crate::mem::zeroed();

                for i in 0..N {
                    ret[i] = f(arr[i]);
                }

                ret
            }

            /// reverse an array
            pub(crate) fn invert_array<T, let M: u32>(array: [T; M]) -> [T; M] {
                let mut ret: [T; M] = crate::mem::zeroed();

                for i in 0..M {
                    ret[i] = array[M - i - 1];
                }
                ret
            }
        }

    }

    pub mod fns {

        use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
        pub mod unconstrained_ops {
            use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
            // This file contains the unconstrained operations that are used directly by BigNum class

            use crate::mavros::vendor::bignum::fns::constrained_ops::derive_from_seed;

            use crate::mavros::vendor::bignum::fns::unconstrained_helpers::{
                __barrett_reduction, __get_bit, __get_msb, __half_mod_odd, __helper_add, __helper_mul,
                __helper_sub, __increment, __is_even, __one, __primitive_root_log_size, __quadratic_non_residue,
                __shl, __shr, __shr1, __tonelli_shanks_sqrt_find_i,
            };

            use crate::mavros::vendor::bignum::constants::TWO_POW_120;

            use crate::mavros::vendor::bignum::params::BigNumParams;

            // ------------------------------ DERIVATION FUNCTIONS ------------------------------

            /// Deterministically derives a `BigNum` from a seed value (unconstrained)
            ///
            /// Takes a seed byte array and generates a `BigNum` in the range [0, modulus-1].
            ///
            /// See more information in `constrained_ops.nr`: `derive_from_seed`
            pub unconstrained fn __derive_from_seed<let N: u32, let MOD_BITS: u32, let SeedBytes: u32>(
                params: BigNumParams<N, MOD_BITS>,
                seed: [u8; SeedBytes],
            ) -> [u128; N] {
                derive_from_seed::<N, MOD_BITS, SeedBytes>(params, seed)
            }

            // ------------------------------ COMPARISON FUNCTIONS ------------------------------

            /// Compare two limb arrays for equality (unconstrained)
            pub unconstrained fn __eq<let N: u32>(lhs: [u128; N], rhs: [u128; N]) -> bool {
                lhs == rhs
            }

            /// Compare a limb array to a zero array (unconstrained)
            pub unconstrained fn __is_zero<let N: u32>(limbs: [u128; N]) -> bool {
                limbs == [0; N]
            }

            /// Compare two little-endian limb arrays for `lhs >= rhs` over integers (unconstrained)
            ///
            /// Starts from the most significant limb (`N - 1`) and returns true
            /// if `lhs` is greater or equal to `rhs`
            pub(crate) unconstrained fn __gte<let N: u32>(lhs: [u128; N], rhs: [u128; N]) -> bool {
                let mut result: bool = true;
                for i in 0..N {
                    let idx: u32 = N - 1 - i;
                    if (lhs[idx] != rhs[idx]) {
                        result = lhs[idx] > rhs[idx];
                        break;
                    }
                }
                result
            }

            // ------------------------------ ARITHMETIC FUNCTIONS ------------------------------

            /// Negates a `BigNum` value, returning `m - x` (unconstrained)
            ///
            /// ## Note
            /// The input is assumed to be less than modulus
            pub unconstrained fn __neg<let N: u32>(modulus: [u128; N], limbs: [u128; N]) -> [u128; N] {
                __helper_sub(modulus, limbs)
            }

            /// Adds two `BigNum` values with modular reduction (unconstrained)
            ///
            /// Sums the limbs one by one, keeping the carry.
            /// In case the result overflows the modulus, the modulus is subtracted once
            ///
            /// ## Note
            /// The `carry` must be `0` at the end of the loop.
            /// No explicit assertion is made, as this condition is validated during evaluation.
            pub unconstrained fn __add<let N: u32>(
                modulus: [u128; N],
                lhs: [u128; N],
                rhs: [u128; N],
            ) -> [u128; N] {
                let mut result: [u128; N] = [0; N];
                let mut carry: u128 = 0;
                let mask: u128 = TWO_POW_120 - 1;

                for i in 0..N {
                    let add_term: u128 = (lhs[i] + rhs[i] + carry);
                    carry = add_term >> 120;
                    result[i] = add_term & mask;
                }

                // check if the result is greater than the modulus
                if __gte(result, modulus) {
                    __helper_sub(result, modulus)
                } else {
                    result
                }
            }

            /// Subtracts two `BigNum` values with modular reduction (unconstrained).
            ///
            /// Computes `x + (m - y)` (mod m)
            pub unconstrained fn __sub<let N: u32>(
                modulus: [u128; N],
                lhs: [u128; N],
                rhs: [u128; N],
            ) -> [u128; N] {
                __add(modulus, lhs, __neg(modulus, rhs))
            }

            /// Multiply `x` and `y` and reduce via Barrett, returning (Q, R) (unconstrained).
            ///
            /// For `BigNum` values `x` and `y` compute (`Q`, `R`) such that:
            ///      x * y = R + Q * m, 0 <= R < m
            /// See `__barrett_reduction` for details.
            pub(crate) unconstrained fn __mul_with_quotient<let N: u32, let MOD_BITS: u32>(
                params: BigNumParams<N, MOD_BITS>,
                lhs: [u128; N],
                rhs: [u128; N],
            ) -> ([u128; N], [u128; N]) {
                let to_reduce: [u128; N * 2] = __helper_mul(lhs, rhs);
                let (q, r): ([u128; N], [u128; N]) =
                    __barrett_reduction(to_reduce, params.redc_param, MOD_BITS, params.modulus);
                (q, r)
            }

            /// Multiplies two `BigNum` values with modular reduction (unconstrained).
            pub unconstrained fn __mul<let N: u32, let MOD_BITS: u32>(
                params: BigNumParams<N, MOD_BITS>,
                lhs: [u128; N],
                rhs: [u128; N],
            ) -> [u128; N] {
                __mul_with_quotient::<N, MOD_BITS>(params, lhs, rhs).1
            }

            /// Squares a `BigNum` value with modular reduction (unconstrained).
            pub unconstrained fn __sqr<let N: u32, let MOD_BITS: u32>(
                params: BigNumParams<N, MOD_BITS>,
                val: [u128; N],
            ) -> [u128; N] {
                __mul_with_quotient::<N, MOD_BITS>(params, val, val).1
            }

            /// Modular exponentiation via square-and-multiply. LSB-first (unconstrained).
            ///
            /// Computes `x^e mod m`
            ///
            /// ## Note
            /// For the loop, we are using `MOD_BITS` instead of `__get_msb`
            /// because it is much much cheaper
            pub unconstrained fn __pow<let N: u32, let MOD_BITS: u32>(
                params: BigNumParams<N, MOD_BITS>,
                val: [u128; N],
                exponent: [u128; N],
            ) -> [u128; N] {
                let mut accumulator: [u128; N] = __one();
                let mut x: [u128; N] = val;
                let num_bits: u32 = MOD_BITS + 1;

                for i in 0..num_bits {
                    if __get_bit(exponent, i) {
                        accumulator = __mul::<N, MOD_BITS>(params, accumulator, x);
                    }
                    x = __sqr::<N, MOD_BITS>(params, x);
                }
                accumulator
            }

            /// Given a `BigNum` value `x` compute x^{-1} (mod MOD) (unconstrained)
            ///
            /// This is an alternative to Fermat inversion(`__pow_invmod`) (`val^(modulus-2) mod MOD`) that uses
            /// the (extended) binary GCD / Stein algorithm.
            /// See [Binary GCD algorithm](https://en.wikipedia.org/wiki/Binary_GCD_algorithm)
            ///
            /// ## Note
            /// - `modulus` must be odd (required by `__half_mod_odd`)
            /// - `gcd(val, modulus) == 1` (i.e. `val` is invertible modulo `modulus`)
            pub unconstrained fn __invmod<let N: u32, let MOD_BITS: u32>(
                params: BigNumParams<N, MOD_BITS>,
                val: [u128; N],
            ) -> [u128; N] {
                assert(params.has_multiplicative_inverse);
                if (__is_zero(val) | __eq(val, params.modulus)) {
                    [0; N]
                } else {
                    let mut u: [u128; N] = val;
                    let mut v: [u128; N] = params.modulus;

                    let one: [u128; N] = __one::<N>();

                    let mut x1: [u128; N] = one;
                    let mut x2: [u128; N] = [0; N];

                    while (!__eq(u, one) & !__eq(v, one)) {
                        // Get rid of the even part of u
                        while __is_even(u) {
                            u = __shr1(u);
                            x1 = __half_mod_odd(params.modulus, x1);
                        }

                        // Get rid of the even part of v
                        while __is_even(v) {
                            v = __shr1(v);
                            x2 = __half_mod_odd(params.modulus, x2);
                        }

                        // Update the intermediate values after both (u, v) are odd
                        if __gte(u, v) {
                            u = __helper_sub(u, v);
                            x1 = __sub(params.modulus, x1, x2);
                        } else {
                            v = __helper_sub(v, u);
                            x2 = __sub(params.modulus, x2, x1);
                        }
                    }

                    if __eq(u, one) {
                        x1
                    } else {
                        x2
                    }
                }
            }

            /// Divides two `BigNum` values with modular reduction (unconstrained).
            ///
            /// Computes `x * y^{-1}` (mod m)
            ///
            /// ## Note
            /// The divisor must be nonzero
            /// No explicit assertion is made, as this condition is validated during evaluation
            pub unconstrained fn __div<let N: u32, let MOD_BITS: u32>(
                params: BigNumParams<N, MOD_BITS>,
                numerator: [u128; N],
                divisor: [u128; N],
            ) -> [u128; N] {
                let inv_divisor: [u128; N] = __invmod::<N, MOD_BITS>(params, divisor);
                __mul::<N, MOD_BITS>(params, numerator, inv_divisor)
            }

            /// Given the `BigNum` inputs `x, y`, compute integer division x / y (unconstrained)
            ///
            /// This function implements binary long division and outputs (`quotient`, `remainder`) such that:
            ///   1. floor(numerator / divisor) = quotient
            ///   2. numerator % divisor = remainder
            ///   3. divisor * quotient + remainder = numerator
            ///
            /// ## Note
            /// The divisor must be nonzero
            /// No explicit assertion is made, as this condition is validated during evaluation
            pub unconstrained fn __udiv_mod<let N: u32>(
                numerator: [u128; N],
                divisor: [u128; N],
            ) -> ([u128; N], [u128; N]) {
                let mut quotient: [u128; N] = [0; N];
                let mut remainder: [u128; N] = numerator;
                let b: [u128; N] = divisor;

                let numerator_msb: u32 = __get_msb(numerator);
                let divisor_msb: u32 = __get_msb(divisor);
                if divisor_msb > numerator_msb {
                    ([0; N], numerator)
                } else {
                    let bit_difference: u32 = __get_msb(remainder) - __get_msb(divisor);
                    let mut divisor: [u128; N] = __shl(divisor, bit_difference);
                    let mut accumulator: [u128; N] = __shl(__one(), bit_difference);

                    // The same as divisor > remainder
                    if (__gte(divisor, __increment(remainder))) {
                        divisor = __shr1(divisor);
                        accumulator = __shr1(accumulator);
                    }

                    for _ in 0..(N * 120) {
                        if (__gte(remainder, b) == false) {
                            break;
                        }
                        // we've shunted 'divisor' up to have the same bit length as our remainder.
                        // If remainder >= divisor, then a is at least '1 << bit_difference' multiples of b
                        if (__gte(remainder, divisor)) {
                            remainder = __helper_sub(remainder, divisor);
                            // we can use OR here instead of +, as
                            // accumulator is always a nice power of two
                            quotient = __helper_add(quotient, accumulator);
                        }
                        divisor = __shr1(divisor);
                        accumulator = __shr1(accumulator);
                    }

                    (quotient, remainder)
                }
            }

            /// Batch modular inversion of `BigNum` values in an array (unconstrained)
            ///
            /// Given values v[0..M), returns inv[0...M) with inv[i] = v[i]^{-1} (mod m)
            ///
            /// We use the Montgomery trick:
            /// First we compute the partial products:
            ///         T0 = 1
            ///         T1 = v1,
            ///         T2 = v1 * v2,
            ///         ...
            ///         T_{m - 1} = T_{m - 2} * v_{m - 1} = v1 * ... * v_{m - 1}
            ///
            ///         P = T_{m-1} * v_m = v1 * v2 * ... * v_m
            ///
            /// Then we calculate a single inverse P^-1 = v1^-1 * v2^-1 * ... * v_m^-1
            /// Finally we compute
            ///         v_m^-1       = (P^-1 * T_{m-1})
            ///         v_{m - 1}^-1 = (P^-1 * v_m * T_{m - 2})
            ///         ....
            ///         v_2^-1       = (P^-1 * v_m * ... * v_3 * T_1)
            ///         v_1^-1       = (P^-1 * v_m * ... * v_2 * T_0)
            ///
            /// ## Note
            /// Zero elements are allowed and are left unchanged in the resulting array
            ///
            /// This interacts poorly with `neg(zero)`:
            /// Calling `neg` on `zero` yields `modulus` rather than `0`.
            /// A value in this form will **not** satisfy
            /// the `__is_zero` check and will lead to incorrect results.
            ///
            /// This edge case should be rare, but it's worth keeping in mind when
            /// composing operations or debugging unexpected behavior
            pub(crate) unconstrained fn batch_invert<let N: u32, let MOD_BITS: u32, let M: u32>(
                params: BigNumParams<N, MOD_BITS>,
                vals: [[u128; N]; M],
            ) -> [[u128; N]; M] {
                let mut accumulator: [u128; N] = __one();
                let mut temporaries: [[u128; N]; M] = [[0; N]; M];

                for i in 0..M {
                    temporaries[i] = accumulator;
                    if (!__is_zero(vals[i])) {
                        accumulator = __mul::<N, MOD_BITS>(params, accumulator, vals[i]);
                    }
                }

                let mut result: [[u128; N]; M] = [[0; N]; M];
                accumulator = __invmod::<N, MOD_BITS>(params, accumulator);
                for i in 0..M {
                    let idx: u32 = M - 1 - i;
                    if (!__is_zero(vals[idx])) {
                        let T0: [u128; N] = __mul::<N, MOD_BITS>(params, accumulator, temporaries[idx]);
                        accumulator = __mul::<N, MOD_BITS>(params, accumulator, vals[idx]);
                        result[idx] = T0;
                    }
                }
                result
            }

            /// Batch modular inversion of `BigNum` values in a slice (unconstrained)
            ///
            /// See `batch_invert` for details
            pub(crate) unconstrained fn batch_invert_slice<let N: u32, let MOD_BITS: u32>(
                params: BigNumParams<N, MOD_BITS>,
                vals: [[u128; N]],
            ) -> [[u128; N]] {
                let mut accumulator: [u128; N] = __one();
                let mut temporaries: [[u128; N]] = @[];

                let M: u32 = vals.len();

                for i in 0..M {
                    temporaries = temporaries.push_back(accumulator);
                    if (!__is_zero(vals[i])) {
                        accumulator = __mul::<N, MOD_BITS>(params, accumulator, vals[i]);
                    }
                }

                let mut result: [[u128; N]] = [];
                accumulator = __invmod::<N, MOD_BITS>(params, accumulator);
                for i in 0..M {
                    let idx: u32 = M - 1 - i;
                    if (!__is_zero(vals[idx])) {
                        let T0: [u128; N] = __mul::<N, MOD_BITS>(params, accumulator, temporaries[idx]);
                        accumulator = __mul::<N, MOD_BITS>(params, accumulator, vals[idx]);
                        result = result.push_front(T0);
                    } else {
                        result = result.push_front([0; N]);
                    }
                }

                result
            }

            /// Compute a modular square root in a prime field (unconstrained)
            pub unconstrained fn __sqrt<let N: u32, let MOD_BITS: u32>(
                params: BigNumParams<N, MOD_BITS>,
                input: [u128; N],
            ) -> crate::option::Option<[u128; N]> {
                assert(
                    params.has_multiplicative_inverse,
                    "BigNum::__sqrt: Must be a field to take square roots",
                );

                if (__is_zero(input)) {
                    Option::some(input)
                } else if (params.modulus[0] % 4 == 3) {
                    __easy_sqrt(params, input)
                } else {
                    __tonelli_shanks_sqrt(params, input)
                }
            }

            /// Compute a modular square root using the Tonelli-Shanks algorithm (unconstrained)
            ///
            /// Solves `x^2 = a (mod MOD)` for odd prime MOD
            ///
            /// ## Algorithm
            ///
            /// Here p = MOD
            ///
            /// Tonelli-Shanks setup
            ///
            /// Write `p - 1 = 2^s * Q`, `Q` - odd
            /// Define:
            ///    `R = a^{(Q+1)/2}`
            ///    `t = a^Q`
            ///
            /// so that `R^2 = a^{Q + 1} = a * a^Q = a * t`
            /// If t = 1, we are done
            ///
            /// By Euler's criterion, `a` is a quadratic reside iff `a^{(p - 1)/2} = 1`
            /// Since `t = a^Q` and `(p - 1) / 2 = Q * 2^{s-1}` we have:
            ///    `t^{2^{s-1}} = a^{Q * 2^{s-1}} = a^{(p-1)/2} = 1`, assuming `a` is a q.r.
            ///
            /// To proceed with computing our square root, we want to transfer `t` into a smaller subgroup,
            /// specifically, the `2^(s-2)`'th roots of unity or lower.
            ///
            /// We do this by finding some value `b`, such that
            /// `(t * b^2)^{2^{s-2}} = 1` and `R' = R * b`
            /// Finding such a `b` is trivial, because from Euler's criterion, we know that,
            /// for any quadratic non-residue `z`, `z^{(p - 1) / 2} = -1`
            /// i.e. `z^{Q * 2^{s-1}} = -1`
            /// => `z^Q` is a `2^{s-1}`'th root of `-1`
            /// => `z^{2 * Q}` is a `2^{s-2}`'th root of `-1`
            ///
            /// Since `t^{2^{s-1}} = 1`, we know that for some `i`, `i <= s - 2: t^{2^{i-1}} = -1`
            /// => `t * z^{2 * Q}` is a `2^{s - 2}`'th root of unity.
            /// We can iteratively transform `t` into ever smaller subgroups, until `t = 1`.
            /// At each iteration, we need to find a new value for `b`, which we can obtain
            /// by repeatedly squaring `z^Q`
            ///
            /// ## Note
            /// Only use for prime fields! Function may infinite loop if used for non-prime fields
            ///
            /// The input is assumed to be a nonzero value
            pub(crate) unconstrained fn __tonelli_shanks_sqrt<let N: u32, let MOD_BITS: u32>(
                params: BigNumParams<N, MOD_BITS>,
                input: [u128; N],
            ) -> crate::option::Option<[u128; N]> {
                let mut result: Option<[u128; N]> = Option::none();

                let one: [u128; N] = __one();
                let s: u32 = __primitive_root_log_size(params.modulus); // p - 1 = 2^s * Q, where Q is odd
                let Q: [u128; N] = __shr(__helper_sub(params.modulus, one), s);
                let Q_minus_one_over_two: [u128; N] = __shr1(__helper_sub(Q, one)); // (Q - 1) / 2

                let z: [u128; N] = __quadratic_non_residue::<N, MOD_BITS>(params);

                // Initialize:
                // b = a^{(Q - 1)/2}
                // R = a * b = a^{(Q + 1) / 2} => R^2 = a * a^Q
                // t = R * b = a^Q
                let mut b: [u128; N] = __pow::<N, MOD_BITS>(params, input, Q_minus_one_over_two);
                let mut r: [u128; N] = __mul::<N, MOD_BITS>(params, input, b);
                let mut t: [u128; N] = __mul::<N, MOD_BITS>(params, r, b);

                let mut check: [u128; N] = t;
                // Assure t^{2^{s - 1}} = a^{(p -1)/2} = 1, otherwise we have met a non-residue
                for _ in 0..s - 1 {
                    check = __sqr::<N, MOD_BITS>(params, check);
                }
                if (__eq(check, one)) {
                    let mut m: u32 = s;
                    let mut c: [u128; N] = __pow::<N, MOD_BITS>(params, z, Q); // z^Q - proper 2^{M}'th root of unity

                    // Tonelli-Shanks main loop

                    // At the beginning of each iteration we have:
                    // M - current exponent such that t lies in the subgroup of order 2^m
                    // t - element, whose order divides 2^m
                    // c - the proper 2^M'th root of unity
                    // R - accumulator with R^2 = a * t

                    // If t == 1, we are done and R is a square root
                    //
                    // Otherwise
                    // 1. We compute 1 <= i < M, such that t^{2^i} = 1, t^{2^{i - 1}} = -1
                    // 2. Set b = c^{2^{M - i - 1}}, so it becomes a proper 2^(i+1)'th root of unity
                    //      Then b^2 has order 2^i which matches the order of t
                    //
                    // 3. Update the state values:
                    //    R <- R * b
                    //    t <- t * b^2
                    //    c <- b^2 = c^{2^{M - i}}
                    //    M <- i
                    // reduces the order of t from 2^i to at most 2^{i - 1} and preserves R^2 = a * t
                    //
                    // The loop runs at most s times because M strictly decreases
                    for _ in 0..s {
                        if (__eq(t, one)) {
                            result = Option::some(r);
                            break;
                        }
                        let i: u32 = __tonelli_shanks_sqrt_find_i::<N, MOD_BITS>(params, t);
                        let j: u32 = m - i - 1;
                        b = c;
                        for _ in 0..j {
                            b = __sqr(params, b);
                        }

                        let b2: [u128; N] = __sqr::<N, MOD_BITS>(params, b);
                        c = b2;
                        t = __mul::<N, MOD_BITS>(params, t, b2);
                        r = __mul::<N, MOD_BITS>(params, r, b);
                        m = i;
                    }
                }
                result
            }

            /// Compute a modular square root for MOD = 3 (mod 4) (unconstrained)
            ///
            /// In case MOD = 3 (mod 4), the square root can be computed using the formula:
            ///     `R = a^{(MOD + 1) / 4} (mod MOD)`
            ///
            /// Then R^2 = a^{(MOD + 1)/ 4 * 2} = a^{(MOD + 1) / 2} = a^{(MOD - 1) / 2 + 1} = a
            ///
            /// ## Note
            /// The input is assumed to be a nonzero value
            ///
            /// This is much cheaper than `__tonelli_shanks_sqrt`
            pub(crate) unconstrained fn __easy_sqrt<let N: u32, let MOD_BITS: u32>(
                params: BigNumParams<N, MOD_BITS>,
                input: [u128; N],
            ) -> crate::option::Option<[u128; N]> {
                let mut result: Option<[u128; N]> = Option::none();

                let one: [u128; N] = __one();
                let p_minus_one_over_two: [u128; N] = __shr1(__helper_sub(params.modulus, one));
                let check: [u128; N] = __pow(params, input, p_minus_one_over_two);
                if (__eq(check, one)) {
                    // a = (MOD - 1) / 2
                    // b = (a + 1) / 2 = ((MOD - 1) / 2 + 1) / 2 = (MOD + 1) / 4
                    let p_plus_one_over_four: [u128; N] = __shr1(__increment(p_minus_one_over_two));
                    result = Option::some(__pow(params, input, p_plus_one_over_four));
                }
                result
            }
        }

        pub mod constrained_ops {

            use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
            use crate::mavros::vendor::bignum::constants::{GRUMPKIN_MODULUS, TWO_POW_120, TWO_POW_240};

            use crate::mavros::vendor::bignum::fns::{
                expressions::{evaluate_quadratic_expression, validate_udiv_mod_expression},
                unconstrained_helpers::{
                    __from_field, compute_add_flags, compute_borrow_flags, compute_gte_flags, compute_sub_flags,
                },
                unconstrained_ops::{__add, __div, __mul, __neg, __sqr, __sub, __udiv_mod},
            };

            use crate::mavros::vendor::bignum::params::BigNumParams;

            use crate::cmp::Ordering;

            /// Lift the limbs of a `BigNum` value onto the circuit `Field`
            ///
            /// Descent the `BigNum` value back into the `Field` and
            ///    - check that it's a proper `BigNum` value
            ///    - validate the limbs sum up to a `Field` value
            pub(crate) fn limbs_to_field<let N: u32, let MOD_BITS: u32>(
                _params: BigNumParams<N, MOD_BITS>,
                limbs: [u128; N],
            ) -> Field {
                validate_in_range::<u128, N, MOD_BITS>(limbs);
                if N > 2 {
                    // validate that the `BigNum` is less than the Grumpkin modulus
                    let mut grumpkin_modulus: [u128; N] = [0; N];
                    grumpkin_modulus[0] = GRUMPKIN_MODULUS[0];
                    grumpkin_modulus[1] = GRUMPKIN_MODULUS[1];
                    grumpkin_modulus[2] = GRUMPKIN_MODULUS[2];
                    validate_gt::<N, MOD_BITS>(grumpkin_modulus, limbs);
                }

                if N < 2 {
                    limbs[0] as Field
                } else if N == 2 {
                    (limbs[0] as Field) + (limbs[1] as Field) * (TWO_POW_120 as Field)
                } else {
                    (limbs[0] as Field)
                        + (limbs[1] as Field) * (TWO_POW_120 as Field)
                        + (limbs[2] as Field) * TWO_POW_240
                }
            }

            /// Construct a `BigNum` value from a native `Field`
            ///
            /// Decomposes the `Field` value into 120-bit limbs
            /// then we have three cases:
            ///     - MOD_BITS < 253 (grumpkin_mod_bits - 1): it is enough to call for `validate_in_field`, which is basically `val <= MOD`
            ///     - MOD_BITS > 253: we need to verify that the obtained `BigNum` `val < GRUMPKIN_MODULUS`
            ///     - MOD_BITS = 253: verify that `val < min(MOD, GRUMPKIN_MODULUS)`
            /// Next we verify that all the limbs are properly ranged
            /// and that the accumulated limbs are equal to the input `Field` value
            pub fn from_field<let N: u32, let MOD_BITS: u32>(
                _params: BigNumParams<N, MOD_BITS>,
                val: Field,
            ) -> [u128; N] {
                // Safety: we check that the resulting limbs represent the intended field element
                // we check the bit length, the limbs being max 120 bits, and the value in total is less than the field modulus
                let result: [u128; N] = unsafe { __from_field::<N>(val) };

                if !crate::runtime::is_unconstrained() {
                    // validate the limbs are in range and the value in total is less than 2^254
                    if MOD_BITS < 253 {
                        // this means that the field modulus is smaller than grumpkin modulus so we have to check if the element fields in the field size
                        validate_in_field(_params, result);
                    } else {
                        let mut grumpkin_modulus: [u128; N] = [0; N];
                        grumpkin_modulus[0] = GRUMPKIN_MODULUS[0];
                        grumpkin_modulus[1] = GRUMPKIN_MODULUS[1];
                        grumpkin_modulus[2] = GRUMPKIN_MODULUS[2];

                        if MOD_BITS > 253 {
                            // this means that the field modulus is larger than grumpkin modulus so we have to check if the element fields in the field size are less than the grumpkin modulus.
                            // also for correct params N is always larger than 3 here
                            validate_gt::<N, MOD_BITS>(grumpkin_modulus, result);
                        } else {
                            // this is the tricky part, when MOD_BITS = 253, we have to compare the limbs of the modulus to the grumpkin modulus limbs
                            // any `BigNum` with 253 bits will have 3 limbs

                            // if MOD is less than grumpkin modulus, this will be true
                            let mut mod_lt_grumpkin: bool = false;
                            for i in 0..3 {
                                if !mod_lt_grumpkin & (_params.modulus[2 - i] < grumpkin_modulus[2 - i]) {
                                    mod_lt_grumpkin = true;
                                }
                            }
                            let min_modulus: [u128; N] = if mod_lt_grumpkin {
                                _params.modulus
                            } else {
                                grumpkin_modulus
                            };
                            validate_gt::<N, MOD_BITS>(min_modulus, result);
                        }
                    }
                    validate_in_range::<u128, N, MOD_BITS>(result);

                    // validate the limbs sum up to the field value
                    let field_val: Field = if N < 2 {
                        result[0] as Field
                    } else if N == 2 {
                        (result[0] as Field) + (result[1] as Field) * (TWO_POW_120 as Field)
                    } else {
                        (result[0] as Field)
                            + (result[1] as Field) * (TWO_POW_120 as Field)
                            + (result[2] as Field) * TWO_POW_240
                    };
                    assert_eq(field_val, val);
                }

                result
            }

            /// Given an input seed, generate a pseudorandom `BigNum` value
            ///
            /// This function *should* produce a uniformly randomly distributed value modulo `MOD`
            ///
            ///  First we take the seed and pack it's 31-byte chunks into `Field`s
            ///  We use a hash function that can be modelled as a random oracle
            ///  We hash the packed seed using Poseidon2 to produce `MOD_BITS * 2` bits of entropy
            ///
            ///  From these bits we construct 4(in case N = 2) or 3(N > 2) `BigNum` values:
            ///  - We fill first `N - 1` limbs and leave the top limb empty
            ///
            ///  Then we accumulate the resulting BigNum values using:
            ///  B = 2^{120 * (N - 1)}
            ///  res = x3 + B * x2 + B^2 * x1 + B^3 * x0
            ///
            /// ## Note
            /// This function will always produce an `x3` `BigNum`
            /// If `MOD = 2^{120 * (N - 1)}`
            /// It will use only `MOD_BITS - 1` bits of entropy
            pub fn derive_from_seed<let N: u32, let MOD_BITS: u32, let SeedBytes: u32>(
                params: BigNumParams<N, MOD_BITS>,
                seed: [u8; SeedBytes],
            ) -> [u128; N] {
                // This compatibility subset is only used for signature verification; seed
                // derivation is not part of that reachable API.
                let _ = params;
                let _ = seed;
                [0; N]
            }

            // ------------------------------ COMPARISON FUNCTIONS ------------------------------

            /// Validate lhs != rhs
            ///
            /// We compare `A` and `B` via their encodings in the circuit `Field`
            ///
            /// Under our range assumptions, equality in `Field` implies equality of the underlying
            /// `BigNum` values, hence equality (mod `MOD`)
            ///
            /// In this library it is possible that A or B is a little bit greater than `MOD`
            /// So A == B (mod `MOD`) implies that A == B, A == B + MOD or A == B - MOD over the integers
            /// Hence we can compute everything (mod p) and constrain that
            /// (A - B) * (A - B + MOD) * (A - B - MOD) != 0 (mod p)
            ///
            /// ## Soundness
            /// This method is *sound* for checking `A != B (mod MOD)`
            ///
            /// If `A == B (mod MOD)`, then `A - B` is in `{0, +-MOD}` as an integer,
            /// so one of the factors `A-B`, `A-B+MOD`, `A-B-MOD` is zero in `Field`, and the product is zero
            /// Therefore, whenever the assertion `target != 0` holds, we must have `A != B (mod MOD)`
            /// under our range assumptions
            ///
            /// ## Completeness
            /// In general, this method is not *complete*: if the admissible range of `A` and `B`
            /// is large enough relative to the circuit `Field` prime `p`, an honest prover with
            /// `A != B (mod MOD)` can still hit an alias where
            ///     (A - B) * (A - B + MOD) * (A - B - MOD) == 0 (mod p),
            /// i.e.
            ///     - A = B       (mod p), or
            ///     - A = B + MOD (mod p), or
            ///     - A = B - MOD (mod p).
            ///
            /// For random `A, B` in such a wide range, the probability of this collision is
            /// roughly 3/p.
            ///
            /// In case `MOD` < `p` this function becomes *complete*
            pub fn assert_is_not_equal<let N: u32, let MOD_BITS: u32>(
                params: BigNumParams<N, MOD_BITS>,
                lhs: [u128; N],
                rhs: [u128; N],
            ) {
                let mut l: Field = 0;
                let mut r: Field = 0;
                let mut modulus_mod_p: Field = 0;
                for i in 0..N {
                    l *= TWO_POW_120 as Field;
                    r *= TWO_POW_120 as Field;
                    modulus_mod_p *= TWO_POW_120 as Field;

                    l += lhs[N - i - 1] as Field;
                    r += rhs[N - i - 1] as Field;
                    modulus_mod_p += params.modulus[N - i - 1] as Field;
                }

                let diff: Field = l - r;
                let target: Field = diff * (diff + modulus_mod_p) * (diff - modulus_mod_p);
                assert(target != 0, "assert_is_not_equal fail");
            }

            /// Compute equality flag
            ///
            /// A == B (mod MOD)
            /// We compute A - B and check whether it is `0` or `MOD`
            /// This is due to subtract constrains the diff value to be < 2^MOD_BITS, not < `MOD`
            ///
            /// ## Soundness
            /// This function is conditionally *sound*. See `sub` for details
            ///
            /// ## Completeness
            /// This function is *complete*. An honest prover will always be able to execute it.
            ///
            /// ## TODO
            /// can do this more efficiently via witngen in unconstrained functions?
            pub fn eq<let N: u32, let MOD_BITS: u32>(
                params: BigNumParams<N, MOD_BITS>,
                lhs: [u128; N],
                rhs: [u128; N],
            ) -> bool {
                let diff: [u128; N] = sub::<N, MOD_BITS>(params, lhs, rhs);
                is_zero::<N, MOD_BITS>(params, diff)
            }

            /// Validate that `val` is not equal to zero when interpreted as an integer.
            ///
            /// This enforces that at least one limb of `val` is non-zero.
            /// It does *not* check "BigNum zero" in the modular sense (e.g. it
            /// treats `MOD` as non-zero).
            ///
            /// ## Assumptions
            /// * Each limb of `val` is range-constrained to be a 120-bit value:
            ///   `0 <= val[i] < 2^120`.
            /// * For our concrete fields and limb counts we have `N * 2^120 < p`,
            ///   so the sum of all limbs fits strictly inside the field modulus.
            ///
            /// ## Completeness
            /// If `val` is non-zero as an integer, then at least one limb is non-zero,
            /// so the integer sum of the limbs satisfies `0 < limb_sum < p`. In this
            /// case `limb_sum != 0` in the `Field`, and the assertion passes.
            ///
            /// ## Soundness
            /// If all limbs of `val` are zero, then `limb_sum` is zero as an integer
            /// and as a field element, so the assertion fails. A witness representing
            /// the zero integer can never satisfy this check.
            ///
            /// ## Note
            /// This is slightly cheaper than doing `val != [0; N]`, as we avoid
            /// creating per-limb boolean equalities and chaining them with `and`s.
            pub fn assert_is_not_zero_integer<T, let N: u32>(val: [T; N])
            where
                T: Into<Field>,
            {
                assert(!is_zero_integer(val), "assert_is_not_zero_integer fail");
            }

            /// Check whether `val` is the zero `BigNum` in the integer sense.
            ///
            /// This returns `true` iff all limbs of `val` are zero. It does *not*
            /// treat `MOD` as zero; for modular `BigNum` zero use `is_zero`.
            ///
            /// See `assert_is_not_zero_integer` for the underlying assumptions.
            ///
            /// ## Note
            /// This is slightly cheaper than testing `val == [0; N]`, as we avoid
            /// creating per-limb boolean equalities and chaining them with `and`s.
            pub fn is_zero_integer<T, let N: u32>(val: [T; N]) -> bool
            where
                T: Into<Field>,
            {
                let mut limb_sum: Field = 0;
                for i in 0..N {
                    limb_sum += val[i].into();
                }
                limb_sum == 0
            }

            /// Validate that a `BigNum` value is not zero modulo `MOD`.
            ///
            /// Convenience wrapper around `assert_is_not_equal`.
            pub fn assert_is_not_zero<let N: u32, let MOD_BITS: u32>(
                params: BigNumParams<N, MOD_BITS>,
                val: [u128; N],
            ) {
                assert_is_not_equal::<N, MOD_BITS>(params, val, [0; N]);
            }

            /// Check whether a `BigNum` value is zero modulo `MOD`.
            ///
            /// This treats both the all-zero limb vector and `params.modulus` as
            /// representing zero. It assumes that all valid `BigNum` values are
            /// range-constrained so that no other representatives of `0 (mod MOD)`
            /// can appear.
            ///
            /// ## Note
            /// This is cheaper than calling `eq(val, [0; N])`
            pub fn is_zero<let N: u32, let MOD_BITS: u32>(
                params: BigNumParams<N, MOD_BITS>,
                val: [u128; N],
            ) -> bool {
                is_zero_integer(val) | (val == params.modulus)
            }

            /// Validate a `BigNum` instance is correctly range constrained to contain no more than `MOD_BITS` bits
            ///
            /// Constrain the `BigNum` instance to be < 2^MOD_BITS by:
            ///     - Constraining each limb(0..N-2) to be 120-bit limb
            ///     - Constraining the last limb to be `MOD_BITS - 120 * (N - 1)`
            ///
            /// ## Note
            /// This can be a very expensive function, when the `TOP_LIMB_BITS` is uncommon
            ///
            /// For example: BLS12_377Fr, with `TOP_LIMB_BITS`=13
            /// It creates a new 13-bit range table, which consists of roughly 3k variables
            /// And ~2k circuit gates
            ///
            /// Compare this to BLS12_377Fq, with `TOP_LIMB_BITS`=17
            /// It is nicely decomposed into
            /// 14-bit range check and 3-bit range check - much much cheaper, since 14-bit range checks
            /// are already pretty common for 120-bit range checks
            pub fn validate_in_range<T, let N: u32, let MOD_BITS: u32>(limbs: [T; N])
            where
                T: Into<Field>,
            {
                for i in 0..(N - 1) {
                    limbs[i].into().assert_max_bit_size::<120>();
                }

                limbs[N - 1].into().assert_max_bit_size::<MOD_BITS - ((N - 1) * 120)>();
            }

            /// Validate quotient produced from `evaluate_quadratic_expression` is well-formed
            ///
            /// Because the inputs into `evaluate_quadratic_expression` may cause the quotient to extend beyond `Params::modulus_bits`.
            ///           We allow the quotient to extend `6` bits beyond `Params::modulus_bits()`
            ///           Why is this?
            ///           several factors:    1. quotient * modulus , limbs cannot overflow `Field` boundary (254 bits)
            ///                               2. in `evaluate_quadratic_expression`, we require that for `expression - quotient * modulus`,
            ///                                  limbs cannot exceed `246` bits (246 magic number due to a higher number adding extra range check gates)
            ///           because of factor 2 and the fact that modulus limbs are 120 bits, quotient limbs cannot be > 126 bits
            pub(crate) fn validate_quotient_in_range<let N: u32, let MOD_BITS: u32>(limbs: [u128; N]) {
                for i in 0..(N - 1) {
                    (limbs[i] as Field).assert_max_bit_size::<120>();
                }
                (limbs[N - 1] as Field).assert_max_bit_size::<MOD_BITS - (N - 1) * 120 + 6>();
            }

            /// Validate that `lhs - rhs` does not underflow i.e. that lhs > rhs over the integers
            ///
            /// Compute `result = lhs - rhs` along with `borrow_flags`,
            /// then constrain `result` to be a valid `BigNum` value.
            ///
            /// ## Completeness
            /// This function is complete and will work only if `lhs > rhs` over the integers.
            ///
            /// ## Soundness
            /// This function is sound:
            ///     result[0]     = lhs[0]     - rhs[0]     + bf[0] * 2^{120}             < 2^{120}
            ///     result[i]     = lhs[i]     - rhs[i]     + bf[i] * 2^{120} - bf[i - 1] < 2^{120},  i = 1..N-2
            ///     result[N - 1] = lhs[N - 1] - rhs[N - 1]                   - bf[N - 2] < 2^{TOP_LIMB_BITS}
            ///     result != 0
            ///
            /// If `lhs < rhs`, then some limb of `result` would have to borrow from a higher limb,
            /// which is impossible because the top limb cannot borrow
            /// Without the extra borrow, the resulting difference will be wrapped around the `Field` modulus
            /// And won't satisfy the range constraint, since all the limbs are < 2^120
            ///
            /// ## Note
            /// `assert_is_not_zero_integer(result)` is crucial. Without it, we could always provide
            /// two identical inputs `x`, `x` and set `borrow_flags = [false; N]`,
            /// which would satisfy the limb constraints.
            pub(crate) fn validate_gt<let N: u32, let MOD_BITS: u32>(lhs: [u128; N], rhs: [u128; N]) {
                // Safety: borrow_flags are constrained by `assert_sub_no_overflow`:
                // - incorrect flags cause the computed result to fail 128-bit range checks
                // - if lhs < rhs, the subtraction overflows Field and fails range checks
                let borrow_flags: [bool; N - 1] = unsafe { compute_borrow_flags(lhs, rhs) };

                // Compute and validate the result in constrained code
                let result: [u128; N] = assert_sub_no_overflow::<N, MOD_BITS>(lhs, rhs, borrow_flags);

                // Constrain it to be strict inequality
                assert_is_not_zero_integer(result);
            }

            /// Compute the result of lhs - rhs given borrow flags and validate it
            ///
            /// This is the constrained analog of the result computation in the unconstrained helpers.
            /// Given borrow_flags, computes result such that:
            ///     result[0]     = lhs[0]     - rhs[0]     + bf[0] * 2^{120}
            ///     result[i]     = lhs[i]     - rhs[i]     + bf[i] * 2^{120} - bf[i - 1],  i = 1..N-2
            ///     result[N - 1] = lhs[N - 1] - rhs[N - 1]                   - bf[N - 2]
            ///
            /// Also validates that the result is a valid BigNum value (each limb in range).
            /// If lhs < rhs, the result will have a negative value (wrapped in Field) and fail validation.
            fn assert_sub_no_overflow<let N: u32, let MOD_BITS: u32>(
                lhs: [u128; N],
                rhs: [u128; N],
                borrow_flags: [bool; N - 1],
            ) -> [u128; N] {
                let mut result: [u128; N] = [0; N];

                // Compute first limb
                let limb_0: Field =
                    (lhs[0] as Field) - (rhs[0] as Field) + (borrow_flags[0] as Field) * (TWO_POW_120 as Field);
                limb_0.assert_max_bit_size::<128>();
                result[0] = limb_0 as u128;

                // Compute middle limbs
                for i in 1..N - 1 {
                    let limb_i: Field = (lhs[i] as Field) - (rhs[i] as Field)
                        + (borrow_flags[i] as Field) * (TWO_POW_120 as Field)
                        - (borrow_flags[i - 1] as Field);
                    limb_i.assert_max_bit_size::<128>();
                    result[i] = limb_i as u128;
                }

                // Compute last limb
                let limb_last: Field =
                    (lhs[N - 1] as Field) - (rhs[N - 1] as Field) - (borrow_flags[N - 2] as Field);
                limb_last.assert_max_bit_size::<128>();
                result[N - 1] = limb_last as u128;

                // Validate that the result is a valid BigNum value (120-bit limbs, TOP_LIMB_BITS for last)
                validate_in_range::<u128, N, MOD_BITS>(result);

                result
            }

            /// Compute the result of modular subtraction given flags and validate it
            ///
            /// Computes `result = lhs - rhs + (underflow ? modulus : 0)` given carry and borrow flags.
            /// The formula for each limb is:
            ///     result[0]     = lhs[0]     - rhs[0]     + addend[0]     + bf[0] * 2^120 - cf[0] * 2^120
            ///     result[i]     = lhs[i]     - rhs[i]     + addend[i]     + bf[i] * 2^120 - cf[i] * 2^120 - bf[i-1] + cf[i-1],  i = 1..N-2
            ///     result[N - 1] = lhs[N - 1] - rhs[N - 1] + addend[N - 1]                               - bf[N-2] + cf[N-2]
            ///
            /// Also validates that the result is a valid BigNum value (each limb in range).
            fn compute_sub_result<let N: u32, let MOD_BITS: u32>(
                modulus: [u128; N],
                lhs: [u128; N],
                rhs: [u128; N],
                carry_flags: [bool; N - 1],
                borrow_flags: [bool; N - 1],
                underflow: bool,
            ) -> [u128; N] {
                let addend: [u128; N] = if underflow { modulus } else { [0; N] };

                let mut result: [u128; N] = [0; N];

                // Compute first limb
                let limb_0: Field = (lhs[0] as Field) - (rhs[0] as Field)
                    + (addend[0] as Field)
                    + (borrow_flags[0] as Field) * (TWO_POW_120 as Field)
                    - (carry_flags[0] as Field) * (TWO_POW_120 as Field);
                limb_0.assert_max_bit_size::<128>();
                result[0] = limb_0 as u128;

                // Compute middle limbs
                for i in 1..N - 1 {
                    let limb_i: Field = (lhs[i] as Field) - (rhs[i] as Field)
                        + (addend[i] as Field)
                        + (borrow_flags[i] as Field) * (TWO_POW_120 as Field)
                        - (carry_flags[i] as Field) * (TWO_POW_120 as Field)
                        - (borrow_flags[i - 1] as Field)
                        + (carry_flags[i - 1] as Field);
                    limb_i.assert_max_bit_size::<128>();
                    result[i] = limb_i as u128;
                }

                // Compute last limb
                let limb_last: Field = (lhs[N - 1] as Field) - (rhs[N - 1] as Field) + (addend[N - 1] as Field)
                    - (borrow_flags[N - 2] as Field)
                    + (carry_flags[N - 2] as Field);
                limb_last.assert_max_bit_size::<128>();
                result[N - 1] = limb_last as u128;

                // Validate that the result is a valid BigNum value (120-bit limbs, TOP_LIMB_BITS for last)
                validate_in_range::<u128, N, MOD_BITS>(result);

                result
            }

            /// Compute the result of modular addition given flags and validate it
            ///
            /// Computes `result = lhs + rhs - (overflow ? modulus : 0)` given carry and borrow flags.
            /// The formula for each limb is:
            ///     result[0]     = lhs[0]     + rhs[0]     - subtrahend[0]     + bf[0] * 2^120 - cf[0] * 2^120
            ///     result[i]     = lhs[i]     + rhs[i]     - subtrahend[i]     + bf[i] * 2^120 - cf[i] * 2^120 - bf[i-1] + cf[i-1],  i = 1..N-2
            ///     result[N - 1] = lhs[N - 1] + rhs[N - 1] - subtrahend[N - 1]                               - bf[N-2] + cf[N-2]
            ///
            /// Also validates that the result is a valid BigNum value (each limb in range).
            fn compute_add_result<let N: u32, let MOD_BITS: u32>(
                modulus: [u128; N],
                lhs: [u128; N],
                rhs: [u128; N],
                carry_flags: [bool; N - 1],
                borrow_flags: [bool; N - 1],
                overflow: bool,
            ) -> [u128; N] {
                let subtrahend: [u128; N] = if overflow { modulus } else { [0; N] };

                let mut result: [u128; N] = [0; N];

                // Compute first limb
                let limb_0: Field = (lhs[0] as Field) + (rhs[0] as Field) - (subtrahend[0] as Field)
                    + (borrow_flags[0] as Field) * (TWO_POW_120 as Field)
                    - (carry_flags[0] as Field) * (TWO_POW_120 as Field);
                limb_0.assert_max_bit_size::<128>();
                result[0] = limb_0 as u128;

                // Compute middle limbs
                for i in 1..N - 1 {
                    let limb_i: Field = (lhs[i] as Field) + (rhs[i] as Field)
                        - (subtrahend[i] as Field)
                        - (borrow_flags[i - 1] as Field)
                        + (carry_flags[i - 1] as Field)
                        + (borrow_flags[i] as Field) * (TWO_POW_120 as Field)
                        - (carry_flags[i] as Field) * (TWO_POW_120 as Field);
                    limb_i.assert_max_bit_size::<128>();
                    result[i] = limb_i as u128;
                }

                // Compute last limb
                let limb_last: Field = (lhs[N - 1] as Field) + (rhs[N - 1] as Field)
                    - (subtrahend[N - 1] as Field)
                    - (borrow_flags[N - 2] as Field)
                    + (carry_flags[N - 2] as Field);
                limb_last.assert_max_bit_size::<128>();
                result[N - 1] = limb_last as u128;

                // Validate that the result is a valid BigNum value (120-bit limbs, TOP_LIMB_BITS for last)
                validate_in_range::<u128, N, MOD_BITS>(result);

                result
            }

            /// Validate that `val` <= `MOD`
            ///
            /// Compute `result = MOD - val` along with `borrow_flags`,
            /// then constrain `result` to be a valid `BigNum` value.
            ///
            /// Basically the same as `validate_gt` but we compute the result on the fly
            /// It is just a bit more optimized as we expect each `BigNum` value to be \leq `MOD`
            ///
            /// ## Note
            /// In contrast to `validate_gt`, we allow the value to be `MOD`
            /// Since it is consistent with the rest of the library
            pub fn validate_in_field<let N: u32, let MOD_BITS: u32>(
                params: BigNumParams<N, MOD_BITS>,
                val: [u128; N],
            ) {
                let modulus: [u128; N] = params.modulus;

                // Safety: borrow_flags are constrained by the `validate_in_range` check on `p_minus_self`:
                // - incorrect flags cause `p_minus_self` limbs to overflow Field, failing range checks
                // - if val > modulus, the subtraction overflows and fails the range check
                let borrow_flags: [bool; (N - 1)] = unsafe { compute_borrow_flags(modulus, val) };

                let mut p_minus_self: [Field; N] = [0; N];
                p_minus_self[0] = (modulus[0] as Field) - (val[0] as Field)
                    + (borrow_flags[0] as Field) * (TWO_POW_120 as Field);
                for i in 1..N - 1 {
                    p_minus_self[i] = (modulus[i] as Field) - (val[i] as Field)
                        + (borrow_flags[i] as Field) * (TWO_POW_120 as Field)
                        - (borrow_flags[i - 1] as Field);
                }
                p_minus_self[N - 1] =
                    (modulus[N - 1] as Field) - (val[N - 1] as Field) - (borrow_flags[N - 2] as Field);
                validate_in_range::<Field, N, MOD_BITS>(p_minus_self);
            }

            /// Compare two `BigNum` values
            ///
            /// Returns `lhs > rhs`
            ///
            /// ## Note
            /// This is a strict value comparison over the integers,
            /// the values do not have to be reduced modulo `MOD`.
            pub fn cmp<let N: u32, let MOD_BITS: u32>(lhs: [u128; N], rhs: [u128; N]) -> Ordering {
                // Safety: underflow and borrow_flags are constrained by `assert_sub_no_overflow`:
                // - we swap (a, b) based on underflow, then compute a - b
                // - if underflow is wrong, a < b and the subtraction overflows, failing range checks
                // - if borrow_flags are wrong, the computed limbs fail 128-bit range checks
                let (underflow, borrow_flags): (bool, [bool; N - 1]) = unsafe { compute_gte_flags(lhs, rhs) };

                // if underflow is true, swap lhs and rhs so we compute larger - smaller
                let (a, b): ([u128; N], [u128; N]) = if underflow { (rhs, lhs) } else { (lhs, rhs) };

                // Enforce correctness of `underflow` by asserting that the subtraction does not overflow.
                let _: [u128; N] = assert_sub_no_overflow::<N, MOD_BITS>(a, b, borrow_flags);

                if lhs == rhs {
                    Ordering::equal()
                } else if underflow {
                    Ordering::less()
                } else {
                    Ordering::greater()
                }
            }

            // ------------------------------ ARITHMETIC FUNCTIONS ------------------------------

            /// Negate a `BigNum` value
            ///
            /// Computes `result = MOD - val` using limb-wise subtraction with borrow flags,
            /// then constrains:
            ///   - all `result` limbs to be a valid `BigNum` value, and
            ///   - the subtraction relation with the borrow flags
            ///
            /// ## Assumptions
            /// - `val` is a valid `BigNum` in the range `0 <= val <= MOD`.
            ///
            /// ## Soundness
            /// This function constrains the following relations:
            ///     result[0]     = MOD[0]     - val[0]     + bf[0] * 2^{120}             < 2^{120}
            ///     result[i]     = MOD[i]     - val[i]     + bf[i] * 2^{120} - bf[i - 1] < 2^{120},  i = 1..N-2
            ///     result[N - 1] = MOD[N - 1] - val[N - 1]                   - bf[N - 2] < 2^{TOP_LIMB_BITS}
            ///
            /// If all `MOD` and `val` limbs are valid `BigNum` limbs, these constraints
            /// ensure that:
            ///   - the borrow flags `bf[i]` form a valid limb-wise subtraction chain, and
            ///   - no underflow can occur in the subtraction `MOD - val`.
            ///
            /// ## Completeness
            /// This function is complete for inputs in the range `0 <= val <= MOD`.
            /// If a value `val > MOD` is passed in (while still `< 2^{MOD_BITS}`), the
            /// constraints above will fail, since there is no valid borrow chain making
            /// `MOD - val` a well-formed `BigNum`.
            ///
            /// In practice, honest provers should not hit this case: all functions in this
            /// module are expected to return values `< MOD`.
            ///
            /// ## Note
            /// This function returns `MOD` when `val` is zero.
            pub fn neg<let N: u32, let MOD_BITS: u32>(
                params: BigNumParams<N, MOD_BITS>,
                val: [u128; N],
            ) -> [u128; N] {
                if crate::runtime::is_unconstrained() {
                    // Safety: unconstrained runtime requires no constraints
                    unsafe {
                        __neg(params.modulus, val)
                    }
                } else {
                    // Safety: borrow_flags are constrained by `assert_sub_no_overflow`:
                    // - incorrect flags cause computed limbs to fail 128-bit range checks
                    // - if val > modulus, the subtraction overflows and fails range checks
                    // (but val > modulus violates function preconditions)
                    let borrow_flags: [bool; N - 1] = unsafe { compute_borrow_flags(params.modulus, val) };

                    // Subtract `val` from the modulus to negate.
                    assert_sub_no_overflow::<N, MOD_BITS>(params.modulus, val, borrow_flags)
                }
            }

            /// Add two `BigNum` values
            ///
            /// Computes `result = lhs + rhs` using limb-wise addition with carry flags,
            /// and an optional subtraction of `MOD` using borrow flags. The function then
            /// constrains:
            ///   - all `result` limbs to be a valid `BigNum` value, and
            ///   - the addition/subtraction relation with the carry/borrow flags and the
            ///     `overflow_modulus` bit.
            ///
            /// ## Assumptions
            /// - All limbs of `lhs`, `rhs` and `MOD` are valid `BigNum` limbs
            ///   (120-bit for non-top limbs, `TOP_LIMB_BITS` for the top limb).
            /// - Semantically, we intend to use this only with `0 <= lhs, rhs < MOD`,
            ///   even though the limb/range constraints allow values up to `< 2^{MOD_BITS}`.
            ///
            /// ## Soundness (intended relation)
            /// This function constrains the following equations:
            ///
            ///     result[0]     = lhs[0]     + rhs[0]
            ///                     - sub[0]
            ///                     + bf[0] * 2^{120}
            ///                     - cf[0]                                        < 2^{120}
            ///
            ///     result[i]     = lhs[i]     + rhs[i]
            ///                     - sub[i]
            ///                     + bf[i] * 2^{120}  - bf[i - 1]
            ///                     - cf[i] * 2^{120}  + cf[i - 1]                 < 2^{120},
            ///                     for i = 1..N-2
            ///
            ///     result[N - 1] = lhs[N - 1] + rhs[N - 1]
            ///                     - sub[N - 1]
            ///                     - bf[N - 2]
            ///                     + cf[N - 2]                                   < 2^{TOP_LIMB_BITS}
            ///
            /// where:
            ///   - `bf[i]` are the borrow flags of the optional subtraction,
            ///   - `cf[i]` are the carry flags of the addition, and
            ///   - `sub` is either the zero vector or `MOD`, depending on the
            ///     `overflow_modulus` flag:
            ///         * `overflow_modulus = 0`  =>  `sub = 0`
            ///         * `overflow_modulus = 1`  =>  `sub = MOD`
            ///
            /// If `lhs`, `rhs` and `MOD` are valid `BigNum` limbs and the witness for
            /// `(bf, cf, overflow_modulus)` is the honest one produced by `compute_add_flags`,
            /// these constraints enforce:
            ///   - a valid limb-wise carry chain for `lhs + rhs`, and
            ///   - a valid limb-wise subtraction chain for either `lhs + rhs` or
            ///     `lhs + rhs - MOD`, with no underflow in any limb.
            ///
            /// Under these assumptions the constrained result equals:
            ///
            ///     result = lhs + rhs (mod MOD)
            ///
            /// in the intended arithmetic.
            ///
            /// ## Limitations / extra satisfying witnesses
            ///
            /// The constraint system itself does **not** uniquely determine the carry/borrow
            /// flags nor the `overflow_modulus` bit:
            ///
            /// - For each limb `i`, `bf[i]` and `cf[i]` only appear in the combination
            ///   `bf[i] * 2^{120} - cf[i] * 2^{120}`. This means that both
            ///
            ///       (bf[i], cf[i]) = (0, 0)  and  (bf[i], cf[i]) = (1, 1)
            ///
            ///   give the same contribution to the equation. As a result, there are multiple
            ///   valid flag assignments for the *same* `lhs`, `rhs`, `sub` and `result`.
            ///
            /// - More importantly, if `lhs + rhs` is greater than `MOD` but still strictly
            ///   less than `2^{MOD_BITS}`, there exist *spurious* witnesses where:
            ///     * `overflow_modulus` is set inconsistently with the true arithmetic
            ///       overflow, and
            ///     * the `(bf, cf)` flags are adjusted accordingly,
            ///
            ///   such that all equations above still hold and all `result` limbs remain
            ///   within range. In particular, when
            ///
            ///       lhs + rhs + MOD < 2^{MOD_BITS},
            ///
            ///   a malicious prover can "hide" an extra `MOD` inside the choice of
            ///   `overflow_modulus`, `bf` and `cf`, so that the circuit is satisfied by a
            ///   witness
            ///
            /// Consequently, this function is only *conditionally* sound: we rely on the
            /// out-of-circuit implementation of `compute_add_flags` to provide the honest
            /// `(borrow_flags, carry_flags, overflow_modulus)` witness. Under that
            /// assumption, the constrained `result` matches `lhs + rhs (mod MOD)`.
            ///
            /// ## Completeness
            ///
            /// For inputs in the range `0 <= lhs, rhs < MOD` and honest flags from
            /// `compute_add_flags`, the constraints are complete: every valid `BigNum` sum
            /// `lhs + rhs (mod MOD)` admits a satisfying assignment.
            ///
            /// Inputs with `lhs` or `rhs` in `[MOD, 2^{MOD_BITS})` are still representable
            /// as limb arrays and may admit satisfying witnesses, but then the operation
            /// no longer corresponds to a unique, well-defined addition in the field
            /// `Z / MOD Z`. Such uses are outside the intended semantics of this function.
            pub fn add<let N: u32, let MOD_BITS: u32>(
                params: BigNumParams<N, MOD_BITS>,
                lhs: [u128; N],
                rhs: [u128; N],
            ) -> [u128; N] {
                if crate::runtime::is_unconstrained() {
                    // Safety: unconstrained runtime requires no constraints
                    unsafe {
                        __add(params.modulus, lhs, rhs)
                    }
                } else {
                    // Safety: flags are constrained by `compute_add_result`:
                    // - incorrect flags cause computed limbs to fail 128-bit range checks
                    // - wrong overflow causes result to be off by modulus, failing `validate_in_range`
                    let (carry_flags, borrow_flags, overflow): ([bool; N - 1], [bool; N - 1], bool) =
                        unsafe { compute_add_flags(params.modulus, lhs, rhs) };

                    // Compute and validate the result in constrained code
                    compute_add_result::<N, MOD_BITS>(
                        params.modulus,
                        lhs,
                        rhs,
                        carry_flags,
                        borrow_flags,
                        overflow,
                    )
                }
            }

            /// Subtract two `BigNum` values
            ///
            /// Computes `result = lhs - rhs` using limb-wise subtraction with borrow flags,
            /// and an optional addition of `MOD` using carry flags. The function then
            /// constrains:
            ///   - all `result` limbs to be a valid `BigNum` value, and
            ///   - the subtraction/addition relation with the carry/borrow flags and the
            ///     `underflow_modulus` bit.
            ///
            /// ## Assumptions
            /// - All limbs of `lhs`, `rhs` and `MOD` are valid `BigNum` limbs
            ///   (120-bit for non-top limbs, `TOP_LIMB_BITS` for the top limb).
            /// - Semantically, we intend to use this only with `0 <= lhs, rhs < MOD`,
            ///   even though the limb/range constraints allow values up to `< 2^{MOD_BITS}`.
            ///
            /// ## Soundness (intended relation)
            /// This function constrains the following equations:
            ///
            ///     result[0]     = lhs[0]     - rhs[0]
            ///                     + add[0]
            ///                     + bf[0] * 2^{120}
            ///                     - cf[0]                                        < 2^{120}
            ///
            ///     result[i]     = lhs[i]     - rhs[i]
            ///                     + add[i]
            ///                     + bf[i] * 2^{120}  - bf[i - 1]
            ///                     - cf[i] * 2^{120}  + cf[i - 1]                 < 2^{120},
            ///                     for i = 1..N-2
            ///
            ///     result[N - 1] = lhs[N - 1] - rhs[N - 1]
            ///                     + add[N - 1]
            ///                     - bf[N - 2]
            ///                     + cf[N - 2]                                   < 2^{TOP_LIMB_BITS}
            ///
            /// where:
            ///   - `bf[i]` are the borrow flags of the subtraction,
            ///   - `cf[i]` are the carry flags of the optional addition, and
            ///   - `add` is either the zero vector or `MOD`, depending on the
            ///     `underflow_modulus` flag:
            ///         * `underflow_modulus = 0`  =>  `add = 0`
            ///         * `underflow_modulus = 1`  =>  `add = MOD`
            ///
            /// If `lhs`, `rhs` and `MOD` are valid `BigNum` limbs and the witness for
            /// `(bf, cf, underflow_modulus)` is the honest one produced by `compute_sub_flags`,
            /// these constraints enforce:
            ///   - a valid limb-wise borrow chain for `lhs - rhs`, and
            ///   - a valid limb-wise addition chain for either `lhs - rhs` or
            ///     `lhs - rhs + MOD`, with no underflow in any limb.
            ///
            /// Under these assumptions the constrained result equals:
            ///
            ///     result = lhs - rhs (mod MOD)
            ///
            /// in the intended arithmetic.
            ///
            /// ## Limitations / extra satisfying witnesses
            ///
            /// The constraint system itself does **not** uniquely determine the carry/borrow
            /// flags nor the `underflow_modulus` bit:
            ///
            /// - For each limb `i`, `bf[i]` and `cf[i]` only appear in the combination
            ///   `bf[i] * 2^{120} - cf[i] * 2^{120}`. This means that both
            ///
            ///       (bf[i], cf[i]) = (0, 0)  and  (bf[i], cf[i]) = (1, 1)
            ///
            ///   give the same contribution to the equation. As a result, there are multiple
            ///   valid flag assignments for the *same* `lhs`, `rhs`, `add` and `result`.
            ///
            /// - More importantly, when `lhs < rhs`, the true field result is
            ///
            ///       lhs - rhs + MOD,
            ///
            ///   and as long as
            ///
            ///       lhs - rhs + MOD < 2^{MOD_BITS},
            ///
            ///   a malicious prover can:
            ///     * set `underflow_modulus` inconsistently with the true underflow, and
            ///     * adjust the `(bf, cf)` flags accordingly,
            ///
            ///   such that all equations above still hold and all `result` limbs remain
            ///   within range. In other words, the circuit can be satisfied by a witness
            ///   that does **not** correspond to the unique intended subtraction modulo
            ///   `MOD` for some inputs with `lhs < rhs`.
            ///
            /// Consequently, this function is only *conditionally* sound: we rely on the
            /// out-of-circuit implementation of `compute_sub_flags` to provide the honest
            /// `(borrow_flags, carry_flags, underflow_modulus)` witness. Under that
            /// assumption, the constrained `result` matches `lhs - rhs (mod MOD)`.
            ///
            /// ## Completeness
            ///
            /// For inputs in the range `0 <= lhs, rhs < MOD` and honest flags from
            /// `compute_sub_flags`, the constraints are complete: every valid `BigNum`
            /// difference `lhs - rhs (mod MOD)` admits a satisfying assignment.
            ///
            /// Inputs with `lhs` or `rhs` in `[MOD, 2^{MOD_BITS})` are still representable
            /// as limb arrays and may admit satisfying witnesses, but then the operation
            /// no longer corresponds to a unique, well-defined subtraction in the field
            /// `Z / MOD Z`. Such uses are outside the intended semantics of this function.
            pub fn sub<let N: u32, let MOD_BITS: u32>(
                params: BigNumParams<N, MOD_BITS>,
                lhs: [u128; N],
                rhs: [u128; N],
            ) -> [u128; N] {
                if crate::runtime::is_unconstrained() {
                    // Safety: unconstrained runtime requires no constraints
                    unsafe {
                        __sub(params.modulus, lhs, rhs)
                    }
                } else {
                    // Safety: flags are constrained by `compute_sub_result`:
                    // - incorrect flags cause computed limbs to fail 128-bit range checks
                    // - wrong underflow causes result to be off by modulus, failing `validate_in_range`
                    let (carry_flags, borrow_flags, underflow): ([bool; N - 1], [bool; N - 1], bool) =
                        unsafe { compute_sub_flags(params.modulus, lhs, rhs) };

                    // Compute and validate the result in constrained code
                    compute_sub_result::<N, MOD_BITS>(
                        params.modulus,
                        lhs,
                        rhs,
                        carry_flags,
                        borrow_flags,
                        underflow,
                    )
                }
            }

            /// Compute the `BigNum` multiplication
            ///
            /// Computes `result = lhs * rhs (mod MOD)` by:
            ///   1. Computing `result` out of circuit via `__mul`.
            ///   2. Constraining the quadratic relation `lhs * rhs - result = 0` with
            ///      `evaluate_quadratic_expression`.
            ///
            /// ## Soundness
            /// Soundness reduces to `evaluate_quadratic_expression` for the relation
            /// `lhs * rhs - result = 0`
            ///
            /// ## Note
            /// When possible, prefer expressing your computation directly as a quadratic
            ///   relation and calling `evaluate_quadratic_expression` instead of using `mul`
            pub fn mul<let N: u32, let MOD_BITS: u32>(
                params: BigNumParams<N, MOD_BITS>,
                lhs: [u128; N],
                rhs: [u128; N],
            ) -> [u128; N] {
                // Safety: we constrain the multiplication result immediately after
                let result: [u128; N] = unsafe { __mul::<N, MOD_BITS>(params, lhs, rhs) };
                if !crate::runtime::is_unconstrained() {
                    // lhs * rhs - result = 0
                    evaluate_quadratic_expression(
                        params,
                        [[lhs]],
                        [[false]],
                        [[rhs]],
                        [[false]],
                        [result],
                        [true],
                    );
                }
                result
            }

            /// Compute the `BigNum` squaring
            ///
            /// Computes `result = val * val (mod MOD)` by:
            ///   1. Computing `result` out of circuit via `__sqr`.
            ///   2. Constraining the quadratic relation `val * val - result = 0` with
            ///      `evaluate_quadratic_expression`.
            ///
            /// ## Soundness
            /// Soundness reduces to `evaluate_quadratic_expression` for the relation
            /// `val * val - result = 0`
            ///
            /// ## Note
            /// When possible, prefer expressing your computation directly as a quadratic
            ///   relation and calling `evaluate_quadratic_expression` instead of using `sqr`
            pub fn sqr<let N: u32, let MOD_BITS: u32>(
                params: BigNumParams<N, MOD_BITS>,
                val: [u128; N],
            ) -> [u128; N] {
                // Safety: we constrain the multiplication result immediately after
                let result: [u128; N] = unsafe { __sqr::<_, MOD_BITS>(params, val) };
                if !crate::runtime::is_unconstrained() {
                    // val * val - result = 0
                    evaluate_quadratic_expression(
                        params,
                        [[val]],
                        [[false]],
                        [[val]],
                        [[false]],
                        [result],
                        [true],
                    );
                }
                result
            }

            /// Compute the `BigNum` division
            ///
            /// Computes `result = lhs * rhs^{-1} (mod MOD)` by:
            ///   1. Computing `result` out of circuit via `__div`.
            ///   2. Constraining the quadratic relation `result * rhs - lhs = 0` with
            ///      `evaluate_quadratic_expression`.
            ///   3. Enforcing `rhs != 0 (mod MOD)`.
            ///
            /// ## Soundness
            /// Soundness reduces to `evaluate_quadratic_expression` for the relation
            /// `result * rhs - lhs = 0`, together with the non-zero check on `rhs` and the
            /// assumption that `MOD` has multiplicative inverses for all non-zero elements
            /// (`params.has_multiplicative_inverse = true`).
            ///
            /// ## Note
            /// - This is **expensive** in witness time due to modular inversion.
            /// - When possible, prefer expressing your computation directly as a quadratic
            ///   relation and calling `evaluate_quadratic_expression` instead of using `div`.
            /// - In the unconstrained runtime, the behavior of `__div` on zero divisors is
            ///   not constrained by this function.
            pub fn div<let N: u32, let MOD_BITS: u32>(
                params: BigNumParams<N, MOD_BITS>,
                lhs: [u128; N],
                rhs: [u128; N],
            ) -> [u128; N] {
                assert(
                    params.has_multiplicative_inverse,
                    "BigNum has no multiplicative inverse. Use udiv for unsigned integer division",
                );
                // Safety: We constrain the result of division immediately after
                let result: [u128; N] = unsafe { __div::<_, MOD_BITS>(params, lhs, rhs) };
                if !crate::runtime::is_unconstrained() {
                    // result * rhs - lhs = 0
                    evaluate_quadratic_expression(
                        params,
                        [[result]],
                        [[false]],
                        [[rhs]],
                        [[false]],
                        [lhs],
                        [true],
                    );
                    assert_is_not_zero(params, rhs);
                }
                result
            }

            /// Compute the `BigNum` integer division with remainder
            ///
            /// Computes `quotient = floor(numerator / divisor)` and
            /// `remainder = numerator % divisor` by:
            ///   1. Computing `(quotient, remainder)` out of circuit via `__udiv_mod`.
            ///   2. Constraining the quadratic relation
            ///        quotient * divisor + remainder - numerator = 0
            ///      with `validate_udiv_mod_expression`.
            ///   3. Enforcing `remainder < divisor`.
            ///
            /// ## Soundness
            /// Soundness reduces to `validate_udiv_mod_expression` for the relation
            ///     quotient * divisor + remainder - numerator = 0,
            /// together with `remainder < divisor` check enforced via `validate_gt`.
            ///
            /// Under these checks, any satisfying assignment corresponds to a valid
            /// integer division `numerator = quotient * divisor + remainder` with
            /// `0 <= remainder < divisor`
            ///
            /// ## Note
            /// Enforcing `divisor != 0` is not necessary. `remainder < divisor`
            /// Already enforces this.
            pub fn udiv_mod<let N: u32, let MOD_BITS: u32>(
                numerator: [u128; N],
                divisor: [u128; N],
            ) -> ([u128; N], [u128; N]) {
                // Safety: We constrain the result of __udiv_mod immediately after
                let (quotient, remainder): ([u128; N], [u128; N]) = unsafe { __udiv_mod(numerator, divisor) };
                if !crate::runtime::is_unconstrained() {
                    // quotient * divisor + remainder - numerator = 0
                    validate_udiv_mod_expression::<N, MOD_BITS>(numerator, divisor, quotient, remainder);
                    // remainder < divisor
                    validate_gt::<N, MOD_BITS>(divisor, remainder);
                }
                (quotient, remainder)
            }

            /// Compute the `BigNum` integer division
            ///
            /// Returns `floor(numerator / divisor)`.
            /// All constraints and soundness details are handled inside `udiv_mod`.
            pub fn udiv<let N: u32, let MOD_BITS: u32>(numerator: [u128; N], divisor: [u128; N]) -> [u128; N] {
                udiv_mod::<N, MOD_BITS>(numerator, divisor).0
            }

            /// Compute the `BigNum` remainder
            ///
            /// Returns `numerator % divisor`.
            /// All constraints and soundness details are handled inside `udiv_mod`.
            pub fn umod<let N: u32, let MOD_BITS: u32>(numerator: [u128; N], divisor: [u128; N]) -> [u128; N] {
                udiv_mod::<N, MOD_BITS>(numerator, divisor).1
            }
        }

        pub mod unconstrained_helpers {

            use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
            // This file contains the unconstrained helpers that are mostly used by unconstrained ops

            use crate::mavros::vendor::bignum::constants::TWO_POW_120;

            use crate::mavros::vendor::bignum::fns::unconstrained_ops::{__eq, __gte, __neg, __pow, __sqr};

            use crate::mavros::vendor::bignum::utils::msb::get_msb;
            use crate::mavros::vendor::bignum::utils::split_bits::{__normalize_limbs, __split_120_bits};

            use crate::mavros::vendor::bignum::params::BigNumParams;

            // ------------------------------ DERIVATION HELPER FUNCTIONS ------------------------------

            ///  Construct a `1` BigNum value (unconstrained)
            pub(crate) unconstrained fn __one<let N: u32>() -> [u128; N] {
                let mut limbs: [u128; N] = [0; N];
                limbs[0] = 1;
                limbs
            }

            /// Construct a BigNum value from Field (unconstrained)
            ///
            /// Split the native `Field` value into `N` 120-bit limbs
            pub(crate) unconstrained fn __from_field<let N: u32>(val: Field) -> [u128; N] {
                let x: Field = val;
                let mut result: [u128; N] = [0; N];

                if (N == 1) {
                    let (first_limb, _): (u128, Field) = __split_120_bits(x);
                    result[0] = first_limb;
                }

                if (N == 2) {
                    let (first_limb, x): (u128, Field) = __split_120_bits(x);
                    let (second_limb, _): (u128, Field) = __split_120_bits(x);
                    result[0] = first_limb;
                    result[1] = second_limb;
                }

                if (N > 2) {
                    let (first_limb, x): (u128, Field) = __split_120_bits(x);
                    let (second_limb, x): (u128, Field) = __split_120_bits(x);
                    let (third_limb, _): (u128, Field) = __split_120_bits(x);
                    result[0] = first_limb;
                    result[1] = second_limb;
                    result[2] = third_limb;
                }
                result
            }

            // ------------------------------ ARITHMETIC WITH FLAGS HELPER FUNCTIONS ------------------------------
            // These are the functions that compute modular operations results as well as borrow and carry flags for constraints

            /// Compute flags for modular addition (unconstrained)
            ///
            /// Given `lhs, rhs, modulus`, computes the carry and borrow flags needed
            /// for `lhs + rhs - (overflow ? modulus : 0)`.
            /// The result is computed in constrained code using `compute_add_result`.
            ///
            /// Returns (carry_flags, borrow_flags, overflow) where:
            /// - carry_flags: carries from adding lhs + rhs
            /// - borrow_flags: borrows from subtracting modulus when overflow occurs
            /// - overflow: true if lhs + rhs >= modulus (so we need to subtract modulus)
            pub(crate) unconstrained fn compute_add_flags<let N: u32>(
                modulus: [u128; N],
                lhs: [u128; N],
                rhs: [u128; N],
            ) -> ([bool; N - 1], [bool; N - 1], bool) {
                let mask: u128 = TWO_POW_120 - 1;

                let add_res: [u128; N] = __helper_add(lhs, rhs);
                let overflow: bool = __gte(add_res, modulus);

                let subtrahend: [u128; N] = if overflow { modulus } else { [0; N] };

                let mut borrow_flags: [bool; N - 1] = [false; N - 1];
                let mut carry_flags: [bool; N - 1] = [false; N - 1];

                let mut carry: u128 = 0;
                let mut borrow: u128 = 0;
                for i in 0..N {
                    let mut add_term: u128 = lhs[i] + rhs[i] + carry;
                    carry = add_term >> 120;
                    add_term &= mask;

                    let sub_term: u128 = subtrahend[i] + borrow;
                    borrow = (sub_term > add_term) as u128;

                    // Only set `borrow` and `carry` if they differ
                    // And if it's not the last limb
                    if (carry != borrow) & (i < N - 1) {
                        carry_flags[i] = carry != 0;
                        borrow_flags[i] = borrow != 0;
                    }
                }
                (carry_flags, borrow_flags, overflow)
            }

            /// Compute flags for modular subtraction (unconstrained)
            ///
            /// Given `lhs, rhs, modulus`, computes the carry and borrow flags needed
            /// for `lhs - rhs + (underflow ? modulus : 0)`.
            /// The result is computed in constrained code using `compute_sub_result`.
            ///
            /// Returns (carry_flags, borrow_flags, underflow) where:
            /// - carry_flags: carries from adding (lhs + modulus) when underflow occurs
            /// - borrow_flags: borrows from subtracting rhs
            /// - underflow: true if lhs < rhs (so we need to add modulus)
            pub(crate) unconstrained fn compute_sub_flags<let N: u32>(
                modulus: [u128; N],
                lhs: [u128; N],
                rhs: [u128; N],
            ) -> ([bool; N - 1], [bool; N - 1], bool) {
                let mask: u128 = TWO_POW_120 - 1;

                let underflow: bool = !__gte(lhs, rhs);

                let addend: [u128; N] = if underflow { modulus } else { [0; N] };

                let mut borrow_flags: [bool; N - 1] = [false; N - 1];
                let mut carry_flags: [bool; N - 1] = [false; N - 1];

                let mut carry: u128 = 0;
                let mut borrow: u128 = 0;
                for i in 0..N {
                    let mut add_term: u128 = lhs[i] + addend[i] + carry;
                    carry = add_term >> 120;
                    add_term &= mask;

                    let sub_term: u128 = rhs[i] + borrow;
                    borrow = (sub_term > add_term) as u128;

                    // Only set `borrow` and `carry` if they differ
                    // And if it's not the last limb
                    if (carry != borrow) & (i < N - 1) {
                        carry_flags[i] = carry != 0;
                        borrow_flags[i] = borrow != 0;
                    }
                }
                (carry_flags, borrow_flags, underflow)
            }

            /// Compute borrow flags for lhs - rhs (unconstrained)
            ///
            /// The result is computed in constrained code using `compute_gte_result`
            pub(crate) unconstrained fn compute_borrow_flags<let N: u32>(
                lhs: [u128; N],
                rhs: [u128; N],
            ) -> [bool; N - 1] {
                let mut borrow_flags: [bool; N - 1] = [false; N - 1];
                borrow_flags[0] = lhs[0] < rhs[0];
                for i in 1..N - 1 {
                    borrow_flags[i] = lhs[i] < rhs[i] + (borrow_flags[i - 1] as u128);
                }
                borrow_flags
            }

            /// Compute underflow and borrow flags for gte comparison (unconstrained)
            ///
            /// Returns (underflow, borrow_flags) where:
            /// - underflow is true if lhs < rhs
            /// - borrow_flags correspond to max(lhs, rhs) - min(lhs, rhs)
            pub(crate) unconstrained fn compute_gte_flags<let N: u32>(
                lhs: [u128; N],
                rhs: [u128; N],
            ) -> (bool, [bool; N - 1]) {
                let underflow: bool = !__gte(lhs, rhs);
                // swap if underflow so we're computing borrow flags for larger - smaller
                let (a, b): ([u128; N], [u128; N]) = if underflow { (rhs, lhs) } else { (lhs, rhs) };

                let borrow_flags = compute_borrow_flags(a, b);
                (underflow, borrow_flags)
            }

            // ------------------------------ BARRETT REDUCTION ------------------------------

            /// `BARRETT_REDUCTION_OVERFLOW_BITS` defines how large an input to barrett reduction can be
            ///
            /// maximum value = modulus^2 << BARRETT_REDUCTION_OVERFLOW_BITS
            /// see __barrett_reduction for more details
            global BARRETT_REDUCTION_OVERFLOW_BITS: u32 = 6;

            /// Optimized modular multiplication (unconstrained)
            ///
            /// The trick is to approximate 1/p with m/2**r, because division by 2**r is much cheaper
            /// In our case m = redc_param = floor(2^{MOD_BITS * 2 + BARRET_REDUCTION_OVERFLOW_BITS} / p)
            ///             r = MOD_BITS * 2 + BARRET_REDUCTION_OVERFLOW_BITS
            ///
            /// When we apply the barrett reduction, the maximum value of the output will be <= p * (1 + x/2^{2k})
            /// where p = modulus,
            ///       x = reduction input
            ///
            /// If x > p * p, we need s to be larger than modulus_bits()
            /// We hardcode s = BARRET_REDUCTION_OVERFLOW_BITS = 6, which means that the maximum value of x is approx. 64 * p * p
            /// This should be larger than most values put into `evaluate_quadratic_expression`
            ///
            /// ## TODO
            /// Detect cases where x might be too large at comptime
            ///
            /// ## Note
            /// very niche edge case error that we need to be aware of:
            ///      N must be large enough to cover the modulus *plus* BARRETT_REDUCTION_OVERFLOW_BITS
            ///      i.e. a 359-bit prime needs (I think) 4 limbs to represent or we may overflow
            ///      when calling __barrett_reduction
            ///
            /// ## Note on final reduction
            ///
            /// Assumptions:
            ///  - k = ceil(log2 p), so p <= 2^k
            ///  - s = 6, m = redc_param = floor(2^{2*k + s}/p)
            ///  - x < 64 * p^2 (x < 2^{2 * k + s})
            ///
            /// Let m' = 2^{2*k + s} / p, and write m = m' - \epsilon, \epsilon \in [0, 1)
            //
            /// quo = floor(x * m / 2^{2 * k + s}) = floor(x * m' / 2^{2 * k + s} - x * \epsilon / 2^{2 * k + s}) =
            /// floor(x / p - x * \epsilon / 2^{2 * k + s})
            ///
            /// Bounds:
            ///   quo <= floor(x / p)
            ///
            ///   floor(a - b) >= floor(a) - ceil(b) (known identity) =>
            ///   quo >= floor(x / p) - ceil(x * \epsilon / 2^{2 * k + s})
            ///       >= floor(x / p) - ceil(x / 2^{2 * k + s}) (epsilon < 1)
            ///
            /// x / 2^{2 * k + s} < C * p^2 / 2^{2 * k + s} <= C * 2^{2 * k} / 2^{2 * k + s} = C / 2^s
            ///
            /// When the assumption holds (C = 64), ceil(x / 2^{2 * k + s}) = 1, for x > 0
            /// Therefore quo = {floor(x/p), floor(x/p) - 1}
            ///
            /// In first case: rem = x - quo * p = x - floor(x/p) * p < p
            /// In second case: rem = x - (floor(x/p) - 1) * p = (x - floor(x/p) * p) + p -> need 1 subtraction
            ///
            /// ### Note
            /// In the worst case, we will have the input > 64 * p^2
            /// (for example (a1 + b1) * (c1 + d1) + ... 64 times)
            /// This is highly unlikely though, but there should be more reductions in that case.
            pub(crate) unconstrained fn __barrett_reduction<let N: u32>(
                x: [u128; 2 * N],
                redc_param: [u128; N],
                k: u32,
                modulus: [u128; N],
            ) -> ([u128; N], [u128; N]) {
                // TODO: switch to __helper_mul, once the compiler is smart enough to handle this
                let mut mulout_field: [Field; 3 * N] = [0; 3 * N];
                for i in 0..(2 * N) {
                    for j in 0..N {
                        mulout_field[i + j] += (x[i] as Field) * (redc_param[j] as Field);
                    }
                }
                let mulout: [u128; 3 * N] = __normalize_limbs(mulout_field);

                let quotient: [u128; 3 * N] = __shr(mulout, (k + k + BARRETT_REDUCTION_OVERFLOW_BITS));

                // Remove a bunch of zeros from the end
                let mut smaller_quotient: [u128; N] = [0; N];
                for i in 0..N {
                    smaller_quotient[i] = quotient[i] as u128;
                }

                // long_quotient_mul_modulus can never exceed input value `x` so can fit into size-2 array
                let long_quotient_mul_modulus: [u128; 2 * N] = __helper_mul(smaller_quotient, modulus);
                let long_remainder: [u128; 2 * N] = __helper_sub(x, long_quotient_mul_modulus);

                // Remove a bunch of zeros from the end
                let mut remainder: [u128; N] = [0; N];
                for i in 0..N {
                    remainder[i] = long_remainder[i];
                }

                if (__gte(remainder, modulus)) {
                    remainder = __helper_sub(remainder, modulus);
                    smaller_quotient = __increment(smaller_quotient);
                }

                (smaller_quotient, remainder)
            }

            // ------------------------------ ARITHMETIC HELPER FUNCTIONS ------------------------------
            // These are the functions that operate on limbs as if they were just big integers

            /// Adds `1` to the BigNum value without modular reduction (unconstrained)
            ///
            /// ## Note
            /// The `carry` must be `0` at the end of the loop.
            /// No explicit assertion is made, as this condition is validated during evaluation.
            pub(crate) unconstrained fn __increment<let N: u32>(val: [u128; N]) -> [u128; N] {
                let mask: u128 = TWO_POW_120 - 1;

                let mut result: [u128; N] = [0; N];
                let mut carry: u128 = 1;
                for i in 0..N {
                    let add_term: u128 = val[i] + carry;
                    carry = add_term >> 120;
                    result[i] = add_term & mask;
                }
                result
            }

            /// Adds two `BigNum` values without modular reduction (unconstrained).
            ///
            /// ## Note
            /// The `carry` must be `0` at the end of the loop.
            /// No explicit assertion is made, as this condition is validated during evaluation.
            pub(crate) unconstrained fn __helper_add<let N: u32>(lhs: [u128; N], rhs: [u128; N]) -> [u128; N] {
                let mut result: [u128; N] = [0; N];

                let mut carry: u128 = 0;
                let mask: u128 = TWO_POW_120 - 1;

                for i in 0..N {
                    let add_term: u128 = lhs[i] + rhs[i] + carry;
                    carry = add_term >> 120;
                    result[i] = add_term & mask;
                }
                result
            }

            /// Subtracts two `BigNum` values without modular reduction (unconstrained).
            ///
            /// ## Note
            /// The `borrow` must be `0` at the end of the loop.
            /// No explicit assertion is made, as this condition is validated during evaluation.
            pub(crate) unconstrained fn __helper_sub<let N: u32>(lhs: [u128; N], rhs: [u128; N]) -> [u128; N] {
                let mut result: [u128; N] = [0; N];

                let mut borrow: u128 = 0;
                for i in 0..N {
                    let subtrahend: u128 = rhs[i] + borrow;
                    borrow = (subtrahend > lhs[i]) as u128;
                    result[i] = (borrow << 120) + lhs[i] - subtrahend;
                }
                result
            }

            /// Multiplies two `BigNum` values without modular reduction (unconstrained).
            ///
            /// Computes the full schoolbook product of two N-limb little-endian arrays
            ///
            /// ## Note
            /// The mathematical product fits in `2 * N - 1` limbs, but we keep `2 * N`
            /// limbs intentionally as the extra high limb safely absorbs a possible single limb overflow
            /// for moduli close to `120 * N` bits.
            pub(crate) unconstrained fn __helper_mul<let N: u32>(
                lhs: [u128; N],
                rhs: [u128; N],
            ) -> [u128; 2 * N] {
                let mut result: [Field; 2 * N] = [0; 2 * N];
                for i in 0..N {
                    for j in 0..N {
                        result[i + j] += (lhs[i] as Field) * (rhs[j] as Field);
                    }
                }
                __normalize_limbs(result)
            }

            /// Computes `x * 2^{-1} (mod MOD)` (unconstrained).
            ///
            /// For odd `MOD`, this is equivalent to:
            /// - `x/2` if `x` is even
            /// - `(x + MOD)/2` if `x` is odd  (since `x + MOD` is even)
            ///
            /// ## Note
            /// - `MOD` must be odd.
            pub(crate) unconstrained fn __half_mod_odd<let N: u32>(
                modulus: [u128; N],
                x: [u128; N],
            ) -> [u128; N] {
                let temp = if __is_even::<N>(x) {
                    x
                } else {
                    __helper_add(x, modulus)
                };
                __shr1(temp)
            }

            // ------------------------------ LOGIC HELPER FUNCTIONS ------------------------------
            // These are the functions that operate on limbs as if they were just big integers

            /// Left-shifts a `BigNum` value by `shift` bits (unconstrained).
            ///
            /// Performs a bitwise left shift across limbs.
            ///
            /// ## Note
            /// The most significant limb is truncated to 120 bits after the shift.
            ///
            /// No bounds check is performed on `num_shifted_limbs`.
            /// However, we use it only in `__udiv_mod`, where it is not possible to reach
            /// `num_shifted_limbs` > `N`
            pub(crate) unconstrained fn __shl<let N: u32>(input: [u128; N], shift: u32) -> [u128; N] {
                let mut result: [u128; N] = [0; N];

                let num_shifted_limbs: u32 = shift / 120;
                let limb_shift: u128 = (shift % 120) as u128;
                let remainder_shift: u128 = 120 - limb_shift;

                let mask: u128 = TWO_POW_120 - 1;
                let mut remainder: u128 = input[0] >> remainder_shift;

                result[num_shifted_limbs] = (input[0] << limb_shift) & mask;

                for i in 1..(N - num_shifted_limbs) {
                    let value: u128 = input[i];
                    let upshift: u128 = ((value << limb_shift) | remainder) & mask;
                    result[i + num_shifted_limbs] = upshift;
                    remainder = value >> remainder_shift;
                }
                result
            }

            /// Right-shifts a `BigNum` value by `shift` bits (unconstrained).
            ///
            /// Performs a bitwise right shift across limbs.
            ///
            /// # Note
            /// No bounds check is performed on `num_shifted_limbs`.
            /// However, we use it only in `__tonelli_shanks_sqrt`, where it is not possible to reach
            /// `num_shifted_limbs` > `N`
            pub(crate) unconstrained fn __shr<let N: u32>(input: [u128; N], shift: u32) -> [u128; N] {
                let mut result: [u128; N] = [0; N];

                let num_shifted_limbs: u32 = shift / 120;
                let limb_shift: u128 = (shift % 120) as u128;

                let remainder_shift: u128 = 120 - limb_shift;
                let low_mask: u128 = (1 as u128 << limb_shift) - 1;

                result[0] = input[num_shifted_limbs] >> limb_shift;
                for i in 1..(N - num_shifted_limbs) {
                    let value: u128 = input[i + num_shifted_limbs];

                    let carry: u128 = (value & low_mask) << remainder_shift;
                    result[i - 1] |= carry;

                    result[i] = value >> limb_shift;
                }
                result
            }

            /// Right-shifts a `BigNum` value by `1` bit  (unconstrained)
            ///
            /// # Note
            /// All the operations on limbs are executed in place
            /// to save opcodes
            pub(crate) unconstrained fn __shr1<let N: u32>(mut input: [u128; N]) -> [u128; N] {
                let value: u128 = input[N - 1];
                let mut remainder: u128 = (value & 1) << 119;
                input[N - 1] >>= 1;

                for i in 1..N {
                    let value: u128 = input[N - 1 - i];
                    input[N - 1 - i] = (value >> 1) | remainder;
                    remainder = (value & 1) << 119;
                }
                input
            }

            /// Returns the index of the most significant set bit in a `BigNum` value (unconstrained).
            pub(crate) unconstrained fn __get_msb<let N: u32>(val: [u128; N]) -> u32 {
                let mut count: u32 = 0;
                for i in 0..N {
                    let idx: u32 = N - 1 - i;
                    let v: u128 = val[idx];
                    if (v > 0) {
                        count = 120 * idx + get_msb(v);
                        break;
                    }
                }
                count
            }

            /// Returns `true` if the bit at position `bit` is set in the `BigNum` (unconstrained).
            ///
            /// ## Note
            /// No bounds check is performed on `bit`
            pub(crate) fn __get_bit<let N: u32>(input: [u128; N], bit: u32) -> bool {
                let segment_index: u32 = bit / 120;
                let uint_index: u128 = (bit % 120) as u128;

                let limb: u128 = input[segment_index];
                let value: u128 = (limb >> uint_index) & 1;
                value == 1
            }

            /// Returns `true` if the `BigNum` value is even (unconstrained)
            pub(crate) unconstrained fn __is_even<let N: u32>(x: [u128; N]) -> bool {
                (x[0] & 1) == 0
            }

            // ------------------------------ SQRT HELPER FUNCTIONS ------------------------------
            // These are the functions that are used during taking a square root

            /// Compute the maximal power of 2 that divides the group order (unconstrained)
            ///
            /// Find the maximum value s such that `MOD = 2^s * q + 1`, where `q` is odd
            ///       This is needed for our Tonelli-Shanks sqrt algorithm
            pub(crate) unconstrained fn __primitive_root_log_size<let N: u32>(modulus: [u128; N]) -> u32 {
                let target: [u128; N] = __helper_sub(modulus, __one());
                let mut result: u32 = 0;
                while !__get_bit(target, result) {
                    result += 1;
                }
                result
            }

            /// Find a quadratic non-residue `g` where `g` is the smallest such value (unconstrained)
            ///       i.e. smallest `g` such that `g^{(p - 1)/2} = -1 (mod MOD)`
            ///       or smallest `g`, such that `x^2 - g = 0 (mod MOD)` has no solutions
            ///
            /// ## Note
            /// WARNING If the field is not prime, this function will enter an infinite loop!
            pub(crate) unconstrained fn __quadratic_non_residue<let N: u32, let MOD_BITS: u32>(
                params: BigNumParams<N, MOD_BITS>,
            ) -> [u128; N] {
                let one: [u128; N] = __one();
                let neg_one: [u128; N] = __neg(params.modulus, one);

                let p_minus_one_over_two: [u128; N] = __shr1(__helper_sub(params.modulus, __one()));

                // We start with 2
                let mut target: [u128; N] = [0; N];
                target[0] = 2;

                let mut expd: [u128; N] = __pow(params, target, p_minus_one_over_two);
                while !__eq(expd, neg_one) {
                    target = __increment(target);
                    expd = __pow(params, target, p_minus_one_over_two);
                }
                target
            }

            /// Compute the smallest `i`, such that `t^{2^i} = 1, t^{2^{i-1}} = -1 (mod MOD)` (unconstrained)
            ///
            /// ## Note
            /// Multiplicative order of t must divide 2^v2(MOD-1), otherwise you'll end up in an infinite loop!
            pub(crate) unconstrained fn __tonelli_shanks_sqrt_find_i<let N: u32, let MOD_BITS: u32>(
                params: BigNumParams<N, MOD_BITS>,
                t: [u128; N],
            ) -> u32 {
                let one: [u128; N] = __one();
                let mut c: [u128; N] = t;

                let mut i: u32 = 0;
                // Compute t^{2^k} until it hits 1 for the first time
                while !__eq(c, one) {
                    c = __sqr::<N, MOD_BITS>(params, c);
                    i += 1;
                }
                i
            }
        }

        pub mod expressions {

            use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
            use crate::mavros::vendor::bignum::utils::split_bits;

            use crate::mavros::vendor::bignum::constants::{TWO_POW_120, TWO_POW_126, TWO_POW_246};

            use crate::mavros::vendor::bignum::fns::constrained_ops::{validate_in_range, validate_quotient_in_range};
            use crate::mavros::vendor::bignum::fns::unconstrained_helpers::__barrett_reduction;
            use crate::mavros::vendor::bignum::fns::unconstrained_ops::__is_zero;

            use crate::mavros::vendor::bignum::params::BigNumParams;

            // ------------------------------ UNCONSTRAINED EXPRESSIONS ------------------------------

            /// Compute the result of a linear combination of (possibly negative) `BigNum` values (unconstrained)
            ///
            /// ## Note
            /// 1. `modulus2` is structured such that all limbs will be greater than `0`, even when subtracting.
            /// To do this, when computing `p - x`, we ensure that each limb in `p` is greater than each limb in `x`.
            /// We know that, for a valid bignum element, the limbs in `x` will be < 2^{120}
            /// Therefore each of the limbs in `p` (except the most significant) will borrow 2^{120} from the more significant limb.
            /// Finally, to ensure we do not underflow in the most significant limb, we use `2p` instead of `p`
            ///
            /// 2. Returns the `Field` values that are not normalized to be 120-bit
            unconstrained fn __add_linear_expression<let N: u32, let MOD_BITS: u32, let M: u32>(
                params: BigNumParams<N, MOD_BITS>,
                vals: [[u128; N]; M],
                flags: [bool; M],
            ) -> ([Field; N]) {
                let mut sum: [Field; N] = [0; N];
                let modulus2: [u128; N] = params.double_modulus;
                for i in 0..M {
                    if (flags[i]) {
                        for j in 0..N {
                            sum[j] = sum[j] + (modulus2[j] as Field) - (vals[i][j] as Field);
                        }
                    } else {
                        for j in 0..N {
                            sum[j] = sum[j] + (vals[i][j] as Field);
                        }
                    }
                }
                sum
            }

            /// Compute the limb products of a quadratic expression (unconstrained)
            ///
            /// See `__compute_quadratic_expression_with_borrow_flags` for full description
            ///
            /// ## Note
            /// Returns the `Field` values that are not normalized to be 120-bit
            unconstrained fn __compute_quadratic_expression_product<let N: u32, let MOD_BITS: u32, let LHS_N: u32, let RHS_N: u32, let NUM_PRODUCTS: u32, let ADD_N: u32>(
                params: BigNumParams<N, MOD_BITS>,
                lhs_terms: [[[u128; N]; LHS_N]; NUM_PRODUCTS],
                lhs_flags: [[bool; LHS_N]; NUM_PRODUCTS],
                rhs_terms: [[[u128; N]; RHS_N]; NUM_PRODUCTS],
                rhs_flags: [[bool; RHS_N]; NUM_PRODUCTS],
                linear_terms: [[u128; N]; ADD_N],
                linear_flags: [bool; ADD_N],
            ) -> [Field; 2 * N] {
                let mut lhs: [[Field; N]; NUM_PRODUCTS] = [[0; N]; NUM_PRODUCTS];
                let mut rhs: [[Field; N]; NUM_PRODUCTS] = [[0; N]; NUM_PRODUCTS];

                for i in 0..NUM_PRODUCTS {
                    lhs[i] = __add_linear_expression(params, lhs_terms[i], lhs_flags[i]);
                    rhs[i] = __add_linear_expression(params, rhs_terms[i], rhs_flags[i]);
                }
                let add: [Field; N] = __add_linear_expression(params, linear_terms, linear_flags);

                let mut mulout: [Field; 2 * N] = [0; 2 * N];
                for i in 0..N {
                    for j in 0..N {
                        for k in 0..NUM_PRODUCTS {
                            mulout[i + j] += lhs[k][i] * rhs[k][j];
                        }
                    }
                    mulout[i] += add[i];
                }
                mulout
            }

            /// Compute the borrow flags for a limb-wise subtraction `lhs - rhs` (unconstrained).
            ///
            /// This helper operates on `N`-limb values where each limb is interpreted as a
            /// 120-bit chunk (base `B = 2^{120}`), but where intermediate limbs may exceed
            /// `2^{120}` due to sums of products.
            ///
            /// Conceptually, we want to model the integer subtraction
            ///
            ///     lhs - rhs
            ///
            /// as a base-`B` subtraction with borrows, without allowing any intermediate
            /// values to go negative in the circuit field. We achieve this by:
            ///
            ///   1. Working with widened 246-bit ranges per limb.
            ///   2. Encoding a borrow from limb `i+1` into limb `i` by:
            ///        - adding `2^{246}` into limb `i`,
            ///        - subtracting `2^{126}` (= 2^{246 - 120}) from limb `i+1` after
            ///          scaling by `2^{-120}`.
            ///
            /// The returned `borrow_flags[i]` indicate whether the canonical integer
            /// subtraction would borrow from limb `i+1` into limb `i`. These flags are
            /// later re-applied in-circuit via `apply_borrow_flags`, and the resulting
            /// limbs are checked by `validate_expression_is_zero`.
            unconstrained fn __compute_borrow_flags<let N: u32>(
                mut lhs_limbs: [Field; N],
                rhs_limbs: [Field; N],
            ) -> [bool; N - 1] {
                // compute borrow flags from mulout_p and mulout_n
                let mut borrow_flags: [bool; N - 1] = [false; N - 1];
                let borrow_shift: Field = TWO_POW_246; // 2^{246}
                let borrow_carry: Field = TWO_POW_126; // 2^{126} = 2^{246 - 120}
                let downshift_120: Field = 1 / (TWO_POW_120 as Field);

                // determine whether we need to borrow from more significant limbs.
                // initial limb is "simple" comparison operation
                borrow_flags[0] = lhs_limbs[0].lt(rhs_limbs[0]);

                // we have N - 1 borrow flags. The number of limbs is N
                // and there is nothing to borrow against for the final limb.
                let mut hi_bits: Field =
                    (lhs_limbs[0] - rhs_limbs[0] + ((borrow_flags[0] as Field) * borrow_shift)) * downshift_120;

                for i in 1..(N - 1) {
                    // compute the contribution from limb `i-1` that gets added into limb `i`, and add into limb `i`
                    lhs_limbs[i] += hi_bits;

                    let subtrahend: Field = rhs_limbs[i] + (borrow_flags[i - 1] as Field) * borrow_carry;

                    // determine whether negative limb values are greater than positive limb values
                    borrow_flags[i] = lhs_limbs[i].lt(subtrahend);
                    let minuend: Field = lhs_limbs[i] + (borrow_flags[i] as Field) * borrow_shift;

                    hi_bits = (minuend - subtrahend) * downshift_120;
                }
                borrow_flags
            }

            /// Given a degree-2 `BigNum` expression that is equal to `0 mod MOD`, compute
            /// the quotient and the borrow flags (unconstrained).
            ///
            /// The expression has the form:
            ///
            ///     sum_{k=0}^{NUM_PRODUCTS-1} (L_k * R_k) + sum_{i=0}^{ADD_N-1} A_i = quotient * MOD
            ///
            /// where each `L_k`, `R_k`, `A_i` is an `N`-limb `BigNum` assembled from the
            /// `(terms, flags)` arrays.
            ///
            /// This helper:
            ///   1. Evaluates the quadratic expression into `mulout_p` as a `2N`-limb
            ///      non-normalized value.
            ///   2. Normalizes `mulout_p` into 120-bit limbs and applies Barrett
            ///      reduction to obtain `(quotient, remainder)`.
            ///   3. Asserts that the integer remainder is zero (debugging aid; range
            ///      checks enforce this later).
            ///   4. Reconstructs `mulout_n = quotient * MOD`.
            ///   5. Computes `borrow_flags` for the limb-wise subtraction
            ///        `mulout_p - mulout_n`,
            ///      using the 2^{246}/2^{126} encoding described in
            ///      `evaluate_quadratic_expression`.
            ///
            /// The returned `quotient` and `borrow_flags` are later constrained
            /// in-circuit by `compute_quadratic_expression_with_modulus` and
            /// `evaluate_quadratic_expression`.
            unconstrained fn __compute_quadratic_expression_with_borrow_flags<let N: u32, let MOD_BITS: u32, let LHS_N: u32, let RHS_N: u32, let NUM_PRODUCTS: u32, let ADD_N: u32>(
                params: BigNumParams<N, MOD_BITS>,
                lhs_terms: [[[u128; N]; LHS_N]; NUM_PRODUCTS],
                lhs_flags: [[bool; LHS_N]; NUM_PRODUCTS],
                rhs_terms: [[[u128; N]; RHS_N]; NUM_PRODUCTS],
                rhs_flags: [[bool; RHS_N]; NUM_PRODUCTS],
                linear_terms: [[u128; N]; ADD_N],
                linear_flags: [bool; ADD_N],
            ) -> ([u128; N], [bool; 2 * N - 2]) {
                let mulout_p: [Field; 2 * N] = __compute_quadratic_expression_product(
                    params,
                    lhs_terms,
                    lhs_flags,
                    rhs_terms,
                    rhs_flags,
                    linear_terms,
                    linear_flags,
                );

                // `__normalize_limbs` will validate that we do not overflow 2N, normally we should not overflow 2N-1
                let relation_result: [u128; 2 * N] = split_bits::__normalize_limbs(mulout_p);
                let (quotient, remainder): ([u128; N], [u128; N]) =
                    __barrett_reduction(relation_result, params.redc_param, MOD_BITS, params.modulus);

                // This is verified later by the range checks but left for debugging purposes
                // Upstream documents this as a debugging assertion. The range constraints below
                // enforce the same relation without trapping an inactive witness branch.

                // We do not normalize `mulout_n` so we won't fill the `2 * N - 1`
                let mut mulout_n: [Field; 2 * N] = [0; 2 * N];
                for i in 0..N {
                    for j in 0..N {
                        mulout_n[i + j] += (quotient[i] as Field) * (params.modulus[j] as Field);
                    }
                }

                let borrow_flags: [bool; 2 * N - 1] = __compute_borrow_flags(mulout_p, mulout_n);

                // We have to copy it because we know that borrow_flags[2 * N - 1] is always 0
                // But we also have to provide 2 * N limbs to `__barrett_reduction`.
                // And keep `__compute_borrow_flags` generic enough
                let mut borrow_flags_real: [bool; 2 * N - 2] = [false; 2 * N - 2];
                for i in 0..2 * N - 2 {
                    borrow_flags_real[i] = borrow_flags[i];
                }

                (quotient, borrow_flags_real)
            }

            /// Computes the quotient/remainder of a quadratic expression (unconstrained)
            ///
            /// See `__compute_quadratic_expression_with_borrow_flags` for full description
            pub(crate) unconstrained fn __compute_quadratic_expression<let N: u32, let MOD_BITS: u32, let LHS_N: u32, let RHS_N: u32, let NUM_PRODUCTS: u32, let ADD_N: u32>(
                params: BigNumParams<N, MOD_BITS>,
                lhs_terms: [[[u128; N]; LHS_N]; NUM_PRODUCTS],
                lhs_flags: [[bool; LHS_N]; NUM_PRODUCTS],
                rhs_terms: [[[u128; N]; RHS_N]; NUM_PRODUCTS],
                rhs_flags: [[bool; RHS_N]; NUM_PRODUCTS],
                linear_terms: [[u128; N]; ADD_N],
                linear_flags: [bool; ADD_N],
            ) -> ([u128; N], [u128; N]) {
                let mulout: [Field; 2 * N] = __compute_quadratic_expression_product(
                    params,
                    lhs_terms,
                    lhs_flags,
                    rhs_terms,
                    rhs_flags,
                    linear_terms,
                    linear_flags,
                );
                // __normalize_limbs will validate that we do not overflow 2N, normally we should not overflow 2N-1
                let relation_result: [u128; 2 * N] = split_bits::__normalize_limbs(mulout);

                let (quotient, remainder): ([u128; N], [u128; N]) =
                    __barrett_reduction(relation_result, params.redc_param, MOD_BITS, params.modulus);

                (quotient, remainder)
            }

            // ------------------------------ CONSTRAINED EXPRESSIONS ------------------------------

            /// Constrained version of `__add_linear_expression`
            ///
            /// Computes all the linear parts of an expression in-circuit
            ///
            /// ## Note
            /// 1. Negative terms are implemented by adding `double_modulus`
            /// `double_modulus` is chosen so that all limbs except the top one
            /// are > 2^{120}, which prevents underflows in intermediate computations.
            ///
            /// 2. For the most significant limb we slightly reduce the padding (to keep the
            /// overall value equal to `2 * MOD`), so in principle there is a narrow edge
            /// case where that limb could underflow if enough negative contributions are
            /// accumulated and the top limb of `MOD` is very small. In practice, the
            /// global `BigNum` range and operand-count assumptions in
            /// `evaluate_quadratic_expression` rule out such patterns.
            fn compute_linear_expressions<let N: u32, let MOD_BITS: u32, let LHS_N: u32, let RHS_N: u32, let NUM_PRODUCTS: u32, let ADD_N: u32>(
                params: BigNumParams<N, MOD_BITS>,
                lhs_terms: [[[u128; N]; LHS_N]; NUM_PRODUCTS],
                lhs_flags: [[bool; LHS_N]; NUM_PRODUCTS],
                rhs_terms: [[[u128; N]; RHS_N]; NUM_PRODUCTS],
                rhs_flags: [[bool; RHS_N]; NUM_PRODUCTS],
                linear_terms: [[u128; N]; ADD_N],
                linear_flags: [bool; ADD_N],
            ) -> ([[Field; N]; NUM_PRODUCTS], [[Field; N]; NUM_PRODUCTS], [Field; N]) {
                // lhs linear terms
                let mut lhs_linear: [[Field; N]; NUM_PRODUCTS] = [[0; N]; NUM_PRODUCTS];
                // rhs linear terms
                let mut rhs_linear: [[Field; N]; NUM_PRODUCTS] = [[0; N]; NUM_PRODUCTS];
                // linear terms
                let mut lin_expr: [Field; N] = [0; N];

                for k in 0..NUM_PRODUCTS {
                    for i in 0..N {
                        for j in 0..LHS_N {
                            // Note: if lhs_flags[k][j] - `is_negative` is not known at comptime this is very expensive
                            if (lhs_flags[k][j]) {
                                lhs_linear[k][i] -= lhs_terms[k][j][i] as Field;
                                lhs_linear[k][i] += params.double_modulus[i] as Field;
                            } else {
                                lhs_linear[k][i] += lhs_terms[k][j][i] as Field;
                            }
                        }
                        for j in 0..RHS_N {
                            // Note: if rhs_flags[k][j] - `is_negative` is not known at comptime this is very expensive
                            if (rhs_flags[k][j]) {
                                rhs_linear[k][i] -= rhs_terms[k][j][i] as Field;
                                rhs_linear[k][i] += params.double_modulus[i] as Field;
                            } else {
                                rhs_linear[k][i] += rhs_terms[k][j][i] as Field;
                            }
                        }
                    }
                }

                for i in 0..N {
                    for j in 0..ADD_N {
                        // Note: if linear_flags[j] - `is_negative` is not known at comptime this is very expensive
                        if (linear_flags[j]) {
                            lin_expr[i] -= linear_terms[j][i] as Field;
                            lin_expr[i] += params.double_modulus[i] as Field;
                        } else {
                            lin_expr[i] += linear_terms[j][i] as Field;
                        }
                    }
                }

                (lhs_linear, rhs_linear, lin_expr)
            }

            /// Constrained version of `__compute_quadratic_expression_product`
            ///
            /// Computes the following expression in-circuit:
            ///     \sum (L_i * R_i) + \sum (A_i) - QUOTIENT * MOD
            ///
            /// Because of the subtraction of `QUOTIENT * MODULUS`, the resulting limbs may
            /// underflow and represent *negative* values. To account for this, we allow the
            /// prover to choose a sequence of borrow flags and interpret the limbs with
            /// additional terms:
            ///
            ///   - for each limb `i` we may add `2^{246}` (via a flag at position `i`);
            ///   - for each limb `i` we may subtract `2^{126 = 246 - 120}` (via the flag
            ///     at position `i - 1`).
            ///
            /// This corresponds to borrowing `2^{126}` from limb `i + 1` and adding
            /// `2^{246}` into limb `i`. After this adjustment, an honest prover can ensure
            /// that every limb lies in `[0, 2^{246})` and that the adjusted limbs encode
            /// the correct integer value.
            ///
            /// Additionally, we constrain the quotient limbs via `validate_quotient_in_range`:
            ///   - limbs `0..N-2` of `quotient` are < 2^{120};
            ///   - the top limb is < 2^{TOP_LIMB_BITS + 6}.
            /// Which validates `quotient < 2^{MOD_BITS + 6}`
            ///
            /// ## TODO
            /// Apply static or runtime checks in this method to validate that the effective
            /// `twiddle_factor` does not exceed 6 under the chosen parameters.
            fn compute_quadratic_expression_with_modulus<let N: u32, let MOD_BITS: u32, let LHS_N: u32, let RHS_N: u32, let NUM_PRODUCTS: u32, let ADD_N: u32>(
                params: BigNumParams<N, MOD_BITS>,
                lhs_terms: [[[u128; N]; LHS_N]; NUM_PRODUCTS],
                lhs_flags: [[bool; LHS_N]; NUM_PRODUCTS],
                rhs_terms: [[[u128; N]; RHS_N]; NUM_PRODUCTS],
                rhs_flags: [[bool; RHS_N]; NUM_PRODUCTS],
                linear_terms: [[u128; N]; ADD_N],
                linear_flags: [bool; ADD_N],
            ) -> [Field; 2 * N - 1] {
                // Safety: use an unconstrained function to compute the value of the quotient and borrow_flags out-of-circuit
                let (quotient, borrow_flags): ([u128; N], [bool; 2 * N - 2]) = unsafe {
                    __compute_quadratic_expression_with_borrow_flags::<N, MOD_BITS, LHS_N, RHS_N, NUM_PRODUCTS, ADD_N>(
                        params,
                        lhs_terms,
                        lhs_flags,
                        rhs_terms,
                        rhs_flags,
                        linear_terms,
                        linear_flags,
                    )
                };

                // Constrain the `quotient < 2^{MOD_BITS + 6}`
                // By constraining limbs(0..N-2) to be < 2^120 and the the top limb < 2^{TOP_LIMB_BITS + 6}
                validate_quotient_in_range::<N, MOD_BITS>(quotient);

                // Compute the linear sums that represent L_i, R_i, A
                let (lhs_linear, rhs_linear, lin_expr): ([[Field; N]; NUM_PRODUCTS], [[Field; N]; NUM_PRODUCTS], [Field; N]) = compute_linear_expressions::<N, MOD_BITS, LHS_N, RHS_N, NUM_PRODUCTS, ADD_N>(
                    params,
                    lhs_terms,
                    lhs_flags,
                    rhs_terms,
                    rhs_flags,
                    linear_terms,
                    linear_flags,
                );

                // We want to evaluate that L * R + A - Quotient * MOD = 0, evaluated over the integers
                // For this we need to be able to borrow values from more-significant limbs into less-significant limbs,
                // so that we can ensure that no limbs will underflow
                let mut expression_limbs: [Field; 2 * N - 1] = [0; 2 * N - 1];

                // Compute the product t0 * t1 + t4 - Quotient * MOD
                // TODO: this is super nasty as it requires a multiplication
                for i in 0..N {
                    for j in 0..N {
                        for k in 0..NUM_PRODUCTS {
                            if k == 0 {
                                let new_term: Field = lhs_linear[k][i] * rhs_linear[k][j]
                                    - (quotient[i] as Field) * (params.modulus[j] as Field);
                                // width-4 optimization
                                crate::as_witness(new_term);
                                expression_limbs[i + j] += new_term;
                            } else {
                                expression_limbs[i + j] += lhs_linear[k][i] * rhs_linear[k][j];
                            }
                        }
                        // This is the fallback for pure linear expression
                        if (NUM_PRODUCTS == 0) {
                            expression_limbs[i + j] -= (quotient[i] as Field) * (params.modulus[j] as Field);
                        }
                    }
                    expression_limbs[i] += lin_expr[i];
                }

                apply_borrow_flags(expression_limbs, borrow_flags)
            }

            /// Apply a precomputed borrow chain to a limb array.
            ///
            /// Given:
            ///
            ///   - `expression_limbs`: an `N`-limb array of `Field` values representing a
            ///     (possibly non-normalized) degree-2-style expression, and
            ///   - `borrow_flags[i]` indicating that we "borrow" from limb `i+1` into limb `i`,
            ///
            /// this function applies the same 2^{246}/2^{126} encoding used in
            /// `__compute_borrow_flags` to produce an adjusted limb array
            ///
            /// This matches the behavior of `__compute_borrow_flags`, which conceptually:
            ///
            ///   1. Adds `2^{246}` into a limb when a borrow is taken at that limb;
            ///   2. After scaling by `2^{-120}`, contributes `2^{126}` into the next limb.
            ///
            /// The resulting `expression_limbs` can then be passed to
            /// `validate_expression_is_zero`, which:
            ///
            ///   - right-shifts by 120 bits per limb,
            ///   - enforces a 126-bit bound,
            ///   - and propagates carries forward, finally checking that the most
            ///     significant limb is zero.
            ///
            /// ## Assumptions
            ///
            /// - `borrow_flags` was computed consistently with the original construction
            ///   of `expression_limbs` (e.g. via `__compute_borrow_flags` on the
            ///   corresponding unconstrained expression).
            /// - The caller has ensured that each adjusted limb remains < 2^{246} for
            ///   honest witnesses (e.g. via `NUM_PRODUCTS < 64`).
            ///
            /// ## Note
            ///
            /// This function does not itself enforce any bit-size bounds; it only applies
            /// the borrow encoding. The actual range checks happen later in
            /// `validate_expression_is_zero`.
            ///
            /// ## TODO
            /// define trade-offs regarding the value of borrow_shift
            /// (the larger the value, the greater the range check that is required on product_limbs)
            /// (126-bit range check is a sweet spot for the barretenberg backend as it decomposes into 9 14-bit range checks)
            /// (the barretenberg backend can evaluate these in 5.25 gates. 127 bits costs 6.5 gates)
            fn apply_borrow_flags<let N: u32>(
                mut expression_limbs: [Field; N],
                borrow_flags: [bool; N - 1],
            ) -> [Field; N] {
                let borrow_shift: Field = TWO_POW_246; // 2^{246}
                let borrow_carry: Field = TWO_POW_126; // 2^{246 - 120} = 2^{126}

                // Get the product_limbs into the form where each entry is a 246-bit value
                expression_limbs[0] += (borrow_flags[0] as Field) * borrow_shift;
                for i in 1..(N - 1) {
                    expression_limbs[i] += (borrow_flags[i] as Field) * borrow_shift
                        - (borrow_flags[i - 1] as Field) * borrow_carry;
                }
                expression_limbs[N - 1] -= (borrow_flags[N - 2] as Field) * borrow_carry;
                expression_limbs
            }

            /// Validate that `limbs` represent the integer value `0`
            ///
            /// ## Assumptions
            ///     - `limbs` is an array of `Field` values that was derived arithmetically as
            ///        a degree-2 expression
            ///     - each limb satisfies `limbs[i] < 2^{246}`, as ensured by
            ///       `compute_quadratic_expression_with_modulus` under its parameter bounds.
            ///
            /// ## Details
            /// Each element `i` in `limbs` overlaps in bit-range with element `i+1`, EXCEPT for the low 120 bits
            /// i.e. we need to do the following for each limb `i`:
            ///      1. validate the limb's low-120 bits equals zero
            ///      2. compute the limb "carry" by right-shifting by 2^{120}
            ///      3. propagate the "carry" into limb `i+1`
            /// We can efficiently do all of the above by multiplying the limb by 2^{-120} and constraining the result to be <2^{126}
            ///
            /// If the low 120 bits are nonzero, there is no value in `[0, 2^{126})` that
            /// could have produced this limb after multiplying by `2^{120}`. Since
            /// multiplication by `2^{120}` is a bijection on the Field, any limb with
            /// non-zero low 120 bits must map outside the `[0, 2^{126})` range after
            /// scaling by `2^{-120}`.
            ///
            /// The most significant limb has no limb to "carry" values into - the entire limb must equal zero
            ///
            /// ## Note
            /// The constant 126 is not arbitrary. We use 120-bit limbs and allow up to 64
            /// products per limb, which contributes at most `log2(64) = 6` bits of headroom.
            /// After scaling by `2^{-120}`, honest witnesses fit in 126 bits. We could in
            /// principle go higher (up to roughly `CircuitModulusBits - 121`), but 126 is
            /// the minimal bound consistent with `NUM_PRODUCTS < 64` and is significantly
            /// cheaper than larger bounds for the barretenberg backend.
            fn validate_expression_is_zero<let N: u32>(mut limbs: [Field; N]) {
                let hi_shift: Field = TWO_POW_120 as Field;
                let hi_downshift: Field = 1 / hi_shift;
                for i in 0..N - 1 {
                    limbs[i] *= hi_downshift;
                    crate::as_witness(limbs[i]);
                    limbs[i].assert_max_bit_size::<126>(); // N.B. is this sufficient? going beyond 126 costs us 1 gate per limb
                    limbs[i + 1] += limbs[i];
                }
                assert(limbs[N - 1] == 0);
            }

            /// Constrain a degree-2 `BigNum` expression to be equal to 0 (mod `MOD`)
            ///
            //
            /// This method is intended for relations where the remainder term of the
            /// degree-2 expression is exactly zero as an integer relation. In other words,
            /// we use it only when we expect:
            ///
            ///     \sum (L_i * R_i) + \sum (A_i) - QUOTIENT * MOD = 0
            ///
            /// as integers, not just modulo the circuit field or `BigNum` field.
            ///
            /// ## Details
            ///
            /// The constrained expression, viewed over the integers, is:
            ///
            ///       \sum_{i=0}^{NUM_PRODUCTS-1} (L_i * R_i)
            ///     + \sum_{i=0}^{ADD_N-1} (A_i)
            ///     - QUOTIENT * MOD
            ///     = 0
            ///
            /// Each `L_i`, `R_i`, `A_i` is an `N`-limb `BigNum` assembled from the
            /// `(terms, flags)` arrays. For example, for `i = 0`:
            ///
            ///     L_0 = \sum_{j=0}^{LHS_N-1} lhs[0][j]    as a `BigNum`
            ///     R_0 = \sum_{j=0}^{RHS_N-1} rhs[0][j]    as a `BigNum`
            ///
            /// The intent is to capture a generic degree-2 expression within Noir's
            /// limitations (no efficient dynamically sized vectors).
            ///
            /// The expensive parts of this algorithm are:
            ///   1. evaluating the limb products required to compute the `L_i * R_i` values;
            ///   2. applying range constraints to validate that the result encodes 0.
            ///
            /// ## Note
            /// When the expression is evaluated over `N`-limb `BigNum` values, the product
            /// has up to `2N - 1` significant limbs. Each limb is a sum of at most
            /// `NUM_PRODUCTS` products of `linear` 120-bit limbs and `quadratic` 240-bit-ish limbs
            ///
            /// Note that:
            ///   * limb-wise multiplication is not uniform across indices. For example:
            ///
            ///       [x0, x1, x2] * [y0, y1, y2] =
            ///       [
            ///         x0*y0,                               // 1 term
            ///         x0*y1 + x1*y0,                       // 2 terms
            ///         x0*y2 + x1*y1 + x2*y0,               // 3 terms
            ///         x1*y2 + x2*y1,                       // 2 terms
            ///         x2*y2                                // 1 term
            ///       ]
            ///       The number of partial products per limb grows linearly from 1 to N,
            ///       then decreases linearly back to 1.
            ///
            ///   * we also allow linear combinations inside the products, which further increases the bound
            ///     on a `quadratic` limb
            ///
            ///   * finally, when `is_negative` flag is set, we effectively add 2 * MOD to each limb, where
            ///     2 * MOD limbs are structured in such a way that they all are > 2**120
            ///
            /// We allow `NUM_PRODUCTS < 64` completeness-wise, but it certainly can overflow the 2^{240 + `twiddle_factor=6`} bound
            /// in edge cases. See completeness section for an example.
            ///
            /// ## Assumptions
            ///
            /// Each `BigNum` value used in this gadget is already range constrained:
            ///   - limbs `0..N-2` satisfy `limb_i < 2^{120}`
            ///   - limb `N-1` satisfies `limb_{N-1} < 2^{TOP_LIMB_BITS}`
            ///
            /// ## Completeness
            /// If an honest prover supplies inputs that satisfy the assumptions above,
            /// it can always find a `quotient` and `borrow_flags` such that the integer
            /// relation holds and all constraints are satisfied.
            ///
            /// The only possibility to break completeness is by providing too many inputs, for example:
            ///
            ///     (a0 + a1 + a2) * (b0 + b1 + b2) + ... (60 times), with each a_i having its limbs at maximum value of 2^{120} - 1
            ///     The middle limb will contain a value that will definitely overflow the 2^{246} bound.
            ///
            /// ## Soundness
            /// This function is conditionally sound: it enforces that the degree-2 relation
            /// holds modulo `MOD`, but it does not, by itself, enforce that any particular
            /// term (for example a result `z`) is a *canonical* representative in
            /// `[0, MOD)`.
            ///
            /// In other words, if the surrounding relation is invariant under adding a
            /// multiple of `MOD` to one of its terms, then a dishonest prover can exploit
            /// this. For example, consider a multiplication relation:
            ///
            ///     x * y - z = 0 (mod MOD)
            ///
            /// Internally we encode this as:
            ///
            ///     x * y + 2 * MOD - z - quotient * MOD = 0
            ///
            /// which is equivalent to:
            ///
            ///     x * y - z = (quotient - 2) * MOD
            ///
            /// Suppose the honest witness uses some `z` satisfying
            /// `0 <= z < MOD` and some `quotient`. If the `BigNum` encoding allows
            /// `z' = z + MOD` (i.e. `z' < 2^{MOD_BITS}` still holds), then a dishonest
            /// prover can instead provide:
            ///
            ///     z' = z + MOD
            ///     quotient' = quotient - 1
            ///
            /// and still satisfy:
            ///
            ///     x * y + 2 * MOD - z' - quotient' * MOD = 0
            ///
            /// even though `z'` is no longer the canonical representative of `x * y mod MOD`.
            ///
            /// The same consideration applies to almost every constrained `BigNum` relation:
            /// whenever a value participates *only* through a modular equality, and no
            /// separate range constraint is imposed on that value, the prover is free to
            /// shift it by an extra `MOD` as long as the resulting limb
            /// encoding still satisfies its bit-bounds. This is inherent in working with
            /// modular constraints; the responsibility for enforcing canonical
            /// representatives lies with the caller when it is required.
            ///
            /// This is the same "extra modulus" phenomenon as in the `add`/`sub` functions:
            /// the constraints are sound for modular arithmetic, but any caller that
            /// requires canonical outputs in `[0, MOD)` must additionally enforce a
            /// range check (for example via `validate_in_field`) on the relevant terms.
            pub(crate) fn evaluate_quadratic_expression<let N: u32, let MOD_BITS: u32, let LHS_N: u32, let RHS_N: u32, let NUM_PRODUCTS: u32, let ADD_N: u32>(
                params: BigNumParams<N, MOD_BITS>,
                lhs_terms: [[[u128; N]; LHS_N]; NUM_PRODUCTS],
                lhs_flags: [[bool; LHS_N]; NUM_PRODUCTS],
                rhs_terms: [[[u128; N]; RHS_N]; NUM_PRODUCTS],
                rhs_flags: [[bool; RHS_N]; NUM_PRODUCTS],
                linear_terms: [[u128; N]; ADD_N],
                linear_flags: [bool; ADD_N],
            ) {
                assert(NUM_PRODUCTS < 64, f"evaluate_quadratic_expression overflow in operands count");
                // NUM_PRODUCTS < 64 is a light bound that tries to ensure each limb sum < 2^{246} so that the 126-bit bound is valid.

                lhs_terms.for_each(|lhs_limbs: [[u128; N]; LHS_N]| {
                    lhs_limbs.for_each(|term: [u128; N]| validate_in_range::<u128, N, MOD_BITS>(term))
                });
                rhs_terms.for_each(|rhs_limbs: [[u128; N]; RHS_N]| {
                    rhs_limbs.for_each(|term: [u128; N]| validate_in_range::<u128, N, MOD_BITS>(term))
                });
                linear_terms.for_each(|term: [u128; N]| validate_in_range::<u128, N, MOD_BITS>(term));

                let expression_limbs: [Field; 2 * N - 1] = compute_quadratic_expression_with_modulus::<N, MOD_BITS, LHS_N, RHS_N, NUM_PRODUCTS, ADD_N>(
                    params,
                    lhs_terms,
                    lhs_flags,
                    rhs_terms,
                    rhs_flags,
                    linear_terms,
                    linear_flags,
                );
                validate_expression_is_zero(expression_limbs);
            }

            // ------------------------------ UDIV MOD EXPRESSION ------------------------------

            /// Given a `udiv_mod` `BigNum` expression that is equal to `0` over integers, compute the borrow flags (unconstrained)
            ///
            /// Mirror function of `__compute_quadratic_expression_with_borrow_flags` optimized to compute borrow flags of an expression:
            ///     divisor * quotient + remainder - numerator = 0
            /// see `__compute_quadratic_expression_with_borrow_flags` for details
            ///
            /// The main differences from it are:
            ///     1. `product_limbs` stores only the least-significant `N` limbs of
            ///         `quotient * divisor + remainder`. This is sufficient to compute the
            ///         borrow flags for the first `N` limbs of
            ///             quotient * divisor + remainder - numerator.
            ///
            ///         For an honest `udiv_mod` relation we also have
            ///             quotient * divisor <= numerator < B^N,
            ///         so the true product fits into `N` limbs as an integer.
            ///     2. Instead of subtracting `quotient * MOD` we subtract `numerator`. This is due to the fact that
            ///        we no longer work over `MOD`, and we can't really do subtractions as we did previously: `double_modulus - x`
            ///
            /// ## Note
            /// We leave the borrow values at 2^{246}, even though we should never reach this bound with just 3 terms
            /// The cases where it can happen are: N >= 64 (middle limb will have 64 additions). And it is a pure completeness issue
            /// But the rest of the library will probably not work with that massive number anyway
            unconstrained fn __compute_udiv_mod_expression_with_borrow_flags<let N: u32, let MOD_BITS: u32>(
                numerator: [u128; N],
                divisor: [u128; N],
                quotient: [u128; N],
                remainder: [u128; N],
            ) -> [bool; N - 1] {
                let mut product_limbs: [Field; N] = [0; N];
                let mut numerator_field: [Field; N] = [0; N];
                for i in 0..N {
                    for j in 0..N - i {
                        product_limbs[i + j] += (quotient[i] as Field) * (divisor[j] as Field);
                    }
                    product_limbs[i] += (remainder[i] as Field);

                    numerator_field[i] = numerator[i] as Field;
                }

                __compute_borrow_flags(product_limbs, numerator_field)
            }

            /// Constrained version of `__compute_udiv_mod_expression_with_borrow_flags`
            ///
            /// Computes the following expression in-circuit:
            ///     quotient * divisor + remainder - numerator = 0
            ///
            /// Mirror function of `compute_quadratic_expression_with_modulus`. See it for details.
            ///
            /// ## Soundness note
            /// We compute the full convolution `quotient * divisor` into `2N - 1` limbs
            /// and then constrain all limbs with index `i >= N` to be zero. For `i >= N`
            /// the value `expression_limbs_full[i]` is a sum of products
            ///     sum_{j+k=i} quotient[j] * divisor[k]
            /// with no contribution from `numerator` or `remainder`.
            ///
            /// Every limb of `quotient` and `divisor` is range-constrained to be a 120-bit
            /// integer, so each product term is < 2^{240} and each coefficient of the
            /// convolution is strictly smaller than the field modulus. In this regime,
            /// the constraint `expression_limbs_full[i] == 0` in `Field` coincides with
            /// the same equality over the integers.
            ///
            /// Vanishing of all high limbs `i >= N` is therefore an integer statement that
            /// the product has degree < N, i.e.
            ///     quotient * divisor < 2^{120 * N},
            /// so `quotient * divisor` fits into `N` 120-bit limbs and does not overflow.
            fn compute_udiv_mod_expression<let N: u32, let MOD_BITS: u32>(
                numerator: [u128; N],
                divisor: [u128; N],
                quotient: [u128; N],
                remainder: [u128; N],
            ) -> [Field; N] {
                // Safety: use an unconstrained function to compute the value of the quotient and borrow_flags out-of-circuit
                let borrow_flags: [bool; N - 1] = unsafe {
                    __compute_udiv_mod_expression_with_borrow_flags::<N, MOD_BITS>(
                        numerator,
                        divisor,
                        quotient,
                        remainder,
                    )
                };

                let mut expression_limbs_full: [Field; 2 * N - 1] = [0; 2 * N - 1];
                for i in 0..N {
                    for j in 0..N {
                        expression_limbs_full[i + j] += (quotient[i] as Field) * (divisor[j] as Field);
                    }
                    expression_limbs_full[i] += (remainder[i] as Field) - (numerator[i] as Field);
                }

                let mut expression_limbs: [Field; N] = [0; N];
                for i in 0..N {
                    expression_limbs[i] = expression_limbs_full[i];
                }
                for i in N..2 * N - 1 {
                    assert(expression_limbs_full[i] == 0);
                }

                apply_borrow_flags(expression_limbs, borrow_flags)
            }

            /// Constrain a `udiv_mod` `BigNum` expression to be equal to 0
            ///
            /// Mirror function of `evaluate_quadratic_expression`
            ///
            /// ## Details
            ///
            /// The constrained expression, viewed over the integers, is:
            ///
            ///     quotient * divisor + remainder - numerator = 0
            ///
            /// ## Completeness
            /// If an honest prover supplies valid `BigNum` inputs that satisfy the equation,
            /// it can always find `borrow_flags` such that the integer
            /// relation holds and all constraints are satisfied.
            ///
            /// The only possibility to break completeness is by providing a `BigNum` with `N >= 64`
            /// See `__compute_udiv_mod_expression_with_borrow_flags` for details
            ///
            /// ## Soundness
            /// This function is conditionally sound: it enforces that the degree-2 relation
            /// holds over the integers, but it does not, by itself, enforce that
            /// this relation is unique for given `numerator` and `divisor`
            ///
            /// For example, if the true relation is
            ///     quotient * divisor + remainder - numerator = 0
            ///
            /// then we can set quotient' = quotient - 1, remainder' = remainder + divisor
            pub(crate) fn validate_udiv_mod_expression<let N: u32, let MOD_BITS: u32>(
                numerator: [u128; N],
                divisor: [u128; N],
                quotient: [u128; N],
                remainder: [u128; N],
            ) {
                validate_in_range::<u128, N, MOD_BITS>(numerator);
                validate_in_range::<u128, N, MOD_BITS>(divisor);
                validate_in_range::<u128, N, MOD_BITS>(quotient);
                validate_in_range::<u128, N, MOD_BITS>(remainder);

                let expression_limbs: [Field; N] =
                    compute_udiv_mod_expression::<N, MOD_BITS>(numerator, divisor, quotient, remainder);
                validate_expression_is_zero(expression_limbs);
            }
        }

        pub mod serialization {

            use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
            use crate::mavros::vendor::bignum::utils::map::invert_array;

            // Conversions between big endian and little endian byte arrays and BigNum instances
            // the byte serialization should have `(MOD_BITS + 7) / 8` bytes.
            // each 120-bit limb is represented by 15 bytes, and there are fewer bytes for covering the most significant limb

            /// Construct a `BigNum` value from a big-endian byte array.
            ///
            /// The input encodes an integer in base 256, which we split into `N` 120-bit limbs.
            ///
            /// ## Note
            /// We only enforce that the value is < 2^MOD_BITS. We do not enforce that it is
            /// reduced modulo the field modulus.
            ///
            /// Consistency between `N` and `MOD_BITS` is expected:
            ///     - `N * 15 >= num_bytes`
            ///     - `num_bytes > (N - 1) * 15`
            ///
            /// Enforcing range constraints on each limb is crucial for efficiency.
            /// In principle, accumulating `u8` values already bounds the integer,
            /// but relying on Noir to infer a `u128` from a large linear combination
            /// would trigger a very general (and expensive) range checks
            pub fn from_be_bytes<let N: u32, let MOD_BITS: u32>(x: [u8; (MOD_BITS + 7) / 8]) -> [u128; N] {
                let num_bytes: u32 = (MOD_BITS + 7) / 8;

                let mut result: [u128; N] = [0; N];

                let excess_bytes: u32 = N * 15 - num_bytes;
                let last_limb_bytes: u32 = 15 - excess_bytes;

                let mut limb: Field = 0;
                for i in 0..last_limb_bytes {
                    limb *= 256;
                    limb += x[i] as Field;
                }

                limb.assert_max_bit_size::<120>();
                result[N - 1] = limb as u128;

                let mut byte_ptr: u32 = last_limb_bytes;
                for i in 1..N {
                    let mut limb: Field = 0;
                    for _ in 0..15 {
                        limb *= 256;
                        limb += x[byte_ptr] as Field;
                        byte_ptr += 1;
                    }
                    limb.assert_max_bit_size::<120>();
                    result[N - i - 1] = limb as u128;
                }

                if (MOD_BITS % 8 != 0) {
                    let most_significant_byte: Field = x[0] as Field;
                    most_significant_byte.assert_max_bit_size::<MOD_BITS % 8>();
                }
                result
            }

            /// Construct a big-endian byte array from a `BigNum` value.
            ///
            /// The output contains `(MOD_BITS + 7) / 8` bytes. We serialize the most
            /// significant limb first, which may occupy fewer than 15 bytes, followed by
            /// the remaining full 15-byte limbs in big-endian order.
            ///
            /// Consistency between `N` and `MOD_BITS` is expected:
            ///     - the most significant limb contributes `((MOD_BITS + 7) / 8) - (N - 1) * 15` bytes;
            ///     - all other limbs are serialized as full 15-byte chunks.
            pub fn to_be_bytes<let N: u32, let MOD_BITS: u32>(val: [u128; N]) -> [u8; (MOD_BITS + 7) / 8] {
                let mut result: [u8; (MOD_BITS + 7) / 8] = [0; (MOD_BITS + 7) / 8];

                let last_limb_num_bytes: u32 = (MOD_BITS + 7) / 8 - (N - 1) * 15;
                let mut byte_ptr: u32 = last_limb_num_bytes;
                // First we deal with the full limbs
                for i in 0..N - 1 {
                    let idx: u32 = (N - 1) - i - 1;
                    let limb_bytes: [u8; 15] = (val[idx] as Field).to_be_bytes();
                    for j in 0..15 {
                        result[byte_ptr] = limb_bytes[j];
                        byte_ptr += 1;
                    }
                }

                let last_limb_bytes: [u8; ((MOD_BITS + 7) / 8 - (N - 1) * 15)] =
                    (val[N - 1] as Field).to_be_bytes();
                for i in 0..last_limb_num_bytes {
                    result[i] = last_limb_bytes[i];
                }

                result
            }

            /// Construct a `BigNum` value from little-endian byte array
            ///
            /// Reverse an array and apply `from_be_bytes`
            ///
            /// See `from_be_bytes` for details
            pub fn from_le_bytes<let N: u32, let MOD_BITS: u32>(x: [u8; (MOD_BITS + 7) / 8]) -> [u128; N] {
                let be_x: [u8; (MOD_BITS + 7) / 8] = invert_array(x);
                from_be_bytes(be_x)
            }

            /// Construct a little-endian byte array from a `BigNum` value
            ///
            /// Apply `to_be_bytes` and reverse an array
            ///
            /// See `to_be_bytes` for details
            pub fn to_le_bytes<let N: u32, let MOD_BITS: u32>(val: [u128; N]) -> [u8; (MOD_BITS + 7) / 8] {
                let result_be: [u8; (MOD_BITS + 7) / 8] = to_be_bytes(val);
                invert_array(result_be)
            }
        }

    }

    pub mod bignum {

        use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
        use crate::mavros::vendor::bignum::fns::constrained_ops::limbs_to_field;
        use crate::mavros::vendor::bignum::params::BigNumParams;
        use crate::ops::{Add, Div, Mul, Neg, Sub};

        /// The core trait for arbitrary-precision modular arithmetic.
        ///
        /// `BigNum` represents integers modulo a compile-time known modulus. Values are stored as
        /// arrays of 120-bit limbs in little-endian format.
        ///
        /// # Important Notes
        ///
        /// - **Single-limb BigNum is not supported**: This library assumes at least two 120-bit limbs
        ///   (modulus >= 2^120). Operations on single-limb BigNums may produce incorrect results.
        ///
        /// - **Constrained vs Unconstrained**: Methods prefixed with `__` are unconstrained and do not
        ///   generate circuit constraints. Use them for witness computation, then constrain results with
        ///   `evaluate_quadratic_expression` for optimal performance.
        ///
        /// - **Performance**: The overloaded operators (`+`, `-`, `*`, `/`) are constrained and
        ///   expensive. For multiple operations, prefer using unconstrained methods followed by
        ///   `evaluate_quadratic_expression`.
        ///
        /// - **Modular vs Integer semantics**: Most operations are modular (mod `MOD`). For integer
        ///   division/remainder, use `udiv`, `umod`, `udiv_mod`.
        ///
        /// # Example
        /// ```
        /// use dep::bignum::U256;
        /// use dep::bignum::BigNum;
        ///
        /// fn example(a: U256, b: U256) -> U256 {
        ///     // Constrained multiplication
        ///     a * b
        /// }
        /// ```
        pub trait BigNum: Neg + Add + Sub + Mul + Div + crate::cmp::Eq {
            /// Number of 120-bit limbs used to represent this BigNum.
            let N: u32;
            /// Number of bits in the modulus.
            let MOD_BITS: u32;

            /// Returns the parameter set for this BigNum type.
            ///
            /// The parameters include the modulus, precomputed reduction constants, and whether
            /// the modulus is prime (has multiplicative inverses).
            fn params() -> BigNumParams<N, MOD_BITS>;

            /// Returns the number of bits in the modulus.
            ///
            /// Note: Takes a dummy `Self` parameter for trait method resolution.
            fn modulus_bits(_: Self) -> u32;

            /// Returns the number of limbs used to represent this BigNum.
            ///
            /// Note: Takes a dummy `Self` parameter for trait method resolution.
            fn num_limbs(_: Self) -> u32;

            /// Returns the modulus as a BigNum value.
            ///
            /// # Note
            /// The modulus itself is considered equivalent to zero in modular arithmetic.
            /// `is_zero(modulus())` returns `true`.
            fn modulus() -> Self;

            /// Creates a new BigNum with value 0.
            ///
            /// Equivalent to `zero()`.
            fn new() -> Self;

            /// Returns the additive identity (0).
            fn zero() -> Self;

            /// Returns the multiplicative identity (1).
            fn one() -> Self;

            /// Constructs a BigNum from raw limbs.
            ///
            /// # Safety
            /// **FOOTGUN**: The limbs are NOT validated. You MUST call `validate_in_range()` on the
            /// result if the limbs come from untrusted input, otherwise malicious witnesses could
            /// bypass range checks.
            ///
            /// Each limb should be at most 120 bits, except the top limb which should be at most
            /// `MOD_BITS - 120 * (N - 1)` bits.
            ///
            /// # Example
            /// ```
            /// let bn = MyBigNum::from_limbs([1, 0, 0]);
            /// bn.validate_in_range(); // Always validate untrusted input!
            /// ```
            fn from_limbs(limbs: [u128; N]) -> Self;

            /// Returns the raw limb representation.
            ///
            /// Limbs are in little-endian order: `limbs[0]` contains the least significant 120 bits.
            fn get_limbs(self) -> [u128; N];

            /// Sets a specific limb to a new value.
            ///
            /// # Safety
            /// **FOOTGUN**: The new limb value is NOT validated. You MUST call `validate_in_range()`
            /// after modification if the value comes from untrusted input.
            fn set_limb(self: &mut Self, idx: u32, value: u128);

            /// Returns the limb at the given index.
            fn get_limb(self: Self, idx: u32) -> u128;

            /// Generates a deterministic pseudorandom BigNum from a seed.
            ///
            /// Uses Poseidon2 hash to derive `MOD_BITS * 2` bits of entropy, then reduces modulo `MOD`.
            ///
            /// # Security Warning
            /// **NOT cryptographically secure for all use cases**. The output is deterministic and
            /// uniformly distributed modulo `MOD`, but the seed-to-output mapping is publicly computable.
            /// Do not use for secret key generation where the seed might be guessable.
            ///
            /// # Constrained
            /// This is a constrained function that generates range check constraints.
            fn derive_from_seed<let SeedBytes: u32>(seed: [u8; SeedBytes]) -> Self;

            /// Unconstrained version of `derive_from_seed`.
            ///
            /// # Safety
            /// No constraints are generated. Use only for witness computation in unconstrained contexts.
            unconstrained fn __derive_from_seed<let SeedBytes: u32>(seed: [u8; SeedBytes]) -> Self;

            /// Constructs a BigNum from big-endian bytes.
            ///
            /// The byte array length must be exactly `(MOD_BITS + 7) / 8`.
            ///
            /// # Panics
            /// Panics if the value represented by the bytes is >= 2^MOD_BITS.
            fn from_be_bytes(x: [u8; (MOD_BITS + 7) / 8]) -> Self;

            /// Converts to big-endian byte representation.
            fn to_be_bytes(self) -> [u8; (MOD_BITS + 7) / 8];

            /// Constructs a BigNum from little-endian bytes.
            ///
            /// The byte array length must be exactly `(MOD_BITS + 7) / 8`.
            ///
            /// # Panics
            /// Panics if the value represented by the bytes is >= 2^MOD_BITS.
            fn from_le_bytes(x: [u8; (MOD_BITS + 7) / 8]) -> Self;

            /// Converts to little-endian byte representation.
            fn to_le_bytes(self) -> [u8; (MOD_BITS + 7) / 8];

            /// Unconstrained equality check.
            ///
            /// Returns `true` if `self == other` as integers (not modular equality).
            ///
            /// # Safety
            /// No constraints are generated. Use `==` operator for constrained equality.
            unconstrained fn __eq(self: Self, other: Self) -> bool;

            /// Unconstrained zero check.
            ///
            /// Returns `true` if all limbs are zero.
            ///
            /// # Note
            /// This checks for integer zero only, not modular zero.
            /// `modulus().__is_zero()` returns `false` even though `modulus == 0 (mod MOD)`.
            ///
            /// # Safety
            /// No constraints are generated. Use `is_zero()` for constrained check.
            unconstrained fn __is_zero(self: Self) -> bool;

            /// Unconstrained negation: computes `-self (mod MOD)`.
            ///
            /// # Note
            /// When `self` is zero, this returns the modulus (which is equivalent to zero mod MOD).
            ///
            /// # Safety
            /// No constraints are generated. Constrain the result using `evaluate_quadratic_expression`.
            unconstrained fn __neg(self) -> Self;

            /// Unconstrained addition: computes `self + other (mod MOD)`.
            ///
            /// # Safety
            /// No constraints are generated. Constrain the result using `evaluate_quadratic_expression`.
            unconstrained fn __add(self, other: Self) -> Self;

            /// Unconstrained subtraction: computes `self - other (mod MOD)`.
            ///
            /// # Safety
            /// No constraints are generated. Constrain the result using `evaluate_quadratic_expression`.
            unconstrained fn __sub(self, other: Self) -> Self;

            /// Unconstrained multiplication: computes `self * other (mod MOD)`.
            ///
            /// # Safety
            /// No constraints are generated. Constrain the result using `evaluate_quadratic_expression`.
            unconstrained fn __mul(self, other: Self) -> Self;

            /// Unconstrained squaring: computes `self * self (mod MOD)`.
            ///
            /// # Safety
            /// No constraints are generated. Constrain the result using `evaluate_quadratic_expression`.
            unconstrained fn __sqr(self) -> Self;

            /// Unconstrained modular division: computes `self * other^{-1} (mod MOD)`.
            ///
            /// # Requirements
            /// - `MOD` must be prime (`params.has_multiplicative_inverse == true`)
            /// - `other` must not be zero
            ///
            /// # Performance Warning
            /// **EXPENSIVE**: Requires modular exponentiation during witness computation.
            /// Consider using `batch_invert` for multiple divisions.
            ///
            /// # Safety
            /// No constraints are generated. Constrain the result using `evaluate_quadratic_expression`.
            unconstrained fn __div(self, other: Self) -> Self;

            /// Unconstrained integer division with remainder.
            ///
            /// Returns `(quotient, remainder)` where `self == quotient * divisor + remainder`
            /// and `0 <= remainder < divisor`.
            ///
            /// # Note
            /// This is INTEGER division, not modular. Use when `MOD` is not prime or when you
            /// need the actual quotient/remainder, not the modular inverse.
            ///
            /// # Safety
            /// No constraints are generated. Use `udiv_mod()` for the constrained version.
            unconstrained fn __udiv_mod(self, divisor: Self) -> (Self, Self);

            /// Unconstrained modular inverse: computes `self^{-1} (mod MOD)`.
            ///
            /// # Requirements
            /// - `MOD` must be prime (`params.has_multiplicative_inverse == true`)
            /// - `self` must not be zero
            ///
            /// # Performance Warning
            /// **EXPENSIVE**: Requires modular exponentiation. For multiple inverses, use `batch_invert`.
            ///
            /// # Safety
            /// No constraints are generated.
            unconstrained fn __invmod(self) -> Self;

            /// Unconstrained modular exponentiation: computes `self^exponent (mod MOD)`.
            ///
            /// # Performance Warning
            /// **EXPENSIVE**: Complexity is O(MOD_BITS) multiplications.
            ///
            /// # Safety
            /// No constraints are generated.
            unconstrained fn __pow(self, exponent: Self) -> Self;

            /// **Deprecated**: use `__sqrt` instead.
            unconstrained fn __tonelli_shanks_sqrt(self) -> crate::option::Option<Self>;

            /// Unconstrained modular square root.
            ///
            /// Returns `Some(root)` where `root * root == self (mod MOD)`, or `None` if no square root exists.
            ///
            /// # Requirements
            /// - `MOD` must be prime (`params.has_multiplicative_inverse == true`)
            ///
            /// # Note
            /// When a square root exists, there are always two: `root` and `-root (mod MOD)`.
            /// This function returns one of them (implementation-defined which one).
            ///
            /// # Safety
            /// No constraints are generated.
            unconstrained fn __sqrt(self) -> crate::option::Option<Self>;

            /// Asserts that `self != other (mod MOD)`.
            ///
            /// # Performance Note
            /// This is cheaper than `assert(self != other)` because it avoids creating a boolean result.
            ///
            /// # Soundness
            /// Sound for `MOD < circuit_field_modulus`. For very large moduli approaching the circuit
            /// field size, there's a negligible (~3/p) probability of false positives.
            fn assert_is_not_equal(self: Self, other: Self);

            /// Returns `true` if `self == 0 (mod MOD)`.
            ///
            /// # Note
            /// This treats both the zero limb vector AND the modulus itself as zero, since
            /// `modulus == 0 (mod MOD)`.
            ///
            /// # Performance Note
            /// Cheaper than `self == BigNum::zero()`.
            fn is_zero(self) -> bool;

            /// Returns `true` if all limbs are zero (integer zero, not modular).
            ///
            /// # Note
            /// Unlike `is_zero()`, this returns `false` for the modulus value even though
            /// `modulus == 0 (mod MOD)`.
            fn is_zero_integer(self) -> bool;

            /// Asserts that `self != 0 (mod MOD)`.
            ///
            /// # Performance Note
            /// Cheaper than `assert(!self.is_zero())` or `assert(self != BigNum::zero())`.
            fn assert_is_not_zero(self);

            /// Asserts that at least one limb is non-zero (integer assertion).
            ///
            /// # Note
            /// Unlike `assert_is_not_zero()`, this will pass for the modulus value.
            ///
            /// # Performance Note
            /// Cheaper than `assert(!self.is_zero_integer())`.
            fn assert_is_not_zero_integer(self);

            /// Validates that each limb is properly range-constrained.
            ///
            /// Ensures:
            /// - Limbs 0 to N-2 are < 2^120
            /// - Limb N-1 is < 2^(MOD_BITS - 120*(N-1))
            ///
            /// # When to Use
            /// Call this after `from_limbs()` or `set_limb()` when the input comes from untrusted sources.
            ///
            /// # Performance Note
            /// Range constraints are deduplicated per value, so calling this multiple times on the same
            /// BigNum does not add duplicate constraints.
            ///
            /// # Note
            /// This does NOT guarantee `self < MOD`. Use `validate_in_field()` for that.
            fn validate_in_range(self);

            /// Validates that `self <= MOD`.
            ///
            /// # Note
            /// - This allows `self == MOD` (which equals 0 in modular arithmetic), consistent with
            ///   the library's treatment of the modulus as a valid representation of zero.
            /// - This is a STRONGER check than `validate_in_range()`.
            /// - Unlike `validate_in_range()`, this check is NOT deduplicated. Repeated calls will
            ///   add redundant constraints.
            fn validate_in_field(self);

            /// Constrained squaring: computes `self * self (mod MOD)`.
            ///
            /// # Performance Note
            /// For multiple operations, prefer using `__sqr()` followed by `evaluate_quadratic_expression`.
            fn sqr(self) -> Self;

            /// Constrained integer division with remainder.
            ///
            /// Returns `(quotient, remainder)` where `self == quotient * divisor + remainder`
            /// and `0 <= remainder < divisor`.
            ///
            /// # Performance Warning
            /// **EXPENSIVE**: Requires long binary division.
            ///
            /// # Panics
            /// Will fail constraint if `divisor == 0`.
            ///
            /// # Note
            /// This is INTEGER division. For modular division on prime fields, use the `/` operator.
            fn udiv_mod(self, divisor: Self) -> (Self, Self);

            /// Constrained integer division: returns `floor(self / divisor)`.
            ///
            /// # Performance Warning
            /// **EXPENSIVE**: Internally calls `udiv_mod()`.
            ///
            /// # Note
            /// This is INTEGER division, not modular. The remainder is discarded.
            fn udiv(self, divisor: Self) -> Self;

            /// Constrained integer modulo: returns `self % divisor`.
            ///
            /// # Performance Warning
            /// **EXPENSIVE**: Internally calls `udiv_mod()`.
            ///
            /// # Note
            /// This is INTEGER modulo. The result is the remainder after integer division.
            fn umod(self, divisor: Self) -> Self;
        }

        // We need macros that implement the BigNum, Default, From, Neg, Add, Sub, Mul, Div, Eq, Ord traits for each bignum type
        pub comptime fn derive_bignum(
            strukt: TypeDefinition,
            N: u32,
            MOD_BITS: u32,
            params: Quoted,
        ) -> Quoted {
            let typ = strukt.as_type();
            quote {

                // implement BigNum for BigNum 
                impl $crate::mavros::vendor::bignum::BigNum for $typ {
                    let N: u32 = $N; 
                    let MOD_BITS: u32 = $MOD_BITS;
            
                    fn modulus_bits(_: Self) -> u32 {
                        $MOD_BITS
                    }
            
                    fn num_limbs(_: Self) -> u32 {
                        $N
                    }

                    fn modulus() -> Self {
                        Self { limbs: Self::params().modulus }
                    }

                    fn params() -> $crate::mavros::vendor::bignum::params::BigNumParams<$N, $MOD_BITS> {
                        $params
                    }

                    fn new() -> Self {
                        Self {limbs: [0; $N]}
                    }

                    /// Note: You have to properly constrain the limbs prior calling this method
                    fn from_limbs(limbs: [u128; $N]) -> Self {
                        Self { limbs }
                    }

                    fn get_limbs(self: Self) -> [u128; $N] {
                        self.limbs
                    }

                    /// Note: You have to properly constrain the limb prior calling this method
                    fn set_limb(self: &mut Self, idx: u32, value: u128) {
                        self.limbs[idx] = value;
                    }

                    fn get_limb(self: Self, idx: u32) -> u128 {
                        self.limbs[idx]
                    }

                    fn zero() -> Self {
                        Self { limbs: [0; $N] }
                    }

                    fn one() -> Self {
                        let mut limbs = [0; $N];
                        limbs[0] = 1;
                        Self { limbs }
                    }

                    fn derive_from_seed<let SeedBytes: u32>(seed: [u8; SeedBytes]) -> Self {
                        let params = Self::params();
                        $typ::from_limbs($crate::mavros::vendor::bignum::internal::derive_from_seed::<$N, $MOD_BITS, SeedBytes>(params, seed))
                    }

                    unconstrained fn __derive_from_seed<let SeedBytes: u32>(seed: [u8; SeedBytes]) -> Self {
                        let params = Self::params();
                        Self { limbs: $crate::mavros::vendor::bignum::internal::__derive_from_seed::<$N, $MOD_BITS, SeedBytes>(params, seed) }
                    }

                    fn from_be_bytes(x: [u8; ($MOD_BITS + 7) / 8]) -> Self {
                        Self { limbs: $crate::mavros::vendor::bignum::internal::from_be_bytes::<$N, $MOD_BITS>(x) }
                    }

                    fn to_be_bytes(self) -> [u8; ($MOD_BITS + 7) / 8] {
                        $crate::mavros::vendor::bignum::internal::to_be_bytes::<$N, $MOD_BITS>(self.limbs)
                    }

                    fn from_le_bytes(x: [u8; ($MOD_BITS + 7) / 8]) -> Self {
                        Self { limbs: $crate::mavros::vendor::bignum::internal::from_le_bytes::<$N, $MOD_BITS>(x) }
                    }

                    fn to_le_bytes(self) -> [u8; ($MOD_BITS + 7) / 8] {
                        $crate::mavros::vendor::bignum::internal::to_le_bytes::<$N, $MOD_BITS>(self.limbs)
                    }

                    unconstrained fn __eq(self: Self, other: Self) -> bool {
                        $crate::mavros::vendor::bignum::internal::__eq(self.get_limbs(), other.get_limbs())
                    }

                    unconstrained fn __is_zero(self: Self) -> bool {
                        $crate::mavros::vendor::bignum::internal::__is_zero(self.get_limbs())
                    }

                    unconstrained fn __neg(self: Self) -> Self {
                        let params = Self::params();
                        Self {limbs: $crate::mavros::vendor::bignum::internal::__neg(params.modulus, self.get_limbs())}
                    }

                    unconstrained fn __add(self: Self, other: Self) -> Self {
                        let params = Self::params();
                        Self {limbs: $crate::mavros::vendor::bignum::internal::__add(params.modulus, self.get_limbs(), other.get_limbs())}
                    }

                    unconstrained fn __sub(self: Self, other: Self) -> Self {
                        let params = Self::params();
                        Self {limbs: $crate::mavros::vendor::bignum::internal::__sub(params.modulus, self.get_limbs(), other.get_limbs())}
                    }

                    unconstrained fn __mul(self: Self, other: Self) -> Self {
                        let params = Self::params();
                        Self {limbs: $crate::mavros::vendor::bignum::internal::__mul(params, self.get_limbs(), other.get_limbs())}
                    }

                    unconstrained fn __sqr(self: Self) -> Self {
                        let params = Self::params();
                        Self {limbs: $crate::mavros::vendor::bignum::internal::__sqr(params, self.get_limbs()) }
                    }

                    unconstrained fn __div(self: Self, divisor: Self) -> Self {
                        let params = Self::params();
                        if $params.has_multiplicative_inverse {
                            Self { limbs: $crate::mavros::vendor::bignum::internal::__div(params, self.get_limbs(), divisor.get_limbs()) }
                        } else {
                            Self { limbs: $crate::mavros::vendor::bignum::internal::__udiv_mod(self.get_limbs(), divisor.get_limbs()).0 }
                        }
                    }

                    unconstrained fn __udiv_mod(self: Self, divisor: Self) -> (Self, Self) {
                        let (q, r) = $crate::mavros::vendor::bignum::internal::__udiv_mod(self.get_limbs(), divisor.get_limbs());
                        (Self{limbs: q}, Self{limbs: r})
                    }

                    unconstrained fn __invmod(self: Self) -> Self {
                        let params = Self::params();
                        Self {limbs: $crate::mavros::vendor::bignum::internal::__invmod(params, self.get_limbs())}
                    }

                    unconstrained fn __pow(self: Self, exponent: Self) -> Self {
                        let params = Self::params();
                        Self {limbs: $crate::mavros::vendor::bignum::internal::__pow(params, self.get_limbs(), exponent.get_limbs())}
                    }

                    #[deprecated("use __sqrt")]
                    unconstrained fn __tonelli_shanks_sqrt(self: Self) -> crate::option::Option<Self> {
                        let params = Self::params();
                        let maybe_limbs: Option<[u128; $N]> = $crate::mavros::vendor::bignum::internal::__sqrt(params, self.get_limbs());
                        maybe_limbs.map(|limbs| Self {limbs: limbs})
                    }

                    unconstrained fn __sqrt(self: Self) -> crate::option::Option<Self> {
                        let params = Self::params();
                        let maybe_limbs: Option<[u128; $N]> = $crate::mavros::vendor::bignum::internal::__sqrt(params, self.get_limbs());
                        maybe_limbs.map(|limbs| Self {limbs: limbs })
                    }

                    fn assert_is_not_equal(self: Self, other: Self) {
                        let params = Self::params();
                        $crate::mavros::vendor::bignum::internal::assert_is_not_equal(
                            params,
                            self.get_limbs(),
                            other.get_limbs(),
                        );
                    }

                    fn validate_in_field(self: Self) {
                        let params = Self::params();
                        $crate::mavros::vendor::bignum::internal::validate_in_field::<$N, $MOD_BITS>(params, self.get_limbs());
                    }

                    fn validate_in_range(self: Self) {
                        $crate::mavros::vendor::bignum::internal::validate_in_range::<u128, $N, $MOD_BITS>(self.get_limbs());
                    }

                    fn sqr(self: Self) -> Self {
                        let params = Self::params();
                        Self { limbs: $crate::mavros::vendor::bignum::internal::sqr::<$N, $MOD_BITS>(params, self.get_limbs()) }
                    }

                    fn udiv_mod(self: Self, divisor: Self) -> (Self, Self) {
                        let (q, r) = $crate::mavros::vendor::bignum::internal::udiv_mod::<$N, $MOD_BITS>(self.get_limbs(), divisor.get_limbs());
                        (Self {limbs: q}, Self {limbs: r})
                    }

                    fn udiv(self: Self, divisor: Self) -> Self {
                        Self {limbs: $crate::mavros::vendor::bignum::internal::udiv::<$N, $MOD_BITS>(self.get_limbs(), divisor.get_limbs())}
                    }

                    fn umod(self: Self, divisor: Self) -> Self {
                        Self {limbs: $crate::mavros::vendor::bignum::internal::umod::<$N, $MOD_BITS>(self.get_limbs(), divisor.get_limbs())}
                    }

                    fn is_zero(self: Self) -> bool {
                        let params = Self::params();
                        $crate::mavros::vendor::bignum::internal::is_zero::<$N, $MOD_BITS>(params, self.get_limbs())
                    }

                    fn is_zero_integer(self: Self) -> bool {
                        $crate::mavros::vendor::bignum::internal::is_zero_integer::<u128,$N>(self.get_limbs())
                    }

                    fn assert_is_not_zero(self: Self) {
                        let params = Self::params();
                        $crate::mavros::vendor::bignum::internal::assert_is_not_zero::<$N, $MOD_BITS>(params, self.get_limbs());
                    }

                    fn assert_is_not_zero_integer(self: Self) {
                        $crate::mavros::vendor::bignum::internal::assert_is_not_zero_integer(self.get_limbs());
                    }
                }

                // implement Default for BigNum
                impl Default for $typ {
                    fn default() -> Self {
                        $typ::from_limbs([0; $N])
                    }
                }

                impl crate::convert::From<Field> for $typ {
                    fn from(input: Field) -> Self {
                        $typ { limbs: $crate::mavros::vendor::bignum::internal::from_field::<$N, $MOD_BITS>($params, input) }
                    }
                }

                impl crate::ops::Neg for $typ {
                    fn neg(self) -> Self {
                        $typ { limbs: $crate::mavros::vendor::bignum::internal::neg::<$N, $MOD_BITS>($params, self.limbs) }
                    }
                }

                impl crate::ops::Add for $typ {
                    fn add(self, other: Self) -> Self {
                        $typ { limbs: $crate::mavros::vendor::bignum::internal::add::<$N, $MOD_BITS>($params, self.limbs, other.limbs) }
                    }
                }

                impl crate::ops::Sub for $typ {
                    fn sub(self, other: Self) -> Self {
                        $typ { limbs: $crate::mavros::vendor::bignum::internal::sub::<$N, $MOD_BITS>($params, self.limbs, other.limbs) }
                    }
                }

                impl crate::ops::Mul for $typ {
                    fn mul(self, other: Self) -> Self {
                        $typ { limbs: $crate::mavros::vendor::bignum::internal::mul::<$N, $MOD_BITS>($params, self.limbs, other.limbs) }
                    }
                }

                impl crate::ops::Div for $typ {
                    fn div(self, other: Self) -> Self {
                        if $params.has_multiplicative_inverse {
                            $typ { limbs: $crate::mavros::vendor::bignum::internal::div::<$N, $MOD_BITS>($params, self.limbs, other.limbs) }
                        } else {
                            $typ { limbs: $crate::mavros::vendor::bignum::internal::udiv::<$N, $MOD_BITS>(self.limbs, other.limbs) }
                        }
                    }
                }

                impl crate::cmp::Eq for $typ {
                    fn eq(self, other: Self) -> bool {
                        $crate::mavros::vendor::bignum::internal::eq::<$N, $MOD_BITS>($params, self.limbs, other.limbs)
                    }
                }

                impl crate::cmp::Ord for $typ {
                    fn cmp(self, other: Self) -> crate::cmp::Ordering {
                        $crate::mavros::vendor::bignum::internal::cmp::<$N, $MOD_BITS>(self.limbs, other.limbs)
                    }
                }

            }
        }

        /// Conditionally selects between two BigNum values.
        ///
        /// Returns `lhs` if `predicate` is `true`, otherwise returns `rhs`.
        ///
        /// # Example
        /// ```
        /// let result = conditional_select(a, b, condition);
        /// // Equivalent to: if condition { a } else { b }
        /// ```
        ///
        /// # Note
        /// This is useful for constant-time selection where you want to avoid branching
        /// on secret values in constrained code.
        pub fn conditional_select<T: BigNum>(lhs: T, rhs: T, predicate: bool) -> T {
            if predicate {
                lhs
            } else {
                rhs
            }
        }

        /// Computes the quotient and remainder of a quadratic expression (unconstrained).
        ///
        /// Evaluates `sum_i(product(lhs_terms[i]) * product(rhs_terms[i])) + sum(linear_terms) = q * MOD + r`
        /// and returns `(q, r)`.
        ///
        /// # Parameters
        /// - `lhs_terms`: Left-hand side terms for each product. Shape: `[NUM_PRODUCTS][LHS_N]`
        /// - `lhs_flags`: If `true`, the corresponding lhs_term is negated
        /// - `rhs_terms`: Right-hand side terms for each product. Shape: `[NUM_PRODUCTS][RHS_N]`
        /// - `rhs_flags`: If `true`, the corresponding rhs_term is negated
        /// - `linear_terms`: Terms to add to the sum. Shape: `[ADD_N]`
        /// - `linear_flags`: If `true`, the corresponding linear_term is negated
        ///
        /// # Safety
        /// **UNCONSTRAINED**: This function generates no constraints. Use `evaluate_quadratic_expression`
        /// to constrain the result, or use the returned values as witnesses that you constrain separately.
        ///
        /// # Example
        /// Computing `a * b + c`:
        /// ```
        /// // Safety: unconstrained witness computation
        /// let (q, r) = unsafe {
        ///     compute_quadratic_expression(
        ///         [[a]],      // lhs_terms
        ///         [[false]],  // lhs_flags (not negated)
        ///         [[b]],      // rhs_terms
        ///         [[false]],  // rhs_flags
        ///         [c],        // linear_terms
        ///         [false],    // linear_flags
        ///     )
        /// };
        /// // r now contains (a * b + c) mod MOD
        /// ```
        pub unconstrained fn compute_quadratic_expression<T: BigNum, let LHS_N: u32, let RHS_N: u32, let NUM_PRODUCTS: u32, let ADD_N: u32>(
            lhs_terms: [[T; LHS_N]; NUM_PRODUCTS],
            lhs_flags: [[bool; LHS_N]; NUM_PRODUCTS],
            rhs_terms: [[T; RHS_N]; NUM_PRODUCTS],
            rhs_flags: [[bool; RHS_N]; NUM_PRODUCTS],
            linear_terms: [T; ADD_N],
            linear_flags: [bool; ADD_N],
        ) -> (T, T) {
            let params = T::params();
            let (q_limbs, r_limbs) = crate::mavros::vendor::bignum::fns::expressions::__compute_quadratic_expression(
                params,
                crate::mavros::vendor::bignum::utils::map::map(
                    lhs_terms,
                    |bns| crate::mavros::vendor::bignum::utils::map::map(bns, |bn: T| bn.get_limbs()),
                ),
                lhs_flags,
                crate::mavros::vendor::bignum::utils::map::map(
                    rhs_terms,
                    |bns| crate::mavros::vendor::bignum::utils::map::map(bns, |bn: T| bn.get_limbs()),
                ),
                rhs_flags,
                crate::mavros::vendor::bignum::utils::map::map(linear_terms, |bn: T| bn.get_limbs()),
                linear_flags,
            );
            (T::from_limbs(q_limbs), T::from_limbs(r_limbs))
        }

        /// Constrains a quadratic expression to equal zero modulo `MOD`.
        ///
        /// This is the **recommended** way to constrain BigNum arithmetic for optimal performance.
        /// Instead of using multiple constrained operations (`+`, `-`, `*`), compute results using
        /// unconstrained functions (`__add`, `__mul`, etc.) and then constrain them with a single
        /// call to this function.
        ///
        /// # Expression Format
        /// Constrains: `sum_i(product(lhs_terms[i]) * product(rhs_terms[i])) + sum(linear_terms) == 0 (mod MOD)`
        ///
        /// # Parameters
        /// - `lhs_terms`: Left-hand side terms for each product. Shape: `[NUM_PRODUCTS][LHS_N]`
        /// - `lhs_flags`: If `true`, the corresponding lhs_term is negated
        /// - `rhs_terms`: Right-hand side terms for each product. Shape: `[NUM_PRODUCTS][RHS_N]`
        /// - `rhs_flags`: If `true`, the corresponding rhs_term is negated
        /// - `linear_terms`: Terms to add to the sum. Shape: `[ADD_N]`
        /// - `linear_flags`: If `true`, the corresponding linear_term is negated
        ///
        /// # Example: Constraining `a * b + c - result == 0`
        /// ```
        /// // First compute result unconstrainedly
        /// // Safety: witness computation, constrained below
        /// let result = unsafe { a.__mul(b).__add(c) };
        ///
        /// // Then constrain: a * b + c - result == 0
        /// evaluate_quadratic_expression(
        ///     [[a]],           // lhs: [a]
        ///     [[false]],       // not negated
        ///     [[b]],           // rhs: [b]
        ///     [[false]],       // not negated
        ///     [c, result],     // linear: c + (-result)
        ///     [false, true],   // c positive, result negated
        /// );
        /// ```
        ///
        /// # Example: Constraining `(a + b) * c - d == 0`
        /// ```
        /// // Safety: witness computation
        /// let d = unsafe { (a.__add(b)).__mul(c) };
        ///
        /// // Constrain: (a + b) * c - d == 0
        /// evaluate_quadratic_expression(
        ///     [[a, b]],        // lhs: (a + b)
        ///     [[false, false]],
        ///     [[c]],           // rhs: c
        ///     [[false]],
        ///     [d],             // linear: -d
        ///     [true],          // d is negated
        /// );
        /// ```
        ///
        /// # Performance Note
        /// This function generates far fewer constraints than equivalent sequences of `+`, `-`, `*`
        /// operations. For complex expressions, the savings can be substantial.
        ///
        /// # Panics
        /// Fails if the expression does not equal zero modulo `MOD`.
        pub fn evaluate_quadratic_expression<T: BigNum, let LHS_N: u32, let RHS_N: u32, let NUM_PRODUCTS: u32, let ADD_N: u32>(
            lhs_terms: [[T; LHS_N]; NUM_PRODUCTS],
            lhs_flags: [[bool; LHS_N]; NUM_PRODUCTS],
            rhs_terms: [[T; RHS_N]; NUM_PRODUCTS],
            rhs_flags: [[bool; RHS_N]; NUM_PRODUCTS],
            linear_terms: [T; ADD_N],
            linear_flags: [bool; ADD_N],
        ) {
            let params = T::params();
            crate::mavros::vendor::bignum::fns::expressions::evaluate_quadratic_expression(
                params,
                crate::mavros::vendor::bignum::utils::map::map(
                    lhs_terms,
                    |bns| crate::mavros::vendor::bignum::utils::map::map(bns, |bn: T| bn.get_limbs()),
                ),
                lhs_flags,
                crate::mavros::vendor::bignum::utils::map::map(
                    rhs_terms,
                    |bns| crate::mavros::vendor::bignum::utils::map::map(bns, |bn: T| bn.get_limbs()),
                ),
                rhs_flags,
                crate::mavros::vendor::bignum::utils::map::map(linear_terms, |bn: T| bn.get_limbs()),
                linear_flags,
            )
        }

        /// Computes the modular inverses of multiple BigNum values efficiently (unconstrained).
        ///
        /// Uses Montgomery's batch inversion trick to compute all inverses with only a single
        /// modular inversion, making this much more efficient than calling `__invmod` repeatedly.
        ///
        /// # Requirements
        /// - `MOD` must be prime (`params.has_multiplicative_inverse == true`)
        /// - Zero elements are allowed and will map to zero in the output
        ///
        /// # Performance
        /// Computes M inverses using:
        /// - 3(M-1) multiplications
        /// - 1 inversion
        ///
        /// This is much cheaper than M separate inversions for M > 1.
        ///
        /// # Safety
        /// **UNCONSTRAINED**: No constraints are generated. Use the results as witnesses and
        /// constrain them appropriately (e.g., verify `x[i] * result[i] == 1` using
        /// `evaluate_quadratic_expression`).
        ///
        /// # Example
        /// ```
        /// // Safety: unconstrained batch inversion
        /// let inverses = unsafe { batch_invert([a, b, c]) };
        /// // inverses[0] == a^{-1}, inverses[1] == b^{-1}, inverses[2] == c^{-1}
        /// ```
        pub unconstrained fn batch_invert<T: BigNum, let M: u32>(x: [T; M]) -> [T; M] {
            let params = T::params();
            assert(params.has_multiplicative_inverse);
            crate::mavros::vendor::bignum::fns::unconstrained_ops::batch_invert(params, x.map(|bn: T| bn.get_limbs())).map(|limbs| {
                T::from_limbs(limbs)
            })
        }

        /// Computes the modular inverses of a slice of BigNum values efficiently (unconstrained).
        ///
        /// Same as `batch_invert` but accepts a dynamically-sized slice instead of a fixed-size array.
        ///
        /// # Requirements
        /// - `MOD` must be prime (`params.has_multiplicative_inverse == true`)
        /// - Zero elements are allowed and will map to zero in the output
        ///
        /// # Safety
        /// **UNCONSTRAINED**: No constraints are generated.
        ///
        /// See `batch_invert` for more details.
        pub unconstrained fn batch_invert_slice<T: BigNum>(x: [T]) -> [T] {
            let params = T::params();
            assert(params.has_multiplicative_inverse);
            crate::mavros::vendor::bignum::fns::unconstrained_ops::batch_invert_slice(params, x.map(|bn: T| bn.get_limbs()))
                .map(|limbs| T::from_limbs(limbs))
        }

        /// Converts a BigNum to a native circuit Field element.
        ///
        /// # Requirements
        /// - The BigNum value must be less than the circuit field modulus (Grumpkin: ~2^254)
        /// - For N > 2, the value is validated to be less than the Grumpkin modulus
        ///
        /// # Panics
        /// - Fails range validation if the value is >= circuit field modulus
        /// - Fails if limbs are not properly range-constrained
        ///
        /// # Note
        /// This is a constrained operation that validates the BigNum is representable as a Field.
        /// For BigNums with `MOD_BITS > 253`, ensure your value is actually less than the circuit
        /// field modulus before calling this.
        pub fn to_field<T: BigNum>(bn: T) -> Field {
            let params = T::params();
            limbs_to_field(params, bn.get_limbs())
        }
    }

    pub mod fields {

        use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
        pub mod secp256r1Fq {
            use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
            use crate::mavros::vendor::bignum::bignum::BigNum;
            use crate::mavros::vendor::bignum::bignum::derive_bignum;
            use crate::mavros::vendor::bignum::params::BigNumParams;

            pub global Secp256r1_Fq_PARAMS: BigNumParams<3, 256> = BigNumParams {
                has_multiplicative_inverse: true,
                modulus: [0xffffffffffffffffffffffff, 0xffff00000001000000000000000000, 0xffff],
                double_modulus: [
                    0x01000001fffffffffffffffffffffffe,
                    0x01fffe00000001ffffffffffffffffff,
                    0x01fffe,
                ],
                redc_param: [0xffffbfffffffc000000000000000c0, 0x3fffffffffffffffbfffffffbfff, 0x400000],
            };

            #[derive_bignum(3, 256, quote { Secp256r1_Fq_PARAMS })]
            pub struct Secp256r1_Fq {
                limbs: [u128; 3],
            }

            pub fn secp256r1_fq_from_limbs(limbs: [u128; 3]) -> Secp256r1_Fq {
                Secp256r1_Fq { limbs }
            }
        }

        pub mod secp256r1Fr {

            use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
            use crate::mavros::vendor::bignum::bignum::BigNum;
            use crate::mavros::vendor::bignum::bignum::derive_bignum;
            use crate::mavros::vendor::bignum::params::BigNumParams;

            pub global Secp256r1_Fr_PARAMS: BigNumParams<3, 256> = BigNumParams {
                has_multiplicative_inverse: true,
                modulus: [0xe6faada7179e84f3b9cac2fc632551, 0xffff00000000ffffffffffffffffbc, 0xffff],
                double_modulus: [
                    0x01cdf55b4e2f3d09e7739585f8c64aa2,
                    0x01fffe00000001ffffffffffffffff78,
                    0x01fffe,
                ],
                redc_param: [0x4154b7c69b08404bff617bb7e6ff99, 0x3fffffffffffffffbfffffffd0c6, 0x400000],
            };

            #[derive_bignum(3, 256, quote { Secp256r1_Fr_PARAMS })]
            pub struct Secp256r1_Fr {
                limbs: [u128; 3],
            }

            pub fn secp256r1_fr_from_limbs(limbs: [u128; 3]) -> Secp256r1_Fr {
                Secp256r1_Fr { limbs }
            }
        }

    }

    pub mod internal {

        use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
        pub use super::fns::constrained_ops::{
            add, assert_is_not_equal, assert_is_not_zero, assert_is_not_zero_integer, cmp, derive_from_seed,
            div, eq, from_field, is_zero, is_zero_integer, mul, neg, sqr, sub, udiv, udiv_mod,
            umod, validate_in_field, validate_in_range,
        };
        pub use super::fns::unconstrained_ops::{
            __add, __derive_from_seed, __div, __eq, __invmod, __is_zero, __mul, __neg, __pow,
            __sqr, __sqrt, __sub, __udiv_mod,
        };
        pub use super::fns::serialization::{from_be_bytes, from_le_bytes, to_be_bytes, to_le_bytes};
    }

    pub use bignum::{BigNum, derive_bignum};
    pub use fields::secp256r1Fq::Secp256r1_Fq;
    pub use fields::secp256r1Fr::Secp256r1_Fr;
}

pub mod bigcurve {

    use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
    pub mod constants {
        use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
        pub(crate) global TWO_POW_120: u128 = 0x1000000000000000000000000000000;
    }

    pub mod scalar_field {

        use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
        use crate::mavros::vendor::bigcurve::constants::TWO_POW_120;
        use crate::mavros::vendor::bignum::BigNum;

        /// Represents a scalar multiplier as a sequence of 4-bit slices.
        ///
        /// # Overview
        /// `ScalarField` efficiently encodes scalars for twisted Edwards curve operations. Since these curves
        /// typically have prime group orders that fit within a field element, scalars can often be reconstructed
        /// from their 4-bit slices by summing the slices and verifying equality with the original scalar.
        ///
        /// For arbitrary field elements (scalars that may be multiples of the curve's group order), additional
        /// validation is required. In these cases, the sum of the 4-bit slices, interpreted as an integer,
        /// must be less than the field modulus. This check is computationally expensive and avoided when possible.
        ///
        /// The `ScalarField<N>` type is flexible:
        /// - `ScalarField<63>`: Efficient bitslice conversions for scalars less than 2^252.
        /// - `ScalarField<64>`: Supports bitslice conversions for arbitrary field elements.
        ///
        /// **Note:** The bit values in `ScalarField` are not constrained to be less than the curve's group order.
        /// This type is intended for use in scalar multiplication, where all operations are performed modulo the
        /// curve order.
        pub struct ScalarField<let N: u32> {
            pub(crate) base4_slices: [u8; N],
            pub(crate) skew: bool,
        }

        /// Decompose a native `Field` element into a fixed-length signed radix-16 (window-4) form (unconstrained).
        ///
        /// Let `x` be interpreted as a non-negative integer and let `N` be the number of 4-bit windows (nibbles)
        /// extracted from `x` in little-endian order, truncated/padded to length `N`.
        ///
        /// This function outputs `slices[i] \in [0, 15]` and `skew \in {0, 1}` such that the odd signed digits
        ///
        ///     a_i = 2 * slices[i] - 15  \in { -15, -13, ..., 13, 15 } (odd)
        ///
        /// satisfy the reconstruction identity:
        ///
        ///     x = ( \sum_{i=0}^{N-1} a_i * 16^{N-1-i} ) - skew
        ///
        /// Equivalently,
        ///
        ///     x + skew = sum_{i=0}^{N-1} (2 * slices[i] - 15) * 16^{N-1-i}.
        ///
        /// The `skew` bit is chosen so that `x + skew` is odd, which makes an all-odd-digit decomposition possible.
        /// For even input nibbles, the routine performs a local carry/borrow adjustment
        ///
        ///     slices[j] += 1,  slices[j+1] -= 8
        ///
        /// which corresponds to
        ///
        ///     a_j += 2,  a_{j+1} -= 16,
        ///
        /// preserving the overall value while keeping digits odd and bounded.
        ///
        /// ## Note
        /// `N` is a protocol parameter (fixed length), not derived from `x`. Higher windows are effectively
        /// padding; uniqueness is not a goal here.
        ///
        /// Returns `ScalarField { base4_slices: slices, skew }`
        unconstrained fn __get_wnaf_slices_from_field<let N: u32>(x: Field) -> ScalarField<N> {
            let mut base4_slices: [u8; N] = [0; N];
            let bytes: [u8; 32] = x.to_le_bytes::<32>();
            let mut nibbles: [u8; N] = __extract_nibbles_from_bytes(bytes);

            let skew: bool = (nibbles[0] & 1) == 0;

            nibbles[0] = nibbles[0] + (skew as u8);
            base4_slices[N - 1] = (nibbles[0] + 15) / 2;

            for i in 1..N {
                let nibble: u8 = nibbles[i];
                base4_slices[N - 1 - i] = (nibble + 15) / 2;
                if (nibble & 1) == 0 {
                    base4_slices[N - 1 - i] += 1;
                    base4_slices[N - i] -= 8;
                }
            }

            ScalarField { base4_slices, skew }
        }

        /// Decompose a `BigNum` element into a fixed-length signed radix-16 (window-4) form (unconstrained).
        ///
        /// see `__get_wnaf_slices_from_field` for details
        unconstrained fn __get_wnaf_slices_from_bignum<let N: u32, BN>(x: BN) -> ScalarField<N>
        where
            BN: BigNum,
        {
            assert(N * 4 >= <BN as BigNum>::MOD_BITS, "Not enough nibbles to decompose a BigNum value");

            let mut base4_slices: [u8; N] = [0; N];

            let bytes: [u8; _] = x.to_le_bytes();
            let mut nibbles: [u8; N] = __extract_nibbles_from_bytes(bytes);

            let skew: bool = (nibbles[0] & 1) == 0;

            nibbles[0] = nibbles[0] + (skew as u8);
            base4_slices[N - 1] = (nibbles[0] + 15) / 2;

            for i in 1..N {
                let nibble: u8 = nibbles[i];
                base4_slices[N - 1 - i] = (nibble + 15) / 2;
                if (nibble & 1) == 0 {
                    base4_slices[N - 1 - i] += 1;
                    base4_slices[N - i] -= 8;
                }
            }

            ScalarField { base4_slices, skew }
        }

        /// Get the circuit native modulus (grumpkin) in wNAF form
        pub(crate) fn get_grumpkin_modulus_slices<let N: u32>() -> [u8; N] {
            let mut expected_slices: [u8; N] = [0; N];

            if N == 64 {
                let slice: [u8; 64] = [
                    9, 8, 3, 2, 2, 7, 3, 9, 7, 0, 9, 8, 13, 0, 1, 4, 13, 12, 2, 8, 2, 2, 13, 11, 4, 0, 12,
                    0, 10, 12, 2, 14, 9, 4, 1, 9, 15, 4, 2, 4, 3, 12, 13, 12, 11, 8, 4, 8, 10, 1, 15, 0, 15,
                    10, 12, 9, 15, 8, 0, 0, 0, 0, 0, 0,
                ];
                for i in 0..N {
                    expected_slices[i] = slice[i];
                }
            } else if N == 65 {
                let slice: [u8; 65] = [
                    8, 1, 8, 3, 2, 2, 7, 3, 9, 7, 0, 9, 8, 13, 0, 1, 4, 13, 12, 2, 8, 2, 2, 13, 11, 4, 0,
                    12, 0, 10, 12, 2, 14, 9, 4, 1, 9, 15, 4, 2, 4, 3, 12, 13, 12, 11, 8, 4, 8, 10, 1, 15, 0,
                    15, 10, 12, 9, 15, 8, 0, 0, 0, 0, 0, 0,
                ];
                for i in 0..N {
                    expected_slices[i] = slice[i];
                }
            } else if N > 65 {
                // For N > 65, we need to insert zeros at the beginning
                let num_zeros: u32 = N - 65;
                expected_slices[0] = 8;
                for i in 1..num_zeros + 1 {
                    expected_slices[i] = 0;
                }
                let slice: [u8; 65] = [
                    8, 1, 8, 3, 2, 2, 7, 3, 9, 7, 0, 9, 8, 13, 0, 1, 4, 13, 12, 2, 8, 2, 2, 13, 11, 4, 0,
                    12, 0, 10, 12, 2, 14, 9, 4, 1, 9, 15, 4, 2, 4, 3, 12, 13, 12, 11, 8, 4, 8, 10, 1, 15, 0,
                    15, 10, 12, 9, 15, 8, 0, 0, 0, 0, 0, 0,
                ];
                for i in num_zeros + 1..N {
                    expected_slices[i] = slice[i - num_zeros];
                }
            }
            expected_slices
        }

        pub fn compare_scalar_field_to_grumpkin<let N: u32>(result: ScalarField<N>) {
            let grumpkin_modulus_slices: [u8; N] = get_grumpkin_modulus_slices::<N>();

            // Lexicographic comparison: stop when we find a strictly smaller number
            let mut should_continue: bool = true;
            for i in 0..N {
                if should_continue {
                    if result.base4_slices[i] < grumpkin_modulus_slices[i] {
                        // Found a strictly smaller number, we can stop - this is valid
                        should_continue = false;
                    } else if result.base4_slices[i] > grumpkin_modulus_slices[i] {
                        // Found a strictly larger number, this is invalid
                        panic(f"Reconstructed number is greater than modulus");
                    }
                    // If equal, continue to the next element (should_continue remains true)
                }
            }
        }

        /// Convert a `Field` element into `ScalarField`
        ///
        /// Decomposes the `Field` value into 4-bit slices.
        /// then we verify that the resulting value is less than GRUMPKIN_MODULUS
        impl<let N: u32> crate::convert::From<Field> for ScalarField<N> {
            /// Constructs an instance from a field element.
            fn from(input: Field) -> Self {
                // Safety: We constrain the ScalarField later via
                //  - field reconstruction equality
                //  - slice-wise integer comparison (comp)
                let result: ScalarField<N> = unsafe { __get_wnaf_slices_from_field(input) };

                if !crate::runtime::is_unconstrained() {
                    // Enforce that limbs are all 4 bits.
                    for i in 0..N {
                        (result.base4_slices[i] as Field).assert_max_bit_size::<4>();
                    }

                    // Enforce consistency with `input`.
                    let reconstructed_input: Field = result.into();
                    assert_eq(reconstructed_input, input);
                    // .into already forces the result to be < GRUMPKIN_MODULUS
                }
                result
            }
        }

        impl<let N: u32> crate::convert::Into<Field> for ScalarField<N> {
            /// Construct from tuple of field elements
            ///
            /// Use this method instead of `new` if you know x/y is on the curve
            fn into(self: Self) -> Field {
                // TODO: This is susceptible to overflow when N is large!
                if !crate::runtime::is_unconstrained() {
                    if N >= 64 {
                        compare_scalar_field_to_grumpkin(self);
                    }
                }
                let mut acc: Field = 0;
                for i in 0..N {
                    acc = acc * 16;
                    acc = acc + (self.base4_slices[i] as Field) * 2 - 15;
                }
                acc -= self.skew as Field;
                acc
            }
        }

        impl<let N: u32> ScalarField<N> {
            pub fn len(_: Self) -> u32 {
                N
            }

            pub fn zero() -> Self {
                let mut result: Self = ScalarField { base4_slices: [0; N], skew: true };
                result.base4_slices[0] = 8; // 8 = numeric value of 1, 0 = numeric value of -15
                result
            }

            pub fn from_bignum<BN>(x: BN) -> Self
            where
                BN: BigNum,
            {
                x.validate_in_field();
                // Safety: separate verification
                let slices_result: ScalarField<N> = unsafe { __get_wnaf_slices_from_bignum(x) };

                let slices: [u8; N] = slices_result.base4_slices;
                let skew: bool = slices_result.skew;

                for i in 0..N {
                    (slices[i] as Field).assert_max_bit_size::<4>();
                }

                let mut result: BN = BN::zero();
                let mut count: u32 = 0;
                {
                    let mut acc: u128 = 0;
                    let mut last_bits: u64 = (result.modulus_bits() % 120) as u64;
                    if (last_bits == 0) {
                        last_bits = 120;
                    }
                    let last_nibbles: u64 = (last_bits / 4 + 1) as u64;
                    for _ in 0..last_nibbles {
                        acc = acc * 16;
                        acc = acc + (slices[count] as u128) * 2 - 15;
                        count = count + 1;
                    }
                    result.set_limb(result.num_limbs() - 1, acc);
                }
                for i in 1..result.num_limbs() {
                    // Using a Field type to allow for negative numbers.
                    let mut acc: Field = 0;
                    for _ in 0..30 {
                        acc = acc * 16;
                        acc = acc + (slices[count] as Field) * 2 - 15;
                        count = count + 1;
                    }
                    // Handle overflow using helper function
                    let acc_u128: u128 = handle_overflow::<BN>(acc, &mut result, i);
                    result.set_limb(result.num_limbs() - 1 - i, acc_u128);
                }
                result.set_limb(0, result.get_limb(0) - skew as u128);
                assert(result == x);
                Self { base4_slices: slices, skew }
            }

            pub fn into_bignum<BN>(self) -> BN
            where
                BN: BigNum,
            {
                let mut result: BN = BN::zero();
                let mut count: u32 = 0;
                {
                    let mut acc: u128 = 0;
                    let mut last_bits: u64 = (result.modulus_bits() % 120) as u64;
                    if (last_bits == 0) {
                        last_bits = 120;
                    }
                    let last_nibbles: u64 = (last_bits / 4 + 1) as u64;
                    for _ in 0..last_nibbles {
                        acc = acc * 16;
                        acc = acc + (self.base4_slices[count] as u128) * 2 - 15;
                        count = count + 1;
                    }
                    result.set_limb(result.num_limbs() - 1, acc);
                }
                for i in 1..result.num_limbs() {
                    // Using a Field type to allow for negative numbers.
                    let mut acc: Field = 0;
                    for _ in 0..30 {
                        acc = acc * 16;
                        acc = acc + (self.base4_slices[count] as Field) * 2 - 15;
                        count = count + 1;
                    }
                    // Handle overflow using helper function
                    let acc_u128: u128 = handle_overflow::<BN>(acc, &mut result, i);
                    result.set_limb(result.num_limbs() - 1 - i, acc_u128);
                }
                result.set_limb(0, result.get_limb(0) - (self.skew as u128));
                result
            }

            pub fn new() -> Self {
                Self { base4_slices: [0; N], skew: false }
            }

            pub fn get(self, idx: u32) -> u8 {
                self.base4_slices[idx]
            }
        }

        // ------------------------------ HELPER FUNCTIONS ------------------------------

        /// Split an array of bytes into an array of nibbles (half-bytes) (unconstrained).
        ///
        /// Given an array of `M` bytes, extracts up to `N` nibbles in little-endian
        /// nibble order: for each byte, the low nibble (bits 0-3) is emitted first,
        /// followed by the high nibble (bits 4-7)
        ///
        /// If fewer than `N` nibbles are available, the remaining entries in the
        /// returned array are filled with zeros.
        pub(crate) unconstrained fn __extract_nibbles_from_bytes<let N: u32, let M: u32>(
            bytes: [u8; M],
        ) -> [u8; N] {
            let mut result: [u8; N] = [0; N];
            let mut nibble_idx: u32 = 0;

            for i in 0..M {
                if nibble_idx >= N {
                    break;
                }
                // Extract low nibble (bits 0-3)
                result[nibble_idx] = bytes[i] & 0x0F;
                nibble_idx += 1;

                if nibble_idx >= N {
                    break;
                }
                // Extract high nibble (bits 4-7)
                result[nibble_idx] = (bytes[i] >> 4);
                nibble_idx += 1;
            }

            result
        }

        // Helper function to handle overflow in wNAF reconstruction
        fn handle_overflow<B: BigNum>(acc: Field, result: &mut B, limb_index: u32) -> u128 {
            let mut acc_u128: u128 = acc as u128;

            // Handle negative numbers. If the number is negative, it is greater than 2^120 in Field.
            // All positive numbers are less than 2^120 in Field.
            // For example, 42 is -6 + 3 * 2^4 with wnaf algorithm, but it should be 42 = 10 + 2 * 2^4
            // To get the actual number 10, we need to subtract the number 6 from 2^4.
            if acc_u128 >= TWO_POW_120 {
                // find the actual number needed to be subtracted.
                let reverse: u128 = (acc * -1) as u128;
                acc_u128 = (1 as u128 << 120) - reverse;
                result.set_limb(
                    result.num_limbs() - limb_index,
                    result.get_limb(result.num_limbs() - limb_index) - 1,
                );
            }
            acc_u128
        }
    }

    pub mod bigcurve {

        use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
        use crate::mavros::vendor::bigcurve::curve_jac::types::AffineTranscript;
        use crate::mavros::vendor::bigcurve::scalar_field::ScalarField;
        use crate::mavros::vendor::bignum::BigNum;

        use crate::ops::{Add, Neg, Sub};

        /// Compile-time parameters for a Short Weierstrass curve `y^2 = x^3 + a*x + b`.
        ///
        /// Stores the curve coefficients, a canonical generator, and the two offset generators
        /// used to avoid edge cases in incomplete addition formulas.
        pub struct BigCurveParams<BN> {
            pub a: BN,
            pub b: BN,
            pub offset_generator: [BN; 2],
            pub offset_generator_final: [BN; 2],
            pub one: [BN; 2],
        }

        /// The core trait for elliptic curve arithmetic over a non-native field.
        ///
        /// Represents a point on a Short Weierstrass curve `y^2 = x^3 + a*x + b` where the
        /// base field is modeled by `BN: BigNum` (not the circuit's native field).
        ///
        /// # Two-phase architecture
        ///
        /// Expensive operations (`mul`, `Add`, `Sub`, `evaluate_linear_expression`) follow a
        /// two-phase pattern:
        ///   1. **Unconstrained**: compute the result in Jacobian coordinates (fast, no constraints)
        ///   2. **Constrained**: verify the result in affine coordinates using `evaluate_quadratic_expression`
        ///
        /// Offset generators are added before computation and subtracted after, ensuring the
        /// incomplete addition formulas never encounter edge cases (equal or inverse points).
        ///
        /// # Implementing a new curve
        ///
        /// Use the `#[derive_curve_impl]` comptime attribute on a struct with fields
        /// `x: BN`, `y: BN`, `is_infinity: bool` to auto-generate all trait methods
        /// and operator impls (`Add`, `Sub`, `Neg`, `Eq`).
        pub trait BigCurve<BN: BigNum>: crate::cmp::Eq + Add + Sub + Neg {
            /// Returns the x-coordinate of the point.
            fn x(self) -> BN;

            /// Returns the y-coordinate of the point.
            fn y(self) -> BN;

            /// Returns the `a` coefficient of the curve equation `y^2 = x^3 + a*x + b`.
            fn a() -> BN;

            /// Returns the `b` coefficient of the curve equation `y^2 = x^3 + a*x + b`.
            fn b() -> BN;

            /// Construct a point from affine coordinates and an infinity flag.
            fn from_coordinates(x: BN, y: BN, is_infinity: bool) -> Self;

            /// Returns the point at infinity (the identity element of the curve group).
            fn point_at_infinity() -> Self;

            /// Returns true if the point is the point at infinity.
            fn is_infinity(self) -> bool;

            /// Returns the canonical generator point of the curve.
            fn one() -> Self;

            /// Returns the initial offset generator, added before scalar multiplication.
            ///
            /// Derived deterministically from `hash_to_curve("bigcurve offset generator")`.
            fn offset_generator() -> Self;

            /// Returns the final offset generator, subtracted after scalar multiplication.
            ///
            /// Equals `2^{4*(NScalarSlices-1)} * offset_generator`, matching the total
            /// number of doublings in the wNAF loop.
            fn offset_generator_final() -> Self;

            /// Validate that the point lies on the curve (constrained).
            ///
            /// Checks `y^2 == x^3 + a*x + b` via `evaluate_quadratic_expression`.
            /// Points at infinity are accepted unconditionally.
            fn validate_on_curve(self);

            /// Scalar multiplication: compute `scalar * self` (constrained).
            ///
            /// Uses wNAF with a 4-bit window. The scalar must be encoded as a `ScalarField`.
            fn mul<let NScalarSlices: u32>(self, scalar: ScalarField<NScalarSlices>) -> Self;

            /// Hash an arbitrary byte seed to a point on the curve (constrained).
            ///
            /// Uses try-and-increment with Poseidon2 hashing.
            fn hash_to_curve<let N: u32>(seed: [u8; N]) -> Self;

            /// Compute `sum_i(mul_scalars[i] * mul_points[i]) + sum_j(add_points[j])` (constrained).
            ///
            /// The most general operation: combines MSM with plain point additions.
            fn evaluate_linear_expression<let NScalarSlices: u32, let NMuls: u32, let NAdds: u32>(
                mul_points: [Self; NMuls],
                mul_scalars: [ScalarField<NScalarSlices>; NMuls],
                add_points: [Self; NAdds],
            ) -> Self;
        }

        /// Comptime macro that generates `BigCurve` trait impl and operator overloads for a curve struct.
        ///
        /// Applied via `#[derive_curve_impl(FieldType, PARAMS)]` on a struct with fields
        /// `x: FieldType`, `y: FieldType`, `is_infinity: bool`.
        ///
        /// Generates: `BigCurve` impl, `Add`, `Sub`, `Neg`, `Eq`.
        pub comptime fn derive_curve_impl(
            struct_def: TypeDefinition,
            field_type: Quoted,
            params: Quoted,
        ) -> Quoted {
            let typ = struct_def.as_type();
            let CurveJ = quote { $crate::mavros::vendor::bigcurve::curve_jac::types::CurveJ };
            let AffineTranscript = quote { $crate::mavros::vendor::bigcurve::curve_jac::types::AffineTranscript };

            quote {
                impl $crate::mavros::vendor::bigcurve::BigCurve<$field_type> for $typ {
                    fn x(self) -> $field_type {
                        self.x
                    }

                    fn y(self) -> $field_type {
                        self.y
                    }

                    fn a() -> $field_type {
                        $params.a
                    }

                    fn b() -> $field_type {
                        $params.b
                    }   

                    fn is_infinity(self) -> bool {
                        self.is_infinity
                    }

                    fn from_coordinates(x: $field_type, y: $field_type, is_infinity: bool) -> Self {
                        Self { x, y, is_infinity }
                    }

                    fn point_at_infinity() -> Self {
                        Self { 
                            x: $field_type::zero(), 
                            y: $field_type::zero(), 
                            is_infinity: true 
                        }
                    }

                    fn offset_generator() -> Self {
                        let result: [$field_type; 2] = $params.offset_generator;
                        Self { 
                            x: result[0], 
                            y: result[1], 
                            is_infinity: false 
                        }
                    }

                    fn offset_generator_final() -> Self {
                        let result: [$field_type; 2] = $params.offset_generator_final;
                        Self { 
                            x: result[0], 
                            y: result[1], 
                            is_infinity: false 
                        }
                    }

                    fn one() -> Self {
                        let result: [$field_type; 2] = $params.one;
                        Self { 
                            x: result[0], 
                            y: result[1], 
                            is_infinity: false 
                        }
                    }

                    fn validate_on_curve(self) {
                        // Point at infinity is trivially on the curve; substitute a valid point for the check
                        let one: Self = Self::one();
                        let x: $field_type = if self.is_infinity { one.x } else { self.x };
                        let y: $field_type = if self.is_infinity { one.y } else { self.y };
                        let xx = crate::ops::Mul::mul(x, x);
                        // (xx + a) * x - y*y + b = 0  i.e.  x^3 + a*x + b - y^2 = 0
                        $crate::mavros::vendor::bigcurve::internal::evaluate_quadratic_expression(
                            [[xx, $params.a], [y, $field_type::zero()]],
                            [[false, false], [false, false]],
                            [[x], [y]],
                            [[false], [true]],
                            [$params.b],
                            [false],
                        );
                    }

                    fn mul<let NScalarSlices: u32>(self, scalar: ScalarField<NScalarSlices>) -> Self {
                        // Safety: transcript is verified step-by-step in mul_with_hint
                        let transcript: [$AffineTranscript<$field_type>; (NScalarSlices * 5) + 6] =
                            unsafe { $crate::mavros::vendor::bigcurve::internal::get_mul_transcript::<NScalarSlices, $field_type, $typ>(self, scalar) };
                        $crate::mavros::vendor::bigcurve::internal::mul_with_hint::<NScalarSlices, (NScalarSlices * 5) + 6, $field_type, $typ>(self, scalar, transcript)
                    }

                    fn hash_to_curve<let N: u32>(seed: [u8; N]) -> Self {
                        // Not part of the ECDSA verifier compatibility subset.
                        let _ = seed;
                        Self::one()
        }

                    fn evaluate_linear_expression<let NScalarSlices: u32, let NMuls: u32, let NAdds: u32>(
                        mul_points: [Self; NMuls],
                        mul_scalars: [ScalarField<NScalarSlices>; NMuls],
                        add_points: [Self; NAdds],
                    ) -> Self {
                        $crate::mavros::vendor::bigcurve::internal::evaluate_linear_expression::<$field_type,$typ, _, _, _>(mul_points, mul_scalars, add_points)
                    }}

                    // Complete addition with full edge-case handling (generates witness via Jacobian intermediates)
                    impl crate::ops::Add for $typ {
                        fn add(self, other: Self) -> Self {
                            let lhsJ = $CurveJ::<$field_type, $typ>::from(self);
                            let rhsJ = $CurveJ::<$field_type, $typ>::from(other);
                            // Safety: Jacobian result verified via add_with_hint constraints
                            let (result_jac, j_transcript) = unsafe { lhsJ.add(rhsJ) };
                            // Safety: affine conversion verified inside add_with_hint
                            let transcript = unsafe { $crate::mavros::vendor::bigcurve::curve_jac::types::AffineTranscript::from_j(j_transcript) };
                            if crate::runtime::is_unconstrained() {
                                $typ::from_coordinates(transcript.x3, transcript.y3, result_jac.is_infinity)
                            } else {
                                $crate::mavros::vendor::bigcurve::internal::add_with_hint::<$field_type, $typ>(self, other, transcript)           
                            }
                        }
                    }

                    impl crate::cmp::Eq for $typ {
                        fn eq(self, other: Self) -> bool {
                            let coords_equal =
                                self.x.eq(other.x) & self.y.eq(other.y) & !self.is_infinity & !other.is_infinity;
                            let infinity = self.is_infinity & other.is_infinity;
                            coords_equal | infinity
                        }
                    }

                    impl crate::ops::Sub for $typ {
                        fn sub(self, other: Self) -> Self {
                            let lhsJ = $CurveJ::<$field_type, $typ>::from(self);
                            let rhsJ = $CurveJ::<$field_type, $typ>::from(other);
                            // Safety: Jacobian result verified via sub_with_hint constraints
                            let (result_jac, j_transcript) = unsafe { lhsJ.sub(rhsJ) };
                            // Safety: affine conversion verified inside sub_with_hint
                            let transcript = unsafe { $crate::mavros::vendor::bigcurve::curve_jac::types::AffineTranscript::from_j(j_transcript) };
                            if crate::runtime::is_unconstrained() {
                                $typ::from_coordinates(transcript.x3, transcript.y3, result_jac.is_infinity)
                            } else {
                                $crate::mavros::vendor::bigcurve::internal::sub_with_hint::<$field_type, $typ>(self, other, transcript)
                            }
                        }
                    }

                    impl crate::ops::Neg for $typ {
                        fn neg(self) -> Self {
                            Self { 
                                x: self.x, 
                                y: crate::ops::Neg::neg(self.y), 
                                is_infinity: self.is_infinity 
                            }
                        }
                    }        
            }
        }
    }

    pub mod curve_jac {

        use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
        pub mod types {
            use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
            use crate::mavros::vendor::bignum::BigNum;

            /// CurveJ represents a Short Weierstrass elliptic curve using Jacobian coordinates.
            ///
            /// # Representation
            /// - **Jacobian form:** (X, Y, Z)
            /// - **Affine form:** (x, y)
            /// - **Relation:** x = X / Z^2, y = Y / Z^3
            ///
            /// # Purpose
            /// This class is designed to speed up witness generation when optimally constraining elliptic curve operations.
            ///
            /// # Motivation
            /// - Constraints are best constructed in affine coordinates due to fewer field multiplications (3 vs ~14 for Jacobian).
            /// - However, affine coordinates require one modular inverse per group operation, which is expensive over a non-native field (~2D multiplications, where D is the field bit-width).
            ///
            /// # Solution
            /// - Perform all computations in unconstrained functions using Jacobian coordinates (no inverses required).
            /// - Record each ECC operation in a `JTranscript` object.
            /// - Use a Montgomery Batch Inverse to compute all required inverses at once when evaluating with affine arithmetic.
            /// - This reduces the number of modular inverses from ~256 or 320 (depending on the curve) to just one.
            ///
            /// # Note
            /// This approach is complex but necessary due to the high cost of generating modular inverses for witness generation.
            pub struct CurveJ<BN, BigCurve> {
                pub(crate) x: BN,
                pub(crate) y: BN,
                pub(crate) z: BN,
                pub is_infinity: bool,
            }

            /// A transcript of a group operation in Jacobian coordinates.
            ///
            /// # Fields
            /// - `x3`, `y3`, `z3`: The output coordinates of the group operation in Jacobian form.
            /// - `lambda_numerator`: The numerator of the `lambda` term used in the group operation.
            ///   (The denominator is assumed to be `z3`.)
            pub struct JTranscript<BN> {
                pub(crate) lambda_numerator: BN,
                pub(crate) x3: BN,
                pub(crate) y3: BN,
                pub(crate) z3: BN,
            }

            impl<BN> JTranscript<BN>
            where
                BN: BigNum,
            {
                pub unconstrained fn new() -> Self {
                    JTranscript { lambda_numerator: BN::zero(), x3: BN::zero(), y3: BN::zero(), z3: BN::zero() }
                }
            }

            /// A transcript of a group operation in Affine coordinates.
            ///
            /// # Fields
            /// - `x3`, `y3`: The output coordinates of the group operation.
            /// - `lambda`: The gradient of the line that passes through the input points of the group operation.
            ///
            /// # Details
            /// - For addition: `lambda = (y2 - y1) / (x2 - x1)`
            /// - For doubling: `lambda = (3 * x1 * x1) / (2 * y1)`
            ///
            /// # Optimization
            /// If you have an array of `JTranscript` objects, you can convert them into `AffineTranscript` objects using only one modular inverse.
            pub struct AffineTranscript<BN> {
                pub lambda: BN,
                pub x3: BN,
                pub y3: BN,
            }

            impl<BN> AffineTranscript<BN>
            where
                BN: BigNum,
            {
                pub(crate) fn new() -> Self {
                    AffineTranscript { lambda: BN::zero(), x3: BN::zero(), y3: BN::zero() }
                }

                /// Convert a single `JTranscript` to `AffineTranscript` (unconstrained)
                ///
                /// Convenience wrapper around `from_jacobian_transcript` for a single entry.
                pub unconstrained fn from_j(j_tx: JTranscript<BN>) -> Self {
                    AffineTranscript::from_jacobian_transcript([j_tx])[0]
                }

                /// Convert a `JTranscript` to `AffineTranscript` given a precomputed `Z3^-1` (unconstrained)
                ///
                /// Applies the Jacobian-to-affine conversion:
                ///
                ///     lambda = lambda_numerator * z_inv
                ///     x3     = X3 * z_inv^2
                ///     y3     = Y3 * z_inv^3
                pub unconstrained fn from_j_with_hint(
                    j_tx: JTranscript<BN>,
                    inverse: BN,
                ) -> AffineTranscript<BN> {
                    let z_inv = inverse;
                    let zz = z_inv.__mul(z_inv);
                    let zzz = zz.__mul(z_inv);
                    let lambda = j_tx.lambda_numerator.__mul(z_inv);
                    let x3 = j_tx.x3.__mul(zz);
                    let y3 = j_tx.y3.__mul(zzz);
                    AffineTranscript { x3, y3, lambda }
                }

                /// Batch-convert an array of `JTranscript` to `AffineTranscript` (unconstrained)
                ///
                /// ## Algorithm
                ///
                /// Uses Montgomery's batch inversion trick to compute all `Z3^-1` values
                /// with a single modular inverse, then converts each entry to affine:
                /// `lambda = lambda_numerator / Z3`, `x3 = X3 / Z3^2`, `y3 = Y3 / Z3^3`.
                pub(crate) unconstrained fn from_jacobian_transcript<let NumEntries: u32>(
                    j_tx: [JTranscript<BN>; NumEntries],
                ) -> [AffineTranscript<BN>; NumEntries] {
                    let mut result: [AffineTranscript<BN>; NumEntries] = [AffineTranscript::new(); NumEntries];

                    let mut inverses: [BN; NumEntries] = [BN::zero(); NumEntries];
                    for i in 0..j_tx.len() {
                        inverses[i] = j_tx[i].z3;
                    }

                    // Montgomery batch inversion: one modular inverse for all Z3 values
                    let inverses: [BN; NumEntries] = crate::mavros::vendor::bignum::bignum::batch_invert(inverses);

                    for i in 0..j_tx.len() {
                        let z_inv = inverses[i];
                        let zz = z_inv.__mul(z_inv);
                        let zzz = zz.__mul(z_inv);
                        let lambda = j_tx[i].lambda_numerator.__mul(z_inv);
                        let x3 = j_tx[i].x3.__mul(zz);
                        let y3 = j_tx[i].y3.__mul(zzz);
                        result[i] = AffineTranscript { lambda, x3, y3 };
                    }
                    result
                }
            }
        }

        pub mod point_table {

            use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
            use crate::mavros::vendor::bigcurve::bigcurve::BigCurve;
            use crate::mavros::vendor::bigcurve::curve_jac::types::{CurveJ, JTranscript};
            use crate::mavros::vendor::bignum::BigNum;

            /// A lookup table used for scalar multiplications using windowed non-adjacent form (WNAF).
            ///
            /// The scalar multiplier is sliced into 4-bit chunks, each represented in WNAF (values in the range [-15, -13, ..., 15]).
            /// A table `T` is computed containing point multiples corresponding to these 4-bit WNAF values: `T = ([-15[P], -13[P], ..., 15[P]])`.
            ///
            /// The accumulator is initialized with `T[most significant WNAF slice]`.
            /// For each remaining slice (starting from the most significant), the accumulator is doubled 4 times,
            /// then `T[slice[i]]` is added to the accumulator.
            ///
            /// For small multi-scalar multiplications (fewer than 512 points), this approach minimizes the number of addition operations.
            pub struct PointTable<BN> {
                x: [BN; 16],
                y: [BN; 16],
                z: [BN; 16],
                pub(crate) transcript: [JTranscript<BN>; 8],
            }

            impl<BN> PointTable<BN>
            where
                BN: BigNum,
            {
                pub(crate) fn empty() -> Self {
                    // Safety: empty table generation
                    PointTable {
                        x: [BN::zero(); 16],
                        y: [BN::zero(); 16],
                        z: [BN::zero(); 16],
                        transcript: [unsafe { JTranscript::new() }; 8],
                    }
                }

                /// Build a `PointTable` from a single input point `P` (unconstrained)
                ///
                /// ## Algorithm
                ///
                /// Exploits the symmetry of odd multiples: `-kP` shares the same `(X, Z)` as `kP`
                /// with negated `Y`. This halves the required group operations:
                ///
                ///   1. Compute `D = 2P` (one doubling)
                ///   2. Set `table[8] = P` (i.e. `1*P`), `table[7] = -P`
                ///   3. For `i = 1..8`: `A = D + A`, set `table[8+i] = A`, `table[7-i] = -A`
                ///
                /// This fills 16 entries `[-15P, -13P, ..., -P, P, ..., 15P]` with only
                /// 8 group operations (1 doubling + 7 additions), versus 14 for a naive table.
                ///
                /// The wNAF slice value `s` maps to table index `(s + 15) / 2`:
                ///   `s = -15 -> idx 0`, `s = -13 -> idx 1`, ..., `s = 15 -> idx 15`.
                pub(crate) unconstrained fn new<Curve>(P: CurveJ<BN, Curve>) -> Self
                where
                    Curve: BigCurve<BN>,
                {
                    let mut result = PointTable {
                        x: [BN::zero(); 16],
                        y: [BN::zero(); 16],
                        z: [BN::zero(); 16],
                        transcript: [JTranscript::new(); 8],
                    };
                    let op = P.dbl();
                    let D2 = op.0;

                    result.transcript[0] = op.1;
                    result.x[7] = P.x;
                    result.y[7] = P.y.__neg();
                    result.z[7] = P.z;
                    result.x[8] = P.x;
                    result.y[8] = P.y;
                    result.z[8] = P.z;
                    let mut A = P;
                    for i in 1..8 {
                        let op = D2.incomplete_add(A);
                        A = op.0;
                        result.transcript[i] = op.1;
                        result.x[8 + i] = A.x;
                        result.y[8 + i] = A.y;
                        result.z[8 + i] = A.z;
                        result.x[7 - i] = A.x;
                        result.y[7 - i] = A.y.__neg();
                        result.z[7 - i] = A.z;
                    }

                    result
                }

                /// Look up a point by wNAF table index (unconstrained)
                ///
                /// `idx` is `(wnaf_slice + 15) / 2`, ranging from 0 (`-15P`) to 15 (`+15P`).
                pub(crate) unconstrained fn get<Curve>(self, idx: u32) -> CurveJ<BN, Curve> {
                    CurveJ { x: self.x[idx], y: self.y[idx], z: self.z[idx], is_infinity: false }
                }
            }
        }

        pub mod ops {

            use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
            use crate::mavros::vendor::bigcurve::bigcurve::BigCurve;
            use crate::mavros::vendor::bigcurve::curve_jac::point_table::PointTable;
            use crate::mavros::vendor::bigcurve::curve_jac::types::{AffineTranscript, CurveJ, JTranscript};
            use crate::mavros::vendor::bigcurve::scalar_field::ScalarField;
            use crate::mavros::vendor::bignum::BigNum;

            /// Construct from BigCurve
            impl<BN, Curve> crate::convert::From<Curve> for CurveJ<BN, Curve>
            where
                BN: BigNum,
                Curve: BigCurve<BN>,
            {
                fn from(affine_point: Curve) -> Self {
                    CurveJ {
                        x: affine_point.x(),
                        y: affine_point.y(),
                        z: BN::one(),
                        is_infinity: affine_point.is_infinity(),
                    }
                }
            }

            /// Are two Jacobian points equal?
            ///
            /// Two Jacobian points `(X1, Y1, Z1)` and `(X2, Y2, Z2)` represent the same affine point iff:
            ///
            ///     X1 * Z2^2 == X2 * Z1^2   AND   Y1 * Z2^3 == Y2 * Z1^3
            ///
            /// Points at infinity are handled separately: `O == O`, and `O != P` for any finite `P`.
            impl<BN, Curve> crate::cmp::Eq for CurveJ<BN, Curve>
            where
                BN: BigNum,
                Curve: BigCurve<BN>,
            {
                fn eq(self, other: Self) -> bool {
                    let mut points_equal: bool = false;

                    // Safety: unconstrained equality check, used only in tests
                    unsafe {
                        let z1: BN = self.z;
                        let z2: BN = other.z;
                        let z1z1: BN = z1.__mul(z1);
                        let z1z1z1: BN = z1z1.__mul(z1);
                        let z2z2: BN = z2.__mul(z2);
                        let z2z2z2: BN = z2z2.__mul(z2);

                        let x_lhs: BN = self.x.__mul(z2z2);
                        let x_rhs: BN = other.x.__mul(z1z1);

                        let y_lhs: BN = self.y.__mul(z2z2z2);
                        let y_rhs: BN = other.y.__mul(z1z1z1);

                        let lhs_infinity: bool = self.is_infinity;
                        let rhs_infinity: bool = other.is_infinity;

                        let neither_infinity: bool = !lhs_infinity & !rhs_infinity;
                        let both_infinity: bool = lhs_infinity & rhs_infinity;

                        // Coordinates only meaningful when both points are finite
                        points_equal = (x_lhs.__eq(x_rhs) & y_lhs.__eq(y_rhs)) & neither_infinity;
                        points_equal = points_equal | both_infinity;
                    }
                    points_equal
                }
            }

            impl<BN, Curve> CurveJ<BN, Curve>
            where
                BN: BigNum,
                Curve: BigCurve<BN>,
            {
                /// Negate a point
                pub(crate) unconstrained fn neg(self) -> Self {
                    CurveJ { x: self.x, y: self.y.__neg(), z: self.z, is_infinity: self.is_infinity }
                }

                pub(crate) unconstrained fn new() -> Self {
                    CurveJ { x: BN::zero(), y: BN::zero(), z: BN::zero(), is_infinity: false }
                }

                pub(crate) unconstrained fn point_at_infinity() -> Self {
                    CurveJ { x: BN::zero(), y: BN::zero(), z: BN::zero(), is_infinity: true }
                }

                pub unconstrained fn sub(self, p2: Self) -> (Self, JTranscript<BN>) {
                    self.add(p2.neg())
                }

                /// Complete addition of two Jacobian points with full edge-case handling (unconstrained)
                ///
                /// Dispatches to `dbl`, `incomplete_add`, or returns identity/operand based on:
                ///   - **double**: same affine x and y, neither infinity -> `self.dbl()`
                ///   - **add**: different affine x, neither infinity -> `self.incomplete_add(p2)`
                ///   - **infinity result**: same x but different y, or both infinity -> point at infinity
                ///   - **one side infinity**: return the other operand
                pub unconstrained fn add(self, p2: Self) -> (Self, JTranscript<BN>) {
                    let X1 = self.x;
                    let X2 = p2.x;

                    let Y1 = self.y;
                    let Y2 = p2.y;

                    let Z1 = self.z;
                    let Z2 = p2.z;

                    let Z2Z2 = Z2.__mul(Z2);
                    let Z1Z1 = Z1.__mul(Z1);
                    let Z2Z2Z2 = Z2Z2.__mul(Z2);
                    let Z1Z1Z1 = Z1Z1.__mul(Z1);
                    let U1 = X1.__mul(Z2Z2);
                    let U2 = X2.__mul(Z1Z1);
                    let S1 = Y1.__mul(Z2Z2Z2);
                    let S2 = Y2.__mul(Z1Z1Z1);

                    let x_equal_predicate: bool = U2.__eq(U1);
                    let y_equal_predicate: bool = S2.__eq(S1);
                    let lhs_infinity: bool = self.is_infinity;
                    let rhs_infinity: bool = p2.is_infinity;

                    let double_predicate: bool =
                        x_equal_predicate & y_equal_predicate & !lhs_infinity & !rhs_infinity;

                    let add_predicate: bool = !x_equal_predicate & !lhs_infinity & !rhs_infinity;
                    let infinity_predicate: bool =
                        (x_equal_predicate & !y_equal_predicate) | (lhs_infinity & rhs_infinity);

                    let mut result: (Self, JTranscript<BN>) = (CurveJ::new(), JTranscript::new());

                    if (double_predicate) {
                        result = self.dbl();
                    } else if (add_predicate) {
                        result = self.incomplete_add(p2);
                    } else if (infinity_predicate) {
                        result = (CurveJ::point_at_infinity(), JTranscript::new());
                    } else if (lhs_infinity & !rhs_infinity) {
                        let new_transcript =
                            JTranscript { x3: p2.x, y3: p2.y, z3: p2.z, lambda_numerator: BN::zero() };
                        result = (p2, new_transcript);
                    } else if (rhs_infinity & !lhs_infinity) {
                        let new_transcript =
                            JTranscript { x3: self.x, y3: self.y, z3: self.z, lambda_numerator: BN::zero() };
                        result = (self, new_transcript);
                    }
                    result
                }

                /// Add two Jacobian points using incomplete formulas (unconstrained)
                ///
                /// Given `P1 = (X1, Y1, Z1)` and `P2 = (X2, Y2, Z2)`, computes `P1 + P2` as:
                ///
                /// ## Algorithm
                ///
                ///     U1 = X1*Z2^2,  U2 = X2*Z1^2
                ///     S1 = Y1*Z2^3,  S2 = Y2*Z1^3
                ///     H  = U2 - U1,  R  = S2 - S1
                ///
                ///     PP = H^2
                ///     X3 = R^2 - PP*(U1 + U2)
                ///     Y3 = PP*(U1*S2 - U2*S1) - X3*R
                ///     Z3 = Z1*Z2*H
                ///
                /// Requires `U1 != U2` (i.e. distinct x-coordinates in affine). With offset generators,
                /// this invariant is guaranteed and the constrained phase asserts it explicitly.
                ///
                /// ## Note
                /// This method minimizes the number of calls to `compute_quadratic_expression`,
                /// which is NOT the same as minimizing the number of multiplications.
                pub(crate) unconstrained fn incomplete_add(self, p2: Self) -> (Self, JTranscript<BN>) {
                    let X1 = self.x;
                    let X2 = p2.x;
                    let Y1 = self.y;
                    let Y2 = p2.y;
                    let Z1 = self.z;
                    let Z2 = p2.z;
                    let Z2Z2 = Z2.__mul(Z2);
                    let Z1Z1 = Z1.__mul(Z1);
                    let Z2Z2Z2 = Z2Z2.__mul(Z2);
                    let Z1Z1Z1 = Z1Z1.__mul(Z1);
                    let U1 = X1.__mul(Z2Z2);
                    let U2 = X2.__mul(Z1Z1);
                    let S1 = Y1.__mul(Z2Z2Z2);
                    let S2 = Y2.__mul(Z1Z1Z1);
                    let R = S2.__sub(S1);

                    // PP = (U2 - U1)^2
                    let (_, PP): (BN, BN) = crate::mavros::vendor::bignum::bignum::compute_quadratic_expression(
                        [[U2, U1]],
                        [[false, true]],
                        [[U2, U1]],
                        [[false, true]],
                        [],
                        [],
                    );

                    // X3 = -PP * (U1 + U2) + R * R
                    let (_, X3): (BN, BN) = crate::mavros::vendor::bignum::bignum::compute_quadratic_expression(
                        [[BN::zero(), PP], [R, BN::zero()]],
                        [[false, true], [false, false]],
                        [[U1, U2], [R, BN::zero()]],
                        [[false, false], [false, false]],
                        [],
                        [],
                    );

                    // U1S2_minus_U2S1 = U1 * S2 - U2 * S1
                    let (_, U1S2_minus_U2S1): (BN, BN) = crate::mavros::vendor::bignum::bignum::compute_quadratic_expression(
                        [[U1], [U2]],
                        [[false], [true]],
                        [[S2], [S1]],
                        [[false], [false]],
                        [],
                        [],
                    );

                    // Y3 = PP * (U1 * S2 - U2 * S1) + X3 * (-R)
                    let (_, Y3): (BN, BN) = crate::mavros::vendor::bignum::bignum::compute_quadratic_expression(
                        [[PP], [X3]],
                        [[false], [false]],
                        [[U1S2_minus_U2S1], [R]],
                        [[false], [true]],
                        [],
                        [],
                    );

                    let Z1Z2 = Z1.__mul(Z2);

                    // Z3 = Z1*Z2 * (U2 - U1)
                    let (_, Z3): (BN, BN) = crate::mavros::vendor::bignum::bignum::compute_quadratic_expression(
                        [[Z1Z2, BN::zero()]],
                        [[false, false]],
                        [[U2, U1]],
                        [[false, true]],
                        [],
                        [],
                    );

                    (
                        CurveJ { x: X3, y: Y3, z: Z3, is_infinity: false },
                        JTranscript { lambda_numerator: R, x3: X3, y3: Y3, z3: Z3 },
                    )
                }

                unconstrained fn conditional_incomplete_add(
                    self,
                    p2: Self,
                    predicate: bool,
                ) -> (Self, JTranscript<BN>) {
                    let operand_output = self.incomplete_add(p2);
                    let result = if predicate { operand_output.0 } else { self };
                    (result, operand_output.1)
                }

                /// Double a Jacobian point (unconstrained)
                ///
                /// Given `P = (X, Y, Z)`, computes `2P` as:
                ///
                /// ## Algorithm
                ///
                ///     M  = 3*X^2 + a*Z^4        (a = curve parameter, skipped when a == 0)
                ///     S  = 4*X*Y^2
                ///     X3 = M^2 - 2*S
                ///     Y3 = M*(S - X3) - 8*Y^4
                ///     Z3 = 2*Y*Z
                ///
                /// ## Note
                /// This method minimizes the number of calls to `compute_quadratic_expression`,
                /// which is NOT the same as minimizing the number of multiplications.
                pub(crate) unconstrained fn dbl(self) -> (Self, JTranscript<BN>) {
                    let X = self.x;
                    let Y = self.y;
                    let Z = self.z;

                    // YY_mul_2 = Y * (Y + Y)
                    let (_, YY_mul_2): (BN, BN) = crate::mavros::vendor::bignum::bignum::compute_quadratic_expression(
                        [[Y]],
                        [[false]],
                        [[Y, Y]],
                        [[false, false]],
                        [],
                        [],
                    );

                    // XX_mul_3 = X * (X + X + X)
                    let mut (_, XX_mul_3): (BN, BN) = crate::mavros::vendor::bignum::bignum::compute_quadratic_expression(
                        [[X]],
                        [[false]],
                        [[X, X, X]],
                        [[false, false, false]],
                        [],
                        [],
                    );

                    if (!Curve::a().__is_zero()) {
                        let ZZ = Z.__mul(Z);
                        let AZZZZ = ZZ.__mul(ZZ).__mul(Curve::a());
                        XX_mul_3 = XX_mul_3.__add(AZZZZ);
                    }

                    // D = (X + X) * YY_mul_2
                    let (_, D): (BN, BN) = crate::mavros::vendor::bignum::bignum::compute_quadratic_expression(
                        [[X, X]],
                        [[false, false]],
                        [[YY_mul_2]],
                        [[false]],
                        [],
                        [],
                    );

                    // X3 = XX_mul_3 * XX_mul_3 - D - D
                    let (_, X3): (BN, BN) = crate::mavros::vendor::bignum::bignum::compute_quadratic_expression(
                        [[XX_mul_3]],
                        [[false]],
                        [[XX_mul_3]],
                        [[false]],
                        [D, D],
                        [true, true],
                    );

                    // Y3 = XX_mul_3 * (D - X3) + YY_mul_2 * (YY_mul_2 + YY_mul_2)
                    let (_, Y3): (BN, BN) = crate::mavros::vendor::bignum::bignum::compute_quadratic_expression(
                        [[XX_mul_3], [YY_mul_2]],
                        [[false], [true]],
                        [[D, X3], [YY_mul_2, YY_mul_2]],
                        [[false, true], [false, false]],
                        [],
                        [],
                    );

                    // Z3 = Y * (Z + Z)
                    let (_, Z3): (BN, BN) = crate::mavros::vendor::bignum::bignum::compute_quadratic_expression(
                        [[Y]],
                        [[false]],
                        [[Z, Z]],
                        [[false, false]],
                        [],
                        [],
                    );

                    (
                        CurveJ { x: X3, y: Y3, z: Z3, is_infinity: false },
                        JTranscript { lambda_numerator: XX_mul_3, x3: X3, y3: Y3, z3: Z3 },
                    )
                }

                pub(crate) unconstrained fn offset_generator() -> Self {
                    let result = Curve::offset_generator();
                    Self { x: result.x(), y: result.y(), z: BN::one(), is_infinity: false }
                }

                pub(crate) unconstrained fn offset_generator_final() -> Self {
                    let result = Curve::offset_generator_final();
                    Self { x: result.x(), y: result.y(), z: BN::one(), is_infinity: false }
                }

                pub(crate) unconstrained fn one() -> Self {
                    let result = Curve::one();
                    Self { x: result.x(), y: result.y(), z: BN::one(), is_infinity: false }
                }

                /// Perform an EC scalar multiplication using wNAF and output the generated AffineTranscript
                /// (unconstrained)
                ///
                /// ## Algorithm
                ///
                /// Uses 4-bit windowed Non-Adjacent Form (wNAF) with a precomputed `PointTable`
                /// of `[P, 3P, 5P, ..., 15P]`.
                ///
                ///   1. Build the `PointTable` (8 entries, each with a `JTranscript`)
                ///   2. Start with `acc = offset_generator + T[slice[0]]`
                ///   3. For each subsequent slice `i`:  `acc = 16*acc + T[slice[i]]`
                ///      (the `16*` is four doublings)
                ///   4. If `skew`, subtract `P` (corrects for the wNAF encoding of even scalars)
                ///   5. Subtract `offset_generator_final` to remove the offset
                pub(crate) unconstrained fn mul<let NScalarSlices: u32>(
                    self,
                    scalar: ScalarField<NScalarSlices>,
                ) -> (Self, [AffineTranscript<BN>; 5 * NScalarSlices + 6]) {
                    let (input, scalar) = if self.is_infinity {
                        (CurveJ::one(), ScalarField::zero())
                    } else {
                        (self, scalar)
                    };

                    let mut ptr: u32 = 0;
                    let T: PointTable<BN> = PointTable::new(input);
                    let mut transcript: [JTranscript<BN>; NScalarSlices * 5 + 6] =
                        [JTranscript::new(); NScalarSlices * 5 + 6];
                    for i in 0..8 {
                        transcript[ptr] = T.transcript[i];
                        ptr += 1;
                    }

                    let mut accumulator: Self = CurveJ::offset_generator();
                    let op = accumulator.incomplete_add(T.get(scalar.base4_slices[0] as u32));
                    transcript[ptr] = op.1;
                    ptr += 1;
                    accumulator = op.0;
                    for i in 1..NScalarSlices {
                        for _ in 0..4 {
                            let op = accumulator.dbl();
                            accumulator = op.0;
                            transcript[ptr] = op.1;
                            ptr += 1;
                        }
                        let op = accumulator.incomplete_add(T.get(scalar.base4_slices[i] as u32));
                        transcript[ptr] = op.1;
                        ptr += 1;
                        accumulator = op.0;
                    }

                    let op = accumulator.conditional_incomplete_add(input.neg(), scalar.skew);
                    transcript[ptr] = op.1;
                    accumulator = op.0;
                    ptr += 1;

                    let op = accumulator.sub(CurveJ::offset_generator_final());
                    transcript[ptr] = op.1;
                    ptr += 1;
                    accumulator = op.0;

                    let affine_transcript: [AffineTranscript<BN>; 5 * NScalarSlices + 6] =
                        AffineTranscript::from_jacobian_transcript(transcript);

                    (accumulator, affine_transcript)
                }

                /// Perform the inner loop of a multi-scalar multiplication (unconstrained)
                ///
                /// Computes `sum_i(scalars[i] * points[i]) + offset_generator` without the final
                /// offset subtraction (that is done by `msm` or `compute_linear_expression_transcript`).
                ///
                /// ## Algorithm
                ///
                /// Interleaved wNAF MSM:
                ///
                ///   1. Build a `PointTable` for each input point
                ///   2. `acc = offset_generator + T_0[slice_0[0]] + T_1[slice_1[0]] + ...`
                ///   3. For each subsequent slice index `i`:
                ///      - Four doublings: `acc = 16 * acc`
                ///      - Add each point's contribution: `acc += T_j[slice_j[i]]` for all `j`
                ///   4. Apply skew corrections: conditionally subtract each `points[i]`
                pub(crate) unconstrained fn msm_partial<let Size: u32, let NScalarSlices: u32, let TranscriptEntries: u32>(
                    points: [Self; Size],
                    scalars: [ScalarField<NScalarSlices>; Size],
                ) -> (Self, [JTranscript<BN>; TranscriptEntries]) {
                    let mut transcript: [JTranscript<BN>; TranscriptEntries] =
                        [JTranscript::new(); TranscriptEntries];
                    let mut tables: [PointTable<BN>; Size] = [PointTable::empty(); Size];

                    let mut _inputs: [Self; Size] = [CurveJ::new(); Size];
                    let mut _scalars: [ScalarField<NScalarSlices>; Size] = [ScalarField::zero(); Size];
                    for i in 0..Size {
                        let (input, scalar) = if points[i].is_infinity {
                            (CurveJ::one(), crate::mavros::vendor::bigcurve::scalar_field::ScalarField::zero())
                        } else {
                            (points[i], scalars[i])
                        };
                        _inputs[i] = input;
                        _scalars[i] = scalar;
                    }
                    let points = _inputs;
                    let scalars = _scalars;
                    let mut ptr: u32 = 0;
                    for i in 0..Size {
                        tables[i] = PointTable::new(points[i]);
                        for j in 0..8 {
                            transcript[ptr] = tables[i].transcript[j];
                            ptr += 1;
                        }
                    }

                    let mut accumulator: Self = CurveJ::offset_generator();
                    let op = accumulator.incomplete_add(tables[0].get(scalars[0].base4_slices[0] as u32));
                    transcript[ptr] = op.1;
                    ptr += 1;
                    accumulator = op.0;

                    for i in 1..Size {
                        let op = accumulator.incomplete_add(tables[i].get(scalars[i].base4_slices[0] as u32));
                        transcript[ptr] = op.1;
                        ptr += 1;
                        accumulator = op.0;
                    }
                    for i in 1..NScalarSlices {
                        for _ in 0..4 {
                            let op = accumulator.dbl();
                            accumulator = op.0;
                            transcript[ptr] = op.1;
                            ptr += 1;
                        }
                        for j in 0..Size {
                            let op =
                                accumulator.incomplete_add(tables[j].get(scalars[j].base4_slices[i] as u32));
                            transcript[ptr] = op.1;
                            ptr += 1;
                            accumulator = op.0;
                        }
                    }

                    for i in 0..Size {
                        let op = accumulator.conditional_incomplete_add(points[i].neg(), scalars[i].skew);
                        transcript[ptr] = op.1;
                        accumulator = op.0;
                        ptr += 1;
                    }

                    (accumulator, transcript)
                }

                /// Perform a multi-scalar multiplication and output the generated AffineTranscript
                /// (unconstrained)
                ///
                /// Wraps `msm_partial`: runs the interleaved wNAF MSM, subtracts `offset_generator_final`,
                /// and converts the Jacobian transcript to affine via batch inversion.
                pub(crate) unconstrained fn msm<let Size: u32, let NScalarSlices: u32>(
                    mut points: [Self; Size],
                    mut scalars: [ScalarField<NScalarSlices>; Size],
                ) -> (Self, [AffineTranscript<BN>; NScalarSlices * Size + NScalarSlices * 4 + Size * 9 - 3]) {
                    let mut (accumulator, transcript)
                        : (Self, [JTranscript<BN>; NScalarSlices * Size + NScalarSlices * 4 + Size * 9 - 3]) =
                        CurveJ::msm_partial(points, scalars);
                    let op = accumulator.sub(CurveJ::offset_generator_final());
                    transcript[NScalarSlices * Size + NScalarSlices * 4 + Size * 9 - 4] = op.1;
                    accumulator = op.0;
                    let affine_transcript: [AffineTranscript<BN>; NScalarSlices * Size + NScalarSlices * 4 + Size * 9 - 3] =
                        AffineTranscript::from_jacobian_transcript(transcript);

                    (accumulator, affine_transcript)
                }

                /// Compute `sum_i(scalars[i] * mul_points[i]) + sum_j(add_points[j])` (unconstrained)
                ///
                /// Combines MSM with plain additions: runs `msm_partial` on the scalar-point pairs,
                /// then conditionally adds each `add_points[j]` (skipping infinity points),
                /// and finally subtracts `offset_generator_final`.
                pub(crate) unconstrained fn compute_linear_expression_transcript<let NScalarSlices: u32, let NMuls: u32, let NAdds: u32>(
                    mut mul_points: [Curve; NMuls],
                    mut scalars: [ScalarField<NScalarSlices>; NMuls],
                    mut add_points: [Curve; NAdds],
                ) -> (Self, [AffineTranscript<BN>; NScalarSlices * NMuls + NScalarSlices * 4 + NMuls * 9 + NAdds - 3]) {
                    let mut mul_j: [CurveJ<BN, Curve>; NMuls] = [CurveJ::new(); NMuls];
                    let mut add_j: [CurveJ<BN, Curve>; NAdds] = [CurveJ::new(); NAdds];
                    for i in 0..NMuls {
                        mul_j[i] = CurveJ::<BN, Curve>::from(mul_points[i]);
                    }
                    for i in 0..NAdds {
                        add_j[i] = CurveJ::from(add_points[i]);
                    }

                    let mut (accumulator, transcript)
                        : (Self, [JTranscript<BN>; NScalarSlices * NMuls + NScalarSlices * 4 + NMuls * 9 + NAdds - 3]) =
                        CurveJ::msm_partial(mul_j, scalars);
                    let mut transcript_ptr: u32 = NScalarSlices * NMuls + NScalarSlices * 4 + NMuls * 9 - 4;
                    for i in 0..NAdds {
                        let op = accumulator.conditional_incomplete_add(add_j[i], !add_j[i].is_infinity);
                        transcript[transcript_ptr] = op.1;
                        accumulator = op.0;
                        transcript_ptr += 1;
                    }

                    let op = accumulator.sub(CurveJ::offset_generator_final());
                    transcript[transcript_ptr] = op.1;
                    accumulator = op.0;
                    let affine_transcript: [AffineTranscript<BN>; NScalarSlices * NMuls + NScalarSlices * 4 + NMuls * 9 + NAdds - 3] =
                        AffineTranscript::from_jacobian_transcript(transcript);

                    (accumulator, affine_transcript)
                }
            }
        }

    }

    pub mod ops {

        use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
        pub mod affine {
            use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
            use crate::mavros::vendor::bigcurve::bigcurve::BigCurve;
            use crate::mavros::vendor::bigcurve::curve_jac::types::AffineTranscript;
            use crate::mavros::vendor::bignum::BigNum;
            use crate::mavros::vendor::bignum::bignum::evaluate_quadratic_expression;

            /// Double a point using an `AffineTranscript` witness (constrained)
            ///
            /// Verifies three constraints against the transcript values `(lambda, x3, y3)`:
            ///
            ///   1. `lambda * 2*y1 - 3*x1^2 - a = 0`   (tangent slope)
            ///   2. `lambda^2 - x3 - 2*x1 = 0`         (output x-coordinate)
            ///   3. `lambda * (x3 - x1) + y3 + y1 = 0` (output y-coordinate)
            pub(crate) fn double_with_hint<BN: BigNum, P: BigCurve<BN>>(
                point: P,
                transcript: AffineTranscript<BN>,
            ) -> P {
                let x1: BN = point.x();
                let y1: BN = point.y();
                let lambda: BN = transcript.lambda;
                let x3: BN = transcript.x3;
                let y3: BN = transcript.y3;

                // Witness values must be well-formed BigNum elements
                x3.validate_in_range();
                y3.validate_in_range();
                lambda.validate_in_range();

                // Constraint 1: lambda * 2*y1 - 3*x1^2 - a = 0
                evaluate_quadratic_expression(
                    [[x1, x1, x1], [y1, y1, BN::zero()]],
                    [[false, false, false], [false, false, false]],
                    [[x1], [lambda]],
                    [[false], [true]],
                    [P::a()],
                    [false],
                );

                // Constraint 2: lambda^2 - x3 - 2*x1 = 0
                evaluate_quadratic_expression(
                    [[lambda]],
                    [[false]],
                    [[lambda]],
                    [[false]],
                    [x3, x1, x1],
                    [true, true, true],
                );

                // Constraint 3: lambda * (x3 - x1) + y3 + y1 = 0
                evaluate_quadratic_expression(
                    [[lambda]],
                    [[false]],
                    [[x3, x1]],
                    [[false, true]],
                    [y3, y1],
                    [false, false],
                );

                P::from_coordinates(x3, y3, false)
            }

            /// Add two points using incomplete formulae (constrained)
            ///
            /// With offset generators, an honest prover never encounters edge cases.
            /// Instead of handling them, we assert `x1 != x2` for soundness.
            ///
            /// Verifies three constraints against the transcript values `(lambda, x3, y3)`:
            ///
            ///   1. `lambda * (x2 - x1) - (y2 - y1) = 0`  (chord slope)
            ///   2. `lambda^2 - x3 - x2 - x1 = 0`         (output x-coordinate)
            ///   3. `lambda * (x3 - x1) + y3 + y1 = 0`    (output y-coordinate)
            pub(crate) fn incomplete_add_with_hint<BN: BigNum, P: BigCurve<BN>>(
                point: P,
                other: P,
                transcript: AffineTranscript<BN>,
            ) -> P {
                let x1: BN = point.x();
                let y1: BN = point.y();
                let x2: BN = other.x();
                let y2: BN = other.y();
                let lambda: BN = transcript.lambda;
                let x3: BN = transcript.x3;
                let y3: BN = transcript.y3;

                // Witness values must be well-formed BigNum elements
                x3.validate_in_range();
                y3.validate_in_range();
                lambda.validate_in_range();

                // Incomplete addition requires distinct x-coordinates
                x1.assert_is_not_equal(x2);

                // Constraint 1: lambda * (x2 - x1) - (y2 - y1) = 0
                evaluate_quadratic_expression(
                    [[lambda]],
                    [[false]],
                    [[x2, x1]],
                    [[false, true]],
                    [y2, y1],
                    [true, false],
                );

                // Constraint 2: lambda^2 - x3 - x2 - x1 = 0
                evaluate_quadratic_expression(
                    [[lambda]],
                    [[false]],
                    [[lambda]],
                    [[false]],
                    [x3, x2, x1],
                    [true, true, true],
                );

                // Constraint 3: lambda * (x3 - x1) + y3 + y1 = 0
                evaluate_quadratic_expression(
                    [[lambda]],
                    [[false]],
                    [[x3, x1]],
                    [[false, true]],
                    [y3, y1],
                    [false, false],
                );

                P::from_coordinates(x3, y3, false)
            }

            /// Conditionally add two points using incomplete formulae (constrained)
            ///
            /// If `predicate` is true, returns `point + other`. Otherwise returns `point` unchanged.
            /// Callers guarantee `point.x != other.x` via the offset generator.
            pub(crate) fn conditional_incomplete_add_with_hint<BN: BigNum, P: BigCurve<BN>>(
                point: P,
                other: P,
                predicate: bool,
                transcript: AffineTranscript<BN>,
            ) -> P {
                let operand_output: P = incomplete_add_with_hint::<BN, P>(point, other, transcript);
                if predicate {
                    operand_output
                } else {
                    point
                }
            }

            /// Conditionally subtract a point using incomplete formulae (constrained)
            ///
            /// If `predicate` is true, returns `point - other`. Otherwise returns `point` unchanged.
            /// Callers guarantee `point.x != other.x` via the offset generator.
            pub(crate) fn conditional_incomplete_subtract_with_hint<BN: BigNum, P: BigCurve<BN>>(
                point: P,
                other: P,
                predicate: bool,
                transcript: AffineTranscript<BN>,
            ) -> P {
                let operand_output: P = incomplete_subtract_with_hint::<BN, P>(point, other, transcript);
                if predicate {
                    operand_output
                } else {
                    point
                }
            }

            /// Subtract two points using incomplete formulae (constrained)
            ///
            /// Computes `point - other` by verifying the subtraction transcript.
            /// Asserts `x1 != x2` (offset generators guarantee this for an honest prover).
            ///
            /// Verifies three constraints:
            ///
            ///   1. `lambda * (x2 - x1) + y2 + y1 = 0`    (chord slope for subtraction)
            ///   2. `lambda^2 - x3 - x2 - x1 = 0`         (output x-coordinate)
            ///   3. `lambda * (x3 - x1) + y3 + y1 = 0`    (output y-coordinate)
            fn incomplete_subtract_with_hint<BN: BigNum, P: BigCurve<BN>>(
                point: P,
                other: P,
                transcript: AffineTranscript<BN>,
            ) -> P {
                let x1: BN = point.x();
                let y1: BN = point.y();
                let x2: BN = other.x();
                let y2: BN = other.y();
                let lambda: BN = transcript.lambda;
                let x3: BN = transcript.x3;
                let y3: BN = transcript.y3;

                // Witness values must be well-formed BigNum elements
                x3.validate_in_range();
                y3.validate_in_range();
                lambda.validate_in_range();

                // Incomplete subtraction requires distinct x-coordinates
                x1.assert_is_not_equal(x2);

                // Constraint 1: lambda * (x2 - x1) + y2 + y1 = 0
                evaluate_quadratic_expression(
                    [[lambda]],
                    [[false]],
                    [[x2, x1]],
                    [[false, true]],
                    [y2, y1],
                    [false, false],
                );

                // Constraint 2: lambda^2 - x3 - x2 - x1 = 0
                evaluate_quadratic_expression(
                    [[lambda]],
                    [[false]],
                    [[lambda]],
                    [[false]],
                    [x3, x2, x1],
                    [true, true, true],
                );

                // Constraint 3: lambda * (x3 - x1) + y3 + y1 = 0
                evaluate_quadratic_expression(
                    [[lambda]],
                    [[false]],
                    [[x3, x1]],
                    [[false, true]],
                    [y3, y1],
                    [false, false],
                );

                P::from_coordinates(x3, y3, false)
            }

            /// Complete point addition with edge-case handling (constrained)
            ///
            /// Handles all cases: distinct points, doubling (`x1 == x2, y1 == y2`),
            /// inverse points (`x1 == x2, y1 != y2` -> infinity), and infinity inputs.
            ///
            /// Dynamically selects between addition and doubling constraints via conditional
            /// operand substitution, so only one `evaluate_quadratic_expression` call validates
            /// the lambda constraint regardless of the case.
            pub fn add_with_hint<BN: BigNum, P: BigCurve<BN>>(
                point: P,
                other: P,
                transcript: AffineTranscript<BN>,
            ) -> P {
                let x1: BN = point.x();
                let y1: BN = point.y();
                let x2: BN = other.x();
                let y2: BN = other.y();
                let lambda: BN = transcript.lambda;
                let x3: BN = transcript.x3;
                let y3: BN = transcript.y3;

                // Witness values must be well-formed BigNum elements
                x3.validate_in_range();
                y3.validate_in_range();
                lambda.validate_in_range();

                // Determine which case applies
                let x_equal_predicate: bool = x2.eq(x1);
                let y_equal_predicate: bool = y2.eq(y1);
                // If x1 == x2, use doubling formula. Also use it when either input is infinity (operands are zeroed below).
                let double_predicate: bool = x_equal_predicate | (point.is_infinity() | other.is_infinity());
                let infinity_predicate: bool =
                    (x_equal_predicate & !y_equal_predicate) | (point.is_infinity() & other.is_infinity());
                let evaluate_group_operation_predicate: bool =
                    !infinity_predicate & !point.is_infinity() & !other.is_infinity();

                // Zero all operands when skipping the group operation (inverse points or infinity inputs)
                let (x1, y1, x2, y2, x3, y3) = if evaluate_group_operation_predicate {
                    (x1, y1, x2, y2, x3, y3)
                } else {
                    (BN::zero(), BN::zero(), BN::zero(), BN::zero(), BN::zero(), BN::zero())
                };

                // Select operands for a unified lambda constraint:
                //   double: lambda * (y1 + y1) + x1 * (-x1 - x1 - x1) - a = 0
                //   add:    lambda * (x2 - x1) + 1 * (-y2 + y1) = 0
                let product_1_rhs_t0: BN = if double_predicate { y1 } else { x2 };
                let product_1_rhs_t1: BN = if double_predicate { y1 } else { x1.neg() };
                let product_2_lhs_t0: BN = if double_predicate { x1 } else { BN::one() };
                let product_2_rhs_t0: BN = if double_predicate { x1 } else { y2 };
                let product_2_rhs_t1: BN = if double_predicate { x1 } else { y1.neg() };
                let product_2_rhs_t2: BN = if double_predicate { x1 } else { BN::zero() };

                // Constraint 1: lambda is correct (unified double/add)
                let a_term: BN = if x_equal_predicate
                    & y_equal_predicate
                    & !point.is_infinity()
                    & !other.is_infinity() {
                    P::a()
                } else {
                    BN::zero()
                };

                evaluate_quadratic_expression(
                    [[lambda], [product_2_lhs_t0]],
                    [[false], [false]],
                    [
                        [product_1_rhs_t0, product_1_rhs_t1, BN::zero()],
                        [product_2_rhs_t0, product_2_rhs_t1, product_2_rhs_t2],
                    ],
                    [[false, false, false], [true, true, true]],
                    [a_term],
                    [true],
                );

                // Constraint 2: lambda^2 - x3 - x2 - x1 = 0
                evaluate_quadratic_expression(
                    [[lambda]],
                    [[false]],
                    [[lambda]],
                    [[false]],
                    [x3, x2, x1],
                    [true, true, true],
                );

                // Constraint 3: lambda * (x3 - x1) + y3 + y1 = 0
                evaluate_quadratic_expression(
                    [[lambda]],
                    [[false]],
                    [[x3, x1]],
                    [[false, true]],
                    [y3, y1],
                    [false, false],
                );

                // Select output based on edge cases
                let output_is_lhs: bool = other.is_infinity() & !point.is_infinity();
                let output_is_rhs: bool = point.is_infinity() & !other.is_infinity();
                let mut x_out: BN = if evaluate_group_operation_predicate {
                    x3
                } else {
                    x1
                };
                x_out = if output_is_lhs { point.x() } else { x_out };
                x_out = if output_is_rhs { other.x() } else { x_out };
                x_out = if infinity_predicate {
                    BN::zero()
                } else {
                    x_out
                };
                let mut y_out: BN = if evaluate_group_operation_predicate {
                    y3
                } else {
                    y1
                };
                y_out = if output_is_lhs { point.y() } else { y_out };
                y_out = if output_is_rhs { other.y() } else { y_out };
                y_out = if infinity_predicate {
                    BN::zero()
                } else {
                    y_out
                };

                let mut infinity_out: bool = !evaluate_group_operation_predicate;
                infinity_out = if infinity_predicate {
                    true
                } else {
                    infinity_out
                };
                infinity_out = if output_is_lhs {
                    point.is_infinity()
                } else {
                    infinity_out
                };
                infinity_out = if output_is_rhs {
                    other.is_infinity()
                } else {
                    infinity_out
                };

                P::from_coordinates(x_out, y_out, infinity_out)
            }

            /// Complete point subtraction with edge-case handling (constrained)
            ///
            /// Computes `point - other`. Handles all cases: distinct points, doubling
            /// (`x1 == x2, y1 == -y2`), inverse points (`x1 == x2, y1 == y2` -> infinity),
            /// and infinity inputs.
            pub fn sub_with_hint<BN: BigNum, P: BigCurve<BN>>(
                point: P,
                other: P,
                transcript: AffineTranscript<BN>,
            ) -> P {
                let x1: BN = point.x();
                let y1: BN = point.y();
                let x2: BN = other.x();
                let y2: BN = other.y();
                let lambda: BN = transcript.lambda;
                let x3: BN = transcript.x3;
                let y3: BN = transcript.y3;

                // Witness values must be well-formed BigNum elements
                x3.validate_in_range();
                y3.validate_in_range();
                lambda.validate_in_range();

                // Determine which case applies (subtraction: y_equal means inverse points -> infinity)
                let x_equal_predicate: bool = x2.eq(x1);
                let y_equal_predicate: bool = y2.eq(y1) == false;
                let double_predicate: bool = x_equal_predicate | (point.is_infinity() | other.is_infinity());
                let infinity_predicate: bool =
                    (x_equal_predicate & !y_equal_predicate) | (point.is_infinity() & other.is_infinity());
                let evaluate_group_operation_predicate: bool =
                    !infinity_predicate & !point.is_infinity() & !other.is_infinity();

                // Zero all operands when skipping the group operation
                let (x1, y1, x2, y2, x3, y3) = if evaluate_group_operation_predicate {
                    (x1, y1, x2, y2, x3, y3)
                } else {
                    (BN::zero(), BN::zero(), BN::zero(), BN::zero(), BN::zero(), BN::zero())
                };

                // Select operands for a unified lambda constraint:
                //   double: -lambda * (y1 + y1) + x1 * (x1 + x1 + x1) + a = 0
                //   sub:    -lambda * (-x2 + x1) + 1 * (y2 + y1) = 0
                let product_1_rhs_t0: BN = if double_predicate { y1 } else { x2.neg() };
                let product_1_rhs_t1: BN = if double_predicate { y1 } else { x1 };
                let product_2_lhs_t0: BN = if double_predicate { x1 } else { BN::one() };
                let product_2_rhs_t0: BN = if double_predicate { x1 } else { y2 };
                let product_2_rhs_t1: BN = if double_predicate { x1 } else { y1 };
                let product_2_rhs_t2: BN = if double_predicate { x1 } else { BN::zero() };

                // Constraint 1: lambda is correct (unified double/sub)
                let a_term: BN = if x_equal_predicate
                    & y_equal_predicate
                    & !point.is_infinity()
                    & !other.is_infinity() {
                    P::a()
                } else {
                    BN::zero()
                };

                evaluate_quadratic_expression(
                    [[lambda], [product_2_lhs_t0]],
                    [[true], [false]],
                    [
                        [product_1_rhs_t0, product_1_rhs_t1, BN::zero()],
                        [product_2_rhs_t0, product_2_rhs_t1, product_2_rhs_t2],
                    ],
                    [[false, false, false], [false, false, false]],
                    [a_term],
                    [false],
                );

                // Constraint 2: lambda^2 - x3 - x2 - x1 = 0
                evaluate_quadratic_expression(
                    [[lambda]],
                    [[false]],
                    [[lambda]],
                    [[false]],
                    [x3, x2, x1],
                    [true, true, true],
                );

                // Constraint 3: lambda * (x3 - x1) + y3 + y1 = 0
                evaluate_quadratic_expression(
                    [[lambda]],
                    [[false]],
                    [[x3, x1]],
                    [[false, true]],
                    [y3, y1],
                    [false, false],
                );

                // Select output based on edge cases
                let output_is_lhs: bool = other.is_infinity() & !point.is_infinity();
                let output_is_rhs: bool = point.is_infinity() & !other.is_infinity();
                let mut x_out: BN = if evaluate_group_operation_predicate {
                    x3
                } else {
                    x1
                };
                x_out = if output_is_lhs { point.x() } else { x_out };
                x_out = if output_is_rhs { other.x() } else { x_out };
                x_out = if infinity_predicate {
                    BN::zero()
                } else {
                    x_out
                };
                let mut y_out: BN = if evaluate_group_operation_predicate {
                    y3
                } else {
                    y1
                };
                y_out = if output_is_lhs { point.y() } else { y_out };
                y_out = if output_is_rhs {
                    other.y().neg()
                } else {
                    y_out
                };
                y_out = if infinity_predicate {
                    BN::zero()
                } else {
                    y_out
                };

                let mut infinity_out: bool = !evaluate_group_operation_predicate;
                infinity_out = if infinity_predicate {
                    true
                } else {
                    infinity_out
                };
                infinity_out = if output_is_lhs {
                    point.is_infinity()
                } else {
                    infinity_out
                };
                infinity_out = if output_is_rhs {
                    other.is_infinity()
                } else {
                    infinity_out
                };

                P::from_coordinates(x_out, y_out, infinity_out)
            }
        }

        pub mod point_table {

            use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
            use crate::mavros::vendor::bigcurve::bigcurve::BigCurve;
            use crate::mavros::vendor::bigcurve::curve_jac::types::AffineTranscript;
            use crate::mavros::vendor::bigcurve::ops::affine::{double_with_hint, incomplete_add_with_hint};
            use crate::mavros::vendor::bignum::BigNum;

            /// Affine-coordinate lookup table for wNAF scalar multiplication (constrained side).
            ///
            /// Stores 16 entries `[-15P, -13P, ..., -P, P, ..., 15P]` in affine `(x, y)` form.
            /// The wNAF slice value `s` maps to table index `(s + 15) / 2`.
            ///
            /// For MSMs with fewer than 512 points, a per-point table minimizes addition operations.
            pub(crate) struct PointTable<BN> {
                x: [BN; 16],
                y: [BN; 16],
            }

            impl<BN: BigNum> PointTable<BN> {
                pub(crate) fn empty() -> Self {
                    PointTable { x: [BN::zero(); 16], y: [BN::zero(); 16] }
                }

                /// Build a `PointTable` from an input point, verifying the transcript (constrained)
                ///
                /// ## Algorithm
                ///
                /// Same symmetry trick as the Jacobian `PointTable`: `-kP` shares the same `x`
                /// with negated `y`. Each step is verified via `double_with_hint` / `incomplete_add_with_hint`.
                ///
                ///   1. `D = 2P` (one constrained doubling)
                ///   2. `table[8] = P`, `table[7] = -P`
                ///   3. For `i = 1..8`: `A = A + D`, `table[8+i] = A`, `table[7-i] = -A`
                pub(crate) fn new_with_hint<P: BigCurve<BN>>(
                    point: P,
                    transcript: [AffineTranscript<BN>; 8],
                ) -> Self {
                    let mut result: PointTable<BN> = PointTable { x: [BN::zero(); 16], y: [BN::zero(); 16] };

                    let D2: P = double_with_hint::<BN, P>(point, transcript[0]);

                    result.x[8] = point.x();
                    result.y[8] = point.y();
                    result.x[7] = point.x();
                    result.y[7] = point.y().neg();

                    let mut A: P = point;
                    for i in 1..8 {
                        A = incomplete_add_with_hint::<BN, P>(A, D2, transcript[i]);
                        result.x[8 + i] = A.x();
                        result.y[8 + i] = A.y();
                        result.x[7 - i] = A.x();
                        result.y[7 - i] = A.y().neg();
                    }
                    result
                }

                /// Look up a point by wNAF table index (constrained)
                ///
                /// `idx` is `(wnaf_slice + 15) / 2`, ranging from 0 (`-15P`) to 15 (`+15P`).
                /// With the barretenberg backend, lookups on 256-bit curves cost ~12 constraints.
                pub(crate) fn get<P: BigCurve<BN>>(self, idx: u32) -> P {
                    P::from_coordinates(self.x[idx], self.y[idx], false)
                }
            }
        }

        pub mod msm {

            use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
            use crate::mavros::vendor::bigcurve::bigcurve::BigCurve;
            use crate::mavros::vendor::bigcurve::curve_jac::types::{AffineTranscript, CurveJ};
            use crate::mavros::vendor::bigcurve::ops::affine::{
                conditional_incomplete_add_with_hint, conditional_incomplete_subtract_with_hint,
                double_with_hint, incomplete_add_with_hint, sub_with_hint,
            };
            use crate::mavros::vendor::bigcurve::ops::point_table::PointTable;
            use crate::mavros::vendor::bigcurve::scalar_field::ScalarField;
            use crate::mavros::vendor::bignum::BigNum;

            /// Perform a multi-scalar multiplication (constrained)
            ///
            /// Computes `sum_i(mul_scalars[i] * mul_points[i])`.
            /// Delegates to `evaluate_linear_expression` with no additive points.
            pub fn msm<P: BigCurve<BN>, BN: BigNum, let NScalarSlices: u32, let NMuls: u32>(
                mul_points: [P; NMuls],
                mul_scalars: [ScalarField<NScalarSlices>; NMuls],
            ) -> P {
                let add_points: [P; 0] = [];
                P::evaluate_linear_expression(mul_points, mul_scalars, add_points)
            }

            /// Perform the inner MSM loop, verifying an `AffineTranscript` (constrained)
            ///
            /// Builds point tables, then replays the interleaved wNAF double-and-add loop,
            /// verifying each step against the provided transcript. Does NOT subtract
            /// `offset_generator_final` -- the caller is responsible for that.
            fn msm_with_hint_internal<let Size: u32, let NScalarSlices: u32, let NTranscriptSlices: u32, BN: BigNum, P: BigCurve<BN>>(
                mut points: [P; Size],
                mut scalars: [ScalarField<NScalarSlices>; Size],
                transcript: [AffineTranscript<BN>; NTranscriptSlices],
            ) -> P {
                // Replace infinity inputs with the generator (scalar zeroed to produce identity)
                let mut _inputs: [P; Size] = [P::one(); Size];
                let mut _scalars: [ScalarField<NScalarSlices>; Size] = [ScalarField::zero(); Size];
                for i in 0..Size {
                    let (input, scalar) = if points[i].is_infinity() {
                        (P::one(), ScalarField::zero())
                    } else {
                        (points[i], scalars[i])
                    };
                    _inputs[i] = input;
                    _scalars[i] = scalar;
                }
                points = _inputs;
                scalars = _scalars;

                let mut tables: [PointTable<BN>; Size] = [PointTable::empty(); Size];
                for i in 0..Size {
                    let mut table_transcript: [AffineTranscript<BN>; 8] = [AffineTranscript::new(); 8];
                    for j in 0..8 {
                        table_transcript[j] = transcript[i * 8 + j];
                    }
                    tables[i] = PointTable::new_with_hint::<P>(points[i], table_transcript);
                }

                // Initialize accumulator: offset_generator + T_0[slice_0[0]] + T_1[slice_1[0]] + ...
                let mut accumulator: P = P::offset_generator();
                let mut accumulator: P = incomplete_add_with_hint::<BN, P>(
                    accumulator,
                    tables[0].get::<P>(scalars[0].base4_slices[0] as u32),
                    transcript[8 * Size],
                );

                for i in 1..Size {
                    accumulator = incomplete_add_with_hint::<BN, P>(
                        accumulator,
                        tables[i].get::<P>(scalars[i].base4_slices[0] as u32),
                        transcript[8 * Size + i],
                    );
                }

                // For each subsequent slice: 4 doublings then add each point's table lookup
                for i in 1..NScalarSlices {
                    accumulator =
                        double_with_hint::<BN, P>(accumulator, transcript[9 * Size + (4 + Size) * (i - 1)]);
                    accumulator =
                        double_with_hint::<BN, P>(accumulator, transcript[9 * Size + (4 + Size) * (i - 1) + 1]);
                    accumulator =
                        double_with_hint::<BN, P>(accumulator, transcript[9 * Size + (4 + Size) * (i - 1) + 2]);
                    accumulator =
                        double_with_hint::<BN, P>(accumulator, transcript[9 * Size + (4 + Size) * (i - 1) + 3]);

                    for j in 0..Size {
                        accumulator = incomplete_add_with_hint::<BN, P>(
                            accumulator,
                            tables[j].get::<P>(scalars[j].base4_slices[i] as u32),
                            transcript[9 * Size + (4 + Size) * (i - 1) + 4 + j],
                        );
                    }
                }

                // wNAF skew correction: subtract P_i when scalar_i is even
                for i in 0..Size {
                    accumulator = conditional_incomplete_subtract_with_hint::<BN, P>(
                        accumulator,
                        points[i],
                        scalars[i].skew,
                        transcript[9 * Size + (4 + Size) * (NScalarSlices - 1) + i],
                    );
                }

                accumulator
            }

            /// Perform a single scalar multiplication, verifying an `AffineTranscript` (constrained)
            ///
            /// Computes `scalar * point` by replaying the wNAF double-and-add loop against
            /// the provided transcript, then subtracts `offset_generator_final`.
            pub fn mul_with_hint<let NScalarSlices: u32, let NTranscriptSlices: u32, BN: BigNum, P: BigCurve<BN>>(
                point: P,
                scalar: ScalarField<NScalarSlices>,
                transcript: [AffineTranscript<BN>; NTranscriptSlices],
            ) -> P {
                let (input, scalar): (P, ScalarField<NScalarSlices>) = if point.is_infinity() {
                    (P::one(), ScalarField::zero())
                } else {
                    (point, scalar)
                };
                let T: PointTable<BN> = PointTable::new_with_hint::<P>(
                    input,
                    [
                        transcript[0],
                        transcript[1],
                        transcript[2],
                        transcript[3],
                        transcript[4],
                        transcript[5],
                        transcript[6],
                        transcript[7],
                    ],
                );

                // Initialize accumulator: offset_generator + T[slice[0]]
                let mut accumulator: P = P::offset_generator();
                let mut accumulator: P = incomplete_add_with_hint::<BN, P>(
                    accumulator,
                    T.get::<P>(scalar.base4_slices[0] as u32),
                    transcript[3 + 5],
                );

                // For each subsequent slice: 4 doublings then add table lookup
                for i in 1..NScalarSlices {
                    accumulator = double_with_hint::<BN, P>(accumulator, transcript[4 + 5 * i]);
                    accumulator = double_with_hint::<BN, P>(accumulator, transcript[5 + 5 * i]);
                    accumulator = double_with_hint::<BN, P>(accumulator, transcript[6 + 5 * i]);
                    accumulator = double_with_hint::<BN, P>(accumulator, transcript[7 + 5 * i]);
                    accumulator = incomplete_add_with_hint::<BN, P>(
                        accumulator,
                        T.get::<P>(scalar.base4_slices[i] as u32),
                        transcript[8 + 5 * i],
                    );
                }

                // wNAF skew correction: subtract P when scalar is even
                accumulator = conditional_incomplete_subtract_with_hint::<BN, P>(
                    accumulator,
                    input,
                    scalar.skew,
                    transcript[4 + 5 * NScalarSlices],
                );

                // Remove the offset generator
                accumulator = sub_with_hint::<BN, P>(
                    accumulator,
                    P::offset_generator_final(),
                    transcript[5 + 5 * NScalarSlices],
                );
                accumulator
            }

            /// Perform an MSM and subtract offset generator (constrained)
            ///
            /// ## Note
            /// Not used anywhere
            fn msm_with_hint<let Size: u32, let NScalarSlices: u32, let NTranscriptSlices: u32, BN: BigNum, P: BigCurve<BN>>(
                mut points: [P; Size],
                mut scalars: [ScalarField<NScalarSlices>; Size],
                transcript: [AffineTranscript<BN>; NTranscriptSlices],
            ) -> P {
                let mut accumulator: P = msm_with_hint_internal::<Size, NScalarSlices, NTranscriptSlices, BN, P>(
                    points,
                    scalars,
                    transcript,
                );
                accumulator = sub_with_hint::<BN, P>(
                    accumulator,
                    P::offset_generator_final(),
                    transcript[10 * Size + (4 + Size) * (NScalarSlices - 1)],
                );
                accumulator
            }

            /// Generate the `AffineTranscript` for a single scalar multiplication (unconstrained)
            pub unconstrained fn get_mul_transcript<let NScalarSlices: u32, BN: BigNum, P: BigCurve<BN>>(
                point: P,
                scalar: ScalarField<NScalarSlices>,
            ) -> [AffineTranscript<BN>; 6 + NScalarSlices * 5] {
                let tmp: CurveJ<BN, P> = CurveJ::from(point);
                let (_, transcript) = tmp.mul(scalar);
                transcript
            }

            /// Evaluate `sum_i(mul_scalars[i] * mul_points[i]) + sum_j(add_points[j])` (constrained)
            ///
            /// Generates the transcript via unconstrained witness generation, then replays
            /// the full MSM + additions loop, verifying each step against the transcript.
            pub fn evaluate_linear_expression<BN: BigNum, Curve: BigCurve<BN>, let NScalarSlices: u32, let NMuls: u32, let NAdds: u32>(
                mul_points: [Curve; NMuls],
                mul_scalars: [ScalarField<NScalarSlices>; NMuls],
                add_points: [Curve; NAdds],
            ) -> Curve {
                // Safety: transcript is verified step-by-step in the constrained replay loop below
                let transcript: [AffineTranscript<BN>; NScalarSlices * NMuls + NScalarSlices * 4 + NMuls * 9 + NAdds - 3] = unsafe {
                    CurveJ::<BN, Curve>::compute_linear_expression_transcript(
                        mul_points,
                        mul_scalars,
                        add_points,
                    )
                        .1
                };

                // Replace infinity inputs with the generator (scalar zeroed to produce identity)
                let mut _inputs: [Curve; NMuls] = [BigCurve::one(); NMuls];
                let mut _scalars: [ScalarField<NScalarSlices>; NMuls] = [ScalarField::zero(); NMuls];
                for i in 0..NMuls {
                    let (input, scalar) = if mul_points[i].is_infinity() {
                        (BigCurve::one(), ScalarField::zero())
                    } else {
                        (mul_points[i], mul_scalars[i])
                    };
                    _inputs[i] = input;
                    _scalars[i] = scalar;
                }
                let msm_points: [Curve; NMuls] = _inputs;
                let scalars: [ScalarField<NScalarSlices>; NMuls] = _scalars;

                let mut tables: [PointTable<BN>; NMuls] = [PointTable::empty(); NMuls];
                for i in 0..NMuls {
                    let mut table_transcript: [AffineTranscript<BN>; 8] = [AffineTranscript::new(); 8];
                    for j in 0..8 {
                        table_transcript[j] = transcript[i * 8 + j];
                    }
                    tables[i] = PointTable::<BN>::new_with_hint::<Curve>(msm_points[i], table_transcript);
                }

                // Initialize accumulator: offset_generator + T_0[slice_0[0]] + T_1[slice_1[0]] + ...
                let mut accumulator: Curve = BigCurve::offset_generator();
                let mut accumulator: Curve = incomplete_add_with_hint::<BN, Curve>(
                    accumulator,
                    tables[0].get::<Curve>(scalars[0].base4_slices[0] as u32),
                    transcript[8 * NMuls],
                );
                for i in 1..NMuls {
                    accumulator = incomplete_add_with_hint::<BN, Curve>(
                        accumulator,
                        tables[i].get::<Curve>(scalars[i].base4_slices[0] as u32),
                        transcript[8 * NMuls + i],
                    );
                }

                // For each subsequent slice: 4 doublings then add each point's table lookup
                for i in 1..NScalarSlices {
                    accumulator = double_with_hint::<BN, Curve>(
                        accumulator,
                        transcript[9 * NMuls + (4 + NMuls) * (i - 1)],
                    );
                    accumulator = double_with_hint::<BN, Curve>(
                        accumulator,
                        transcript[9 * NMuls + (4 + NMuls) * (i - 1) + 1],
                    );
                    accumulator = double_with_hint::<BN, Curve>(
                        accumulator,
                        transcript[9 * NMuls + (4 + NMuls) * (i - 1) + 2],
                    );
                    accumulator = double_with_hint::<BN, Curve>(
                        accumulator,
                        transcript[9 * NMuls + (4 + NMuls) * (i - 1) + 3],
                    );
                    for j in 0..NMuls {
                        accumulator = incomplete_add_with_hint::<BN, Curve>(
                            accumulator,
                            tables[j].get::<Curve>(scalars[j].base4_slices[i] as u32),
                            transcript[9 * NMuls + (4 + NMuls) * (i - 1) + 4 + j],
                        );
                    }
                }

                // wNAF skew correction: subtract P_i when scalar_i is even
                for i in 0..NMuls {
                    accumulator = conditional_incomplete_subtract_with_hint::<BN, Curve>(
                        accumulator,
                        msm_points[i],
                        scalars[i].skew,
                        transcript[9 * NMuls + (4 + NMuls) * (NScalarSlices - 1) + i],
                    );
                }

                // Add the additive points (skipping infinity)
                for i in 0..NAdds {
                    accumulator = conditional_incomplete_add_with_hint::<BN, Curve>(
                        accumulator,
                        add_points[i],
                        !add_points[i].is_infinity(),
                        transcript[10 * NMuls + (4 + NMuls) * (NScalarSlices - 1) + i],
                    );
                }

                // Remove the offset generator
                accumulator = sub_with_hint::<BN, Curve>(
                    accumulator,
                    BigCurve::offset_generator_final(),
                    transcript[NScalarSlices * NMuls + NScalarSlices * 4 + NMuls * 9 + NAdds - 4],
                );

                accumulator
            }
        }

    }

    pub mod curves {

        use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
        pub mod secp256r1 {
            use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
            use crate::mavros::vendor::bignum::BigNum;
            pub use crate::mavros::vendor::bignum::{Secp256r1_Fq, Secp256r1_Fr};
            use crate::mavros::vendor::bignum::fields::secp256r1Fq::secp256r1_fq_from_limbs;

            use crate::mavros::vendor::bigcurve::bigcurve::BigCurve;
            use crate::mavros::vendor::bigcurve::bigcurve::BigCurveParams;
            use crate::mavros::vendor::bigcurve::bigcurve::derive_curve_impl;
            use crate::mavros::vendor::bigcurve::scalar_field::ScalarField;

            use crate::mavros::vendor::bigcurve::curve_jac::types::CurveJ;

            pub(crate) type Secp256r1J = CurveJ<Secp256r1_Fq, Secp256r1>;

            pub global SECP256r1_PARAM: BigCurveParams<Secp256r1_Fq> = BigCurveParams {
                a: secp256r1_fq_from_limbs([
                    0xfffffffffffffffffffffffc,
                    0xffff00000001000000000000000000,
                    0xffff,
                ]),
                b: secp256r1_fq_from_limbs([
                    0x1d06b0cc53b0f63bce3c3e27d2604b,
                    0x35d8aa3a93e7b3ebbd55769886bc65,
                    0x5ac6,
                ]),
                one: [
                    secp256r1_fq_from_limbs([
                        0x037d812deb33a0f4a13945d898c296,
                        0xd1f2e12c4247f8bce6e563a440f277,
                        0x6b17,
                    ]),
                    secp256r1_fq_from_limbs([
                        0xce33576b315ececbb6406837bf51f5,
                        0x42e2fe1a7f9b8ee7eb4a7c0f9e162b,
                        0x4fe3,
                    ]),
                ],
                offset_generator: [
                    secp256r1_fq_from_limbs([
                        351296668306655348126371036589410180,
                        14099522553413269749449432069244305,
                        23572,
                    ]),
                    secp256r1_fq_from_limbs([
                        531116666173442154165394067505655294,
                        354979494418088410567905796042406908,
                        33819,
                    ]),
                ],
                offset_generator_final: [
                    secp256r1_fq_from_limbs([
                        0x4e53e542bc502419d0cb4b99c0aa48,
                        0x74d3a045b26542b1a5931d721650ab,
                        0x6982,
                    ]),
                    secp256r1_fq_from_limbs([
                        0xbe3c1ef0581c8f76517e499277b269,
                        0xfd562740986614ce3b7025c0dfb8aa,
                        0xfc88,
                    ]),
                ],
            };

            pub global SECP256r1_SCALAR_SLICES: u32 = 65;

            pub type Secp256r1Scalar = ScalarField<SECP256r1_SCALAR_SLICES>;

            #[derive_curve_impl(quote { crate::mavros::vendor::bignum::Secp256r1_Fq }, quote { SECP256r1_PARAM })]
            pub struct Secp256r1 {
                pub x: Secp256r1_Fq,
                pub y: Secp256r1_Fq,
                pub is_infinity: bool,
            }
        }
    }

    pub mod internal {

        use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
        pub use crate::mavros::vendor::bignum::bignum::evaluate_quadratic_expression;
        pub use super::ops::msm::{evaluate_linear_expression, get_mul_transcript, mul_with_hint};
        pub use super::ops::affine::{add_with_hint, sub_with_hint};
    }

    pub use bigcurve::{BigCurve, BigCurveParams, derive_curve_impl};
    pub use scalar_field::ScalarField;
}
