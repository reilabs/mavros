// Adapted compatibility subset of zkpassport/noir-bignum v0.10.0-1 and
// zkpassport/noir_bigcurve v0.14.0-1. Kept inside the injected std crate so low-level foreign
// replacements follow Mavros' existing pure-Noir path. See THIRD_PARTY_NOTICES.md.

pub mod bignum {

    use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
    pub mod constants {
        use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
        pub(crate) global TWO_POW_120: u128 = 0x1000000000000000000000000000000;
        pub(crate) global TWO_POW_126: Field = 0x40000000000000000000000000000000;
        pub(crate) global TWO_POW_246: Field =
            0x40000000000000000000000000000000000000000000000000000000000000;
    }

    pub mod params {

        use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
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

    }

    pub mod utils {

        use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
        pub mod split_bits {
            use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
            use crate::mavros::vendor::bignum::constants::TWO_POW_120;

            pub(crate) unconstrained fn __split_120_bits(mut x: Field) -> (u128, Field) {
                let low: u128 = (x as u128) % TWO_POW_120;
                let high: Field = ((x - low as Field) / TWO_POW_120 as Field);
                (low, high)
            }

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

            use crate::mavros::vendor::bignum::fns::unconstrained_helpers::{
                __barrett_reduction, __half_mod_odd, __helper_mul, __helper_sub, __is_even, __one,
                __shr1,
            };

            use crate::mavros::vendor::bignum::constants::TWO_POW_120;

            use crate::mavros::vendor::bignum::params::BigNumParams;

            // ------------------------------ COMPARISON FUNCTIONS ------------------------------

            pub unconstrained fn __eq<let N: u32>(lhs: [u128; N], rhs: [u128; N]) -> bool {
                lhs == rhs
            }

            pub unconstrained fn __is_zero<let N: u32>(limbs: [u128; N]) -> bool {
                limbs == [0; N]
            }

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

            pub unconstrained fn __neg<let N: u32>(modulus: [u128; N], limbs: [u128; N]) -> [u128; N] {
                __helper_sub(modulus, limbs)
            }

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

            pub unconstrained fn __sub<let N: u32>(
                modulus: [u128; N],
                lhs: [u128; N],
                rhs: [u128; N],
            ) -> [u128; N] {
                __add(modulus, lhs, __neg(modulus, rhs))
            }

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

            pub unconstrained fn __mul<let N: u32, let MOD_BITS: u32>(
                params: BigNumParams<N, MOD_BITS>,
                lhs: [u128; N],
                rhs: [u128; N],
            ) -> [u128; N] {
                __mul_with_quotient::<N, MOD_BITS>(params, lhs, rhs).1
            }

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

        }

        pub mod constrained_ops {

            use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
            use crate::mavros::vendor::bignum::constants::TWO_POW_120;

            use crate::mavros::vendor::bignum::fns::{
                expressions::evaluate_quadratic_expression,
                unconstrained_helpers::{compute_add_flags, compute_borrow_flags, compute_gte_flags, compute_sub_flags},
                unconstrained_ops::{__add, __mul, __neg, __sub},
            };

            use crate::mavros::vendor::bignum::params::BigNumParams;

            use crate::cmp::Ordering;

            // ------------------------------ COMPARISON FUNCTIONS ------------------------------

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

            pub fn eq<let N: u32, let MOD_BITS: u32>(
                params: BigNumParams<N, MOD_BITS>,
                lhs: [u128; N],
                rhs: [u128; N],
            ) -> bool {
                let diff: [u128; N] = sub::<N, MOD_BITS>(params, lhs, rhs);
                is_zero::<N, MOD_BITS>(params, diff)
            }

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

            pub fn is_zero<let N: u32, let MOD_BITS: u32>(
                params: BigNumParams<N, MOD_BITS>,
                val: [u128; N],
            ) -> bool {
                is_zero_integer(val) | (val == params.modulus)
            }

            pub fn validate_in_range<T, let N: u32, let MOD_BITS: u32>(limbs: [T; N])
            where
                T: Into<Field>,
            {
                for i in 0..(N - 1) {
                    limbs[i].into().assert_max_bit_size::<120>();
                }

                limbs[N - 1].into().assert_max_bit_size::<MOD_BITS - ((N - 1) * 120)>();
            }

            pub(crate) fn validate_quotient_in_range<let N: u32, let MOD_BITS: u32>(limbs: [u128; N]) {
                for i in 0..(N - 1) {
                    (limbs[i] as Field).assert_max_bit_size::<120>();
                }
                (limbs[N - 1] as Field).assert_max_bit_size::<MOD_BITS - (N - 1) * 120 + 6>();
            }

            fn assert_sub_no_overflow<let N: u32, let MOD_BITS: u32>(
                lhs: [u128; N],
                rhs: [u128; N],
                borrow_flags: [bool; N - 1],
            ) -> [u128; N] {
                let mut result: [u128; N] = [0; N];
                let limb_0: Field =
                    (lhs[0] as Field) - (rhs[0] as Field) + (borrow_flags[0] as Field) * (TWO_POW_120 as Field);
                limb_0.assert_max_bit_size::<128>();
                result[0] = limb_0 as u128;

                for i in 1..N - 1 {
                    let limb_i: Field = (lhs[i] as Field) - (rhs[i] as Field)
                        + (borrow_flags[i] as Field) * (TWO_POW_120 as Field)
                        - (borrow_flags[i - 1] as Field);
                    limb_i.assert_max_bit_size::<128>();
                    result[i] = limb_i as u128;
                }

                let limb_last: Field =
                    (lhs[N - 1] as Field) - (rhs[N - 1] as Field) - (borrow_flags[N - 2] as Field);
                limb_last.assert_max_bit_size::<128>();
                result[N - 1] = limb_last as u128;
                validate_in_range::<u128, N, MOD_BITS>(result);
                result
            }

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

        }

        pub mod unconstrained_helpers {

            use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
            // This file contains the unconstrained helpers that are mostly used by unconstrained ops

            use crate::mavros::vendor::bignum::constants::TWO_POW_120;

            use crate::mavros::vendor::bignum::fns::unconstrained_ops::{__gte, __neg};
            use crate::mavros::vendor::bignum::utils::split_bits::{__normalize_limbs, __split_120_bits};

            use crate::mavros::vendor::bignum::params::BigNumParams;

            // ------------------------------ DERIVATION HELPER FUNCTIONS ------------------------------

            pub(crate) unconstrained fn __one<let N: u32>() -> [u128; N] {
                let mut limbs: [u128; N] = [0; N];
                limbs[0] = 1;
                limbs
            }

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

            global BARRETT_REDUCTION_OVERFLOW_BITS: u32 = 6;

            //
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

            pub(crate) unconstrained fn __is_even<let N: u32>(x: [u128; N]) -> bool {
                (x[0] & 1) == 0
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

            //
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

        }

        pub mod serialization {

            use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
            use crate::mavros::vendor::bignum::utils::map::invert_array;

            // Conversions between big endian and little endian byte arrays and BigNum instances
            // the byte serialization should have `(MOD_BITS + 7) / 8` bytes.
            // each 120-bit limb is represented by 15 bytes, and there are fewer bytes for covering the most significant limb

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

            pub fn to_le_bytes<let N: u32, let MOD_BITS: u32>(val: [u128; N]) -> [u8; (MOD_BITS + 7) / 8] {
                let result_be: [u8; (MOD_BITS + 7) / 8] = to_be_bytes(val);
                invert_array(result_be)
            }
        }

    }

    pub mod bignum {

        use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
        use crate::mavros::vendor::bignum::params::BigNumParams;
        use crate::ops::{Add, Mul, Neg, Sub};

        pub trait BigNum: Neg + Add + Sub + Mul + crate::cmp::Eq {
            let N: u32;
            let MOD_BITS: u32;

            fn params() -> BigNumParams<N, MOD_BITS>;
            fn modulus_bits(_: Self) -> u32;
            fn num_limbs(_: Self) -> u32;
            fn modulus() -> Self;
            fn zero() -> Self;
            fn one() -> Self;
            fn from_limbs(limbs: [u128; N]) -> Self;
            fn get_limbs(self) -> [u128; N];
            fn set_limb(self: &mut Self, idx: u32, value: u128);
            fn get_limb(self: Self, idx: u32) -> u128;
            fn from_be_bytes(x: [u8; (MOD_BITS + 7) / 8]) -> Self;
            fn to_le_bytes(self) -> [u8; (MOD_BITS + 7) / 8];

            unconstrained fn __eq(self: Self, other: Self) -> bool;
            unconstrained fn __is_zero(self: Self) -> bool;
            unconstrained fn __neg(self) -> Self;
            unconstrained fn __add(self, other: Self) -> Self;
            unconstrained fn __sub(self, other: Self) -> Self;
            unconstrained fn __mul(self, other: Self) -> Self;
            unconstrained fn __invmod(self) -> Self;

            fn assert_is_not_equal(self: Self, other: Self);
            fn validate_in_range(self);
            fn validate_in_field(self);
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
                impl $crate::mavros::vendor::bignum::BigNum for $typ {
                    let N: u32 = $N;
                    let MOD_BITS: u32 = $MOD_BITS;

                    fn params() -> $crate::mavros::vendor::bignum::params::BigNumParams<$N, $MOD_BITS> {
                        $params
                    }

                    fn modulus_bits(_: Self) -> u32 { $MOD_BITS }
                    fn num_limbs(_: Self) -> u32 { $N }
                    fn modulus() -> Self { Self { limbs: Self::params().modulus } }
                    fn zero() -> Self { Self { limbs: [0; $N] } }

                    fn one() -> Self {
                        let mut limbs = [0; $N];
                        limbs[0] = 1;
                        Self { limbs }
                    }

                    fn from_limbs(limbs: [u128; $N]) -> Self { Self { limbs } }
                    fn get_limbs(self: Self) -> [u128; $N] { self.limbs }

                    fn set_limb(self: &mut Self, idx: u32, value: u128) {
                        self.limbs[idx] = value;
                    }

                    fn get_limb(self: Self, idx: u32) -> u128 { self.limbs[idx] }

                    fn from_be_bytes(x: [u8; ($MOD_BITS + 7) / 8]) -> Self {
                        Self {
                            limbs: $crate::mavros::vendor::bignum::internal::from_be_bytes::<$N, $MOD_BITS>(x)
                        }
                    }

                    fn to_le_bytes(self) -> [u8; ($MOD_BITS + 7) / 8] {
                        $crate::mavros::vendor::bignum::internal::to_le_bytes::<$N, $MOD_BITS>(self.limbs)
                    }

                    unconstrained fn __eq(self: Self, other: Self) -> bool {
                        $crate::mavros::vendor::bignum::internal::__eq(self.limbs, other.limbs)
                    }

                    unconstrained fn __is_zero(self: Self) -> bool {
                        $crate::mavros::vendor::bignum::internal::__is_zero(self.limbs)
                    }

                    unconstrained fn __neg(self: Self) -> Self {
                        Self { limbs: $crate::mavros::vendor::bignum::internal::__neg(Self::params().modulus, self.limbs) }
                    }

                    unconstrained fn __add(self: Self, other: Self) -> Self {
                        Self { limbs: $crate::mavros::vendor::bignum::internal::__add(Self::params().modulus, self.limbs, other.limbs) }
                    }

                    unconstrained fn __sub(self: Self, other: Self) -> Self {
                        Self { limbs: $crate::mavros::vendor::bignum::internal::__sub(Self::params().modulus, self.limbs, other.limbs) }
                    }

                    unconstrained fn __mul(self: Self, other: Self) -> Self {
                        Self { limbs: $crate::mavros::vendor::bignum::internal::__mul(Self::params(), self.limbs, other.limbs) }
                    }

                    unconstrained fn __invmod(self: Self) -> Self {
                        Self { limbs: $crate::mavros::vendor::bignum::internal::__invmod(Self::params(), self.limbs) }
                    }

                    fn assert_is_not_equal(self: Self, other: Self) {
                        $crate::mavros::vendor::bignum::internal::assert_is_not_equal(
                            Self::params(),
                            self.limbs,
                            other.limbs,
                        );
                    }

                    fn validate_in_field(self: Self) {
                        $crate::mavros::vendor::bignum::internal::validate_in_field::<$N, $MOD_BITS>(
                            Self::params(),
                            self.limbs,
                        );
                    }

                    fn validate_in_range(self: Self) {
                        $crate::mavros::vendor::bignum::internal::validate_in_range::<u128, $N, $MOD_BITS>(self.limbs);
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


        pub fn conditional_select<T: BigNum>(lhs: T, rhs: T, predicate: bool) -> T {
            if predicate {
                lhs
            } else {
                rhs
            }
        }

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

        pub unconstrained fn batch_invert<T: BigNum, let M: u32>(x: [T; M]) -> [T; M] {
            let params = T::params();
            assert(params.has_multiplicative_inverse);
            crate::mavros::vendor::bignum::fns::unconstrained_ops::batch_invert(params, x.map(|bn: T| bn.get_limbs())).map(|limbs| {
                T::from_limbs(limbs)
            })
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

        }

    }

    pub mod internal {

        use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
        pub use super::fns::constrained_ops::{
            add, assert_is_not_equal, cmp, eq, mul, neg, sub, validate_in_field, validate_in_range,
        };
        pub use super::fns::unconstrained_ops::{
            __add, __eq, __invmod, __is_zero, __mul, __neg, __sub,
        };
        pub use super::fns::serialization::{from_be_bytes, to_le_bytes};
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

        pub struct ScalarField<let N: u32> {
            pub(crate) base4_slices: [u8; N],
            pub(crate) skew: bool,
        }

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

        impl<let N: u32> ScalarField<N> {
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

        }

        // ------------------------------ HELPER FUNCTIONS ------------------------------

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

        pub struct BigCurveParams<BN> {
            pub a: BN,
            pub b: BN,
            pub offset_generator: [BN; 2],
            pub offset_generator_final: [BN; 2],
            pub one: [BN; 2],
        }

        pub trait BigCurve<BN: BigNum> {
            fn x(self) -> BN;

            fn y(self) -> BN;

            fn a() -> BN;

            fn from_coordinates(x: BN, y: BN, is_infinity: bool) -> Self;

            fn is_infinity(self) -> bool;

            fn one() -> Self;

            fn offset_generator() -> Self;

            fn offset_generator_final() -> Self;

            fn validate_on_curve(self);

            fn evaluate_linear_expression<let NScalarSlices: u32, let NMuls: u32, let NAdds: u32>(
                mul_points: [Self; NMuls],
                mul_scalars: [ScalarField<NScalarSlices>; NMuls],
                add_points: [Self; NAdds],
            ) -> Self;
        }

        pub comptime fn derive_curve_impl(
            struct_def: TypeDefinition,
            field_type: Quoted,
            params: Quoted,
        ) -> Quoted {
            let typ = struct_def.as_type();
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

                    fn is_infinity(self) -> bool {
                        self.is_infinity
                    }

                    fn from_coordinates(x: $field_type, y: $field_type, is_infinity: bool) -> Self {
                        Self { x, y, is_infinity }
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

                    fn evaluate_linear_expression<let NScalarSlices: u32, let NMuls: u32, let NAdds: u32>(
                        mul_points: [Self; NMuls],
                        mul_scalars: [ScalarField<NScalarSlices>; NMuls],
                        add_points: [Self; NAdds],
                    ) -> Self {
                        $crate::mavros::vendor::bigcurve::internal::evaluate_linear_expression::<$field_type,$typ, _, _, _>(mul_points, mul_scalars, add_points)
                    }}

            }
        }
    }


    pub mod curve_jac {

        use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
        pub mod types {
            use crate::prelude::{Eq, Ord, From, Into, Default, Option, panic};
            use crate::mavros::vendor::bignum::BigNum;

            pub struct CurveJ<BN, BigCurve> {
                pub(crate) x: BN,
                pub(crate) y: BN,
                pub(crate) z: BN,
                pub is_infinity: bool,
            }

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

            pub(crate) struct PointTable<BN> {
                x: [BN; 16],
                y: [BN; 16],
            }

            impl<BN: BigNum> PointTable<BN> {
                pub(crate) fn empty() -> Self {
                    PointTable { x: [BN::zero(); 16], y: [BN::zero(); 16] }
                }

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
        pub use super::ops::msm::evaluate_linear_expression;
        pub use super::ops::affine::sub_with_hint;
    }

    pub use bigcurve::{BigCurve, BigCurveParams, derive_curve_impl};
    pub use scalar_field::ScalarField;
}
