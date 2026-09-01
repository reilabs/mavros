// Adapted compatibility subset of zkpassport/noir-bignum v0.10.0-1 and the bigcurve package in
// AztecProtocol/aztec-packages at commit 9df414ec0e810d46d44b0357341f8dae0abe82b9. Kept inside the
// injected std crate so low-level foreign replacements follow Mavros' existing pure-Noir path. See
// the root THIRD_PARTY_LICENSES file.

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

        fn from_coordinates_unsafe(x: BN, y: BN, is_infinity: bool) -> Self;

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
                    let result = Self { x, y, is_infinity };
                    result.validate_on_curve();
                    result
                }

                fn from_coordinates_unsafe(x: $field_type, y: $field_type, is_infinity: bool) -> Self {
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

            P::from_coordinates_unsafe(x3, y3, false)
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

            P::from_coordinates_unsafe(x3, y3, false)
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

            P::from_coordinates_unsafe(x3, y3, false)
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

            P::from_coordinates_unsafe(x_out, y_out, infinity_out)
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
                P::from_coordinates_unsafe(self.x[idx], self.y[idx], false)
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
    pub mod secp256k1 {
        use crate::mavros::vendor::bigcurve::bigcurve::{
            BigCurve, BigCurveParams, derive_curve_impl,
        };
        use crate::mavros::vendor::bigcurve::scalar_field::ScalarField;
        use crate::mavros::vendor::bignum::BigNum;
        use crate::mavros::vendor::bignum::fields::secp256k1Fq::secp256k1_fq_from_limbs;
        pub use crate::mavros::vendor::bignum::{Secp256k1_Fq, Secp256k1_Fr};

        pub global SECP256k1_PARAM: BigCurveParams<Secp256k1_Fq> = BigCurveParams {
            a: secp256k1_fq_from_limbs([0x00, 0x00, 0x00]),
            b: secp256k1_fq_from_limbs([0x07, 0x00, 0x00]),
            one: [
                secp256k1_fq_from_limbs([
                    0x9bfcdb2dce28d959f2815b16f81798,
                    0x667ef9dcbbac55a06295ce870b0702,
                    0x79be,
                ]),
                secp256k1_fq_from_limbs([
                    0x17b448a68554199c47d08ffb10d4b8,
                    0xda7726a3c4655da4fbfc0e1108a8fd,
                    0x483a,
                ]),
            ],
            offset_generator: [
                secp256k1_fq_from_limbs([
                    0xad9b0399372a80a40485fb4765c830,
                    0x592524e36cf8f8b10532990e0c318b,
                    0x4130,
                ]),
                secp256k1_fq_from_limbs([
                    0x0f86d7ffdfd71295914f8c96c8e696,
                    0xb2a472eb30ec190dc82c3f416b8c6d,
                    0xfedf,
                ]),
            ],
            offset_generator_final: [
                secp256k1_fq_from_limbs([
                    0xf69e3511e0f28578e8516402091e2d,
                    0xa0afbd65c86c4bb7593dace897e155,
                    0x4f04,
                ]),
                secp256k1_fq_from_limbs([
                    0x414e4311b6eaed7bdf0a669162ba0b,
                    0x5cccfc054797f3e2b9133b36f2d7a7,
                    0x765c,
                ]),
            ],
        };

        #[derive_curve_impl(quote { crate::mavros::vendor::bignum::Secp256k1_Fq }, quote { SECP256k1_PARAM })]
        pub struct Secp256k1 {
            pub x: Secp256k1_Fq,
            pub y: Secp256k1_Fq,
            pub is_infinity: bool,
        }
    }

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
                    0x221da828a9fac3f3182aab2336419f,
                    0x1d4b43fc35f66f89260a3e92036a69,
                    0x0865,
                ]),
                secp256r1_fq_from_limbs([
                    0x0cb150c8de290839e3ba7ac9ca4e9a,
                    0x747dd91cdd0f0e456c02e570d40b9a,
                    0x56ef,
                ]),
            ],
            offset_generator_final: [
                secp256r1_fq_from_limbs([
                    0x9145ddf08596099a7d522ff60feeff,
                    0x26f0893dce2ba7914f469b05749605,
                    0xcebf,
                ]),
                secp256r1_fq_from_limbs([
                    0xb9172e6de500d0d4b72cf2322515ac,
                    0x1fc60013458aa1f5666f15620c3dcf,
                    0x371d,
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
