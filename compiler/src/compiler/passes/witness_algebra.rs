use ark_ff::{AdditiveGroup, Field as _};

use crate::compiler::{
    Field,
    analysis::{types::FunctionTypeInfo, value_range_analysis::IntInterval},
    ssa::{
        ValueId,
        hlssa::{CastTarget, Endianness, OpCode, Radix, TypeExpr, builder::HLEmitter},
    },
};

/// Rangecheck `value in [0, 2^bits)` for any `bits >= 1`.
///
/// Cost:
///   bits == 1                     -> 0 lookups (boolean check, 2 algebraic constraints)
///   bits == 8q (byte-aligned)     -> q lookups
///   bits == 8q + r, r in (0, 8)   -> q + 2 lookups
pub(crate) fn gen_witness_rangecheck_bits(
    b: &mut impl HLEmitter,
    value: ValueId,
    bits: usize,
    flag: ValueId,
) {
    assert!(bits >= 1, "rangecheck width must be at least 1 bit");

    if bits == 1 {
        let v_plain = b.value_of(value);
        let t_hint = b.mul(v_plain, v_plain);
        let t = b.write_witness(t_hint);
        b.constrain(value, value, t);
        let diff = b.sub(t, value);
        let zero = b.field_const(Field::ZERO);
        b.constrain(flag, diff, zero);
        return;
    }

    let full_bytes = bits / 8;
    let leftover_bits = bits % 8;
    let total_chunks = full_bytes + usize::from(leftover_bits > 0);

    let pure_value = b.value_of(value);
    let bytes_val = b.fresh_value();
    b.emit(OpCode::ToRadix {
        result: bytes_val,
        value: pure_value,
        radix: Radix::Bytes,
        endianness: Endianness::Big,
        count: total_chunks,
    });
    let two_to_8 = b.field_const(Field::from(256));

    let mut partial = b.field_const(Field::ZERO);
    let mut top_chunk: Option<ValueId> = None;
    for i in 0..total_chunks - 1 {
        let idx = b.u_const(32, i as u128);
        let byte = b.array_get(bytes_val, idx);
        let byte_field = b.cast_to_field(byte);
        let byte_wit = b.write_witness(byte_field);
        b.lookup_rngchk_8(byte_wit, flag);
        if i == 0 {
            top_chunk = Some(byte_wit);
        }
        let shift_prev = b.mul(partial, two_to_8);
        partial = b.add(shift_prev, byte_wit);
    }

    let partial_shifted = b.mul(partial, two_to_8);
    let lsb = b.sub(value, partial_shifted);
    b.lookup_rngchk_8(lsb, flag);
    if total_chunks == 1 {
        top_chunk = Some(lsb);
    }

    if leftover_bits > 0 {
        let top = top_chunk.expect("top_chunk set when total_chunks >= 1");
        let bound = b.field_const(Field::from((1u128 << leftover_bits) - 1));
        let gap = b.sub(bound, top);
        b.lookup_rngchk_8(gap, flag);
    }
}

/// Lower a witness-tainted Lt comparison, emitting the result into `result`.
/// Generates range-check constraints to prove the comparison.
pub(crate) fn lower_witness_lt(
    b: &mut impl HLEmitter,
    function_type_info: &FunctionTypeInfo,
    lhs: ValueId,
    rhs: ValueId,
    result: ValueId,
    l_taint: bool,
    r_taint: bool,
    l_range: &IntInterval,
    r_range: &IntInterval,
) {
    let rhs_stripped = function_type_info.get_value_type(rhs).strip_witness().expr;
    let (s, is_signed) = match rhs_stripped {
        TypeExpr::U(s) => (s, false),
        TypeExpr::I(s) => (s, true),
        _ => panic!("ICE: rhs is not an integer type"),
    };
    let u1 = CastTarget::U(1);
    assert!(
        !matches!(
            function_type_info.get_value_type(lhs).strip_witness().expr,
            TypeExpr::Field
        ),
        "ICE: lower_witness_lt got Field-typed lhs; integer-typed operands required"
    );
    assert!(
        !matches!(
            function_type_info.get_value_type(rhs).strip_witness().expr,
            TypeExpr::Field
        ),
        "ICE: lower_witness_lt got Field-typed rhs; integer-typed operands required"
    );
    let lhs_pure = if l_taint { b.value_of(lhs) } else { lhs };
    let rhs_pure = if r_taint { b.value_of(rhs) } else { rhs };
    let res_hint = b.lt(lhs_pure, rhs_pure);
    let res_hint_field = b.cast_to_field(res_hint);
    let res_witness = b.write_witness(res_hint_field);
    b.emit(OpCode::Cast {
        result,
        value: res_witness,
        target: u1,
    });

    let l_field = b.cast_to_field(lhs);
    let r_field = b.cast_to_field(rhs);
    let lr_diff = b.sub(l_field, r_field);

    let two = b.field_const(Field::from(2));
    let result_field = b.cast_to_field(result);
    let two_res = b.mul(result_field, two);
    let one = b.field_const(Field::ONE);
    let adjustment = b.sub(one, two_res);

    let lr_diff_pure = b.value_of(lr_diff);
    let adjustment_pure = b.value_of(adjustment);
    let adjusted_diff_hint = b.mul(lr_diff_pure, adjustment_pure);
    let adjusted_diff_wit = b.write_witness(adjusted_diff_hint);
    b.constrain(lr_diff, adjustment, adjusted_diff_wit);

    if is_signed {
        let always_flag = b.field_const(Field::ONE);

        let sign_a = extract_sign_bit(b, l_field, s, always_flag, l_taint, l_range);
        let sign_b = extract_sign_bit(b, r_field, s, always_flag, r_taint, r_range);

        let sa_pure = if l_taint { b.value_of(sign_a) } else { sign_a };
        let sb_pure = if r_taint { b.value_of(sign_b) } else { sign_b };
        let sa_sb_hint = b.mul(sa_pure, sb_pure);
        let sa_sb = b.write_witness(sa_sb_hint);
        b.constrain(sign_a, sign_b, sa_sb);

        let two_sa_sb = b.mul(sa_sb, two);
        let sa_plus_sb = b.add(sign_a, sign_b);
        let signs_differ = b.sub(sa_plus_sb, two_sa_sb);
        let signs_same = b.sub(one, signs_differ);

        gen_witness_rangecheck_bits(b, adjusted_diff_wit, s, signs_same);

        let zero = b.field_const(Field::ZERO);
        let diff_r_sa = b.sub(result_field, sign_a);
        b.constrain(signs_differ, diff_r_sa, zero);
    } else {
        let rc_flag = b.field_const(Field::from(1));
        gen_witness_rangecheck_bits(b, adjusted_diff_wit, s, rc_flag);
    }
}

/// Extract the sign bit (MSB) of an n-bit value.
pub(crate) fn extract_sign_bit(
    b: &mut impl HLEmitter,
    value: ValueId,
    bits: usize,
    flag: ValueId,
    is_witness: bool,
    value_range: &IntInterval,
) -> ValueId {
    if value_range.is_non_negative_in_signed(bits) {
        return b.field_const(Field::ZERO);
    }

    let two_n_minus_1 = b.field_const(Field::from(1u128 << (bits - 1)));
    let pure_val = if is_witness { b.value_of(value) } else { value };
    let low_hint = pure_low_bits_hint(b, pure_val, bits - 1);
    let high_hint = b.sub(pure_val, low_hint);
    let sign_hint = b.div(high_hint, two_n_minus_1);

    if !is_witness {
        return sign_hint;
    }

    let sign_wit = b.write_witness(sign_hint);
    gen_witness_rangecheck_bits(b, sign_wit, 1, flag);

    let sign_shifted = b.mul(sign_wit, two_n_minus_1);
    let low = b.sub(value, sign_shifted);
    gen_witness_rangecheck_bits(b, low, bits - 1, flag);

    sign_wit
}

fn pure_low_bits_hint(b: &mut impl HLEmitter, value: ValueId, bits: usize) -> ValueId {
    if bits == 0 {
        return b.field_const(Field::ZERO);
    }

    let full_bytes = bits / 8;
    let partial_bits = bits % 8;
    let byte_count = full_bytes + usize::from(partial_bits > 0);
    let bytes = b.to_radix(value, Radix::Bytes, Endianness::Big, byte_count);
    let two_to_8 = b.field_const(Field::from(256u128));
    let mut result = b.field_const(Field::ZERO);

    for i in 0..byte_count {
        let idx = b.u_const(32, i as u128);
        let mut byte = b.array_get(bytes, idx);
        if i == 0 && partial_bits > 0 {
            let mask = b.u_const(8, (1u128 << partial_bits) - 1);
            byte = b.and(byte, mask);
        }
        let byte_field = b.cast_to_field(byte);
        let shifted = b.mul(result, two_to_8);
        result = b.add(shifted, byte_field);
    }

    result
}
