//! The tests for the PRE optimisation pass.

use super::{
    edge_split::{JmpIfArm, split_jmp_if_edge},
    totality::TotalityOracle,
};
use crate::compiler::{
    Field,
    analysis::{
        click_cooper::{ClickCooper, test::run_in_test},
        flow_analysis::FlowAnalysis,
        types::{TypeInfo, Types},
    },
    passes::dead_code_elimination::{Config as DceConfig, DCE},
    ssa::{
        FunctionId, Instruction, SourceLocation, SourcePosition, Terminator,
        hlssa::{
            BinaryArithOpKind, CallTarget, CastTarget, CmpKind, Constant, Endianness, HLSSA,
            LookupTarget, OpCode, Radix, SequenceTargetType, Type,
        },
    },
};
use ark_ff::Zero;

// EDGE SPLITTING
// ================================================================================================

/// `entry: JmpIf(cond, T, F)`, with `T`/`F` returning.
fn diamond_arms() -> (
    HLSSA,
    crate::compiler::ssa::BlockId,
    crate::compiler::ssa::BlockId,
) {
    let mut ssa = HLSSA::with_main("main".to_string());
    let cond = ssa.fresh_value();
    let f = ssa.get_unique_entrypoint_mut();
    let t = f.add_block();
    let e = f.add_block();
    let entry = f.get_entry_mut();
    entry.push_parameter(cond, Type::u(1));
    entry.set_terminator(Terminator::JmpIf(cond, t, e));
    f.get_block_mut(t)
        .set_terminator(Terminator::Return(vec![]));
    f.get_block_mut(e)
        .set_terminator(Terminator::Return(vec![]));
    (ssa, t, e)
}

#[test]
fn split_true_arm_rewires_and_jumps_to_target() {
    let (mut ssa, t, e) = diamond_arms();
    let f = ssa.get_unique_entrypoint_mut();
    let entry_id = f.get_entry_id();

    let split = split_jmp_if_edge(f, entry_id, JmpIfArm::True);

    assert!(matches!(
        f.get_entry().get_terminator(),
        Some(Terminator::JmpIf(_, tt, ff)) if *tt == split && *ff == e
    ));
    assert!(matches!(
        f.get_block(split).get_terminator(),
        Some(Terminator::Jmp(target, args)) if *target == t && args.is_empty()
    ));
    assert_eq!(f.get_block(split).get_instructions().count(), 0);
}

#[test]
fn split_false_arm_leaves_true_arm_alone() {
    let (mut ssa, t, e) = diamond_arms();
    let f = ssa.get_unique_entrypoint_mut();
    let entry_id = f.get_entry_id();

    let split = split_jmp_if_edge(f, entry_id, JmpIfArm::False);

    assert!(matches!(
        f.get_entry().get_terminator(),
        Some(Terminator::JmpIf(_, tt, ff)) if *tt == t && *ff == split
    ));
    assert!(matches!(
        f.get_block(split).get_terminator(),
        Some(Terminator::Jmp(target, args)) if *target == e && args.is_empty()
    ));
}

/// Both arms of one `JmpIf` targeting the same block are individually addressable edges: each
/// split gets its own block.
#[test]
fn split_both_arms_of_shared_target() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let cond = ssa.fresh_value();
    let f = ssa.get_unique_entrypoint_mut();
    let t = f.add_block();
    let entry = f.get_entry_mut();
    entry.push_parameter(cond, Type::u(1));
    entry.set_terminator(Terminator::JmpIf(cond, t, t));
    f.get_block_mut(t)
        .set_terminator(Terminator::Return(vec![]));
    let entry_id = f.get_entry_id();

    let s_true = split_jmp_if_edge(f, entry_id, JmpIfArm::True);
    let s_false = split_jmp_if_edge(f, entry_id, JmpIfArm::False);

    assert_ne!(s_true, s_false);
    assert!(matches!(
        f.get_entry().get_terminator(),
        Some(Terminator::JmpIf(_, tt, ff)) if *tt == s_true && *ff == s_false
    ));
    for s in [s_true, s_false] {
        assert!(matches!(
            f.get_block(s).get_terminator(),
            Some(Terminator::Jmp(target, _)) if *target == t
        ));
    }
}

#[test]
#[should_panic(expected = "must end in JmpIf")]
fn split_panics_on_jmp_predecessor() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let f = ssa.get_unique_entrypoint_mut();
    let t = f.add_block();
    f.get_entry_mut().set_terminator(Terminator::Jmp(t, vec![]));
    f.get_block_mut(t)
        .set_terminator(Terminator::Return(vec![]));
    let entry_id = f.get_entry_id();

    split_jmp_if_edge(f, entry_id, JmpIfArm::True);
}

#[test]
#[should_panic(expected = "parameterized block")]
fn split_panics_on_parameterized_target() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (cond, param) = (ssa.fresh_value(), ssa.fresh_value());
    let f = ssa.get_unique_entrypoint_mut();
    let t = f.add_block();
    let entry = f.get_entry_mut();
    entry.push_parameter(cond, Type::u(1));
    // Malformed on purpose: a `JmpIf` edge cannot bind `t`'s parameter.
    entry.set_terminator(Terminator::JmpIf(cond, t, t));
    let tb = f.get_block_mut(t);
    tb.push_parameter(param, Type::field());
    tb.set_terminator(Terminator::Return(vec![]));
    let entry_id = f.get_entry_id();

    split_jmp_if_edge(f, entry_id, JmpIfArm::True);
}

// TOTALITY
// ================================================================================================

/// Analysis + typing environment over the finished `ssa`.
fn oracle_env(ssa: &HLSSA) -> (FunctionId, ClickCooper, TypeInfo) {
    let fid = ssa.get_entry_points()[0];
    let cc = run_in_test(ssa);
    let flow = FlowAnalysis::run(ssa);
    let types = Types::new().run(ssa, &flow);
    (fid, cc, types)
}

/// A `main` whose entry declares one parameter per type in `params`, terminated by a bare
/// return. Ops under test are queried hypothetically — the oracle only reads operand types — so
/// nothing needs to be pushed into the block.
fn main_with_params(params: &[Type]) -> (HLSSA, Vec<crate::compiler::ssa::ValueId>) {
    let mut ssa = HLSSA::with_main("main".to_string());
    let values: Vec<_> = params.iter().map(|_| ssa.fresh_value()).collect();
    let f = ssa.get_unique_entrypoint_mut();
    let entry = f.get_entry_mut();
    for (v, ty) in values.iter().zip(params) {
        entry.push_parameter(*v, ty.clone());
    }
    entry.set_terminator(Terminator::Return(vec![]));
    (ssa, values)
}

fn bin(
    kind: BinaryArithOpKind,
    result: crate::compiler::ssa::ValueId,
    lhs: crate::compiler::ssa::ValueId,
    rhs: crate::compiler::ssa::ValueId,
) -> OpCode {
    OpCode::BinaryArithOp {
        kind,
        result,
        lhs,
        rhs,
    }
}

#[test]
fn field_arithmetic_is_total() {
    let (ssa, vals) = main_with_params(&[
        Type::field(),
        Type::field(),
        Type::witness_of(Type::field()),
    ]);
    let (a, b, wa) = (vals[0], vals[1], vals[2]);
    let c2 = ssa.add_const(Constant::Field(Field::from(2u64)));
    let r = ssa.fresh_value();

    let (fid, cc, types) = oracle_env(&ssa);
    let oracle = TotalityOracle::new(&cc, &ssa, fid, types.get_function(fid));
    let entry = ssa.get_unique_entrypoint().get_entry_id();

    use BinaryArithOpKind::*;
    for kind in [Add, Sub, Mul] {
        assert!(oracle.is_total_at(&bin(kind, r, a, b), entry));
        // Witness-ness does not matter for field arithmetic: the gadgets are functional.
        assert!(oracle.is_total_at(&bin(kind, r, wa, wa), entry));
    }
    assert!(oracle.is_total_at(
        &OpCode::MulConst {
            result: r,
            const_val: c2,
            var: a
        },
        entry
    ));
}

#[test]
fn integer_wrap_arithmetic_is_not_total() {
    let (ssa, vals) = main_with_params(&[Type::u(32), Type::u(32), Type::witness_of(Type::u(32))]);
    let (a, b, wa) = (vals[0], vals[1], vals[2]);
    let c2 = ssa.add_const(Constant::U(32, 2));
    let r = ssa.fresh_value();

    let (fid, cc, types) = oracle_env(&ssa);
    let oracle = TotalityOracle::new(&cc, &ssa, fid, types.get_function(fid));
    let entry = ssa.get_unique_entrypoint().get_entry_id();

    use BinaryArithOpKind::*;
    for kind in [Add, Sub, Mul] {
        assert!(!oracle.is_total_at(&bin(kind, r, a, b), entry));
        assert!(!oracle.is_total_at(&bin(kind, r, wa, wa), entry));
    }
    assert!(!oracle.is_total_at(
        &OpCode::MulConst {
            result: r,
            const_val: c2,
            var: a
        },
        entry
    ));
    // Bitwise ops cannot overflow.
    for kind in [And, Or, Xor] {
        assert!(oracle.is_total_at(&bin(kind, r, a, b), entry));
    }
}

#[test]
fn comparisons_selects_and_bit_ops_are_total() {
    let (ssa, vals) = main_with_params(&[Type::u(32), Type::u(32), Type::u(1)]);
    let (a, b, cond) = (vals[0], vals[1], vals[2]);
    let r = ssa.fresh_value();

    let (fid, cc, types) = oracle_env(&ssa);
    let oracle = TotalityOracle::new(&cc, &ssa, fid, types.get_function(fid));
    let entry = ssa.get_unique_entrypoint().get_entry_id();

    for kind in [CmpKind::Eq, CmpKind::Lt] {
        assert!(oracle.is_total_at(
            &OpCode::Cmp {
                kind,
                result: r,
                lhs: a,
                rhs: b
            },
            entry
        ));
    }
    assert!(oracle.is_total_at(
        &OpCode::Not {
            result: r,
            value: cond
        },
        entry
    ));
    assert!(oracle.is_total_at(
        &OpCode::Select {
            result: r,
            cond,
            if_t: a,
            if_f: b
        },
        entry
    ));
    assert!(oracle.is_total_at(
        &OpCode::SExt {
            result: r,
            value: a,
            from_bits: 32,
            to_bits: 64
        },
        entry
    ));
    assert!(oracle.is_total_at(
        &OpCode::BitRange {
            result: r,
            value: a,
            offset: 0,
            width: 8
        },
        entry
    ));
}

#[test]
fn representation_casts_are_total_map_is_not() {
    let (ssa, vals) = main_with_params(&[Type::field(), Type::witness_of(Type::u(32))]);
    let (a, wa) = (vals[0], vals[1]);
    let r = ssa.fresh_value();

    let (fid, cc, types) = oracle_env(&ssa);
    let oracle = TotalityOracle::new(&cc, &ssa, fid, types.get_function(fid));
    let entry = ssa.get_unique_entrypoint().get_entry_id();

    let cast = |target: CastTarget, value| OpCode::Cast {
        result: r,
        value,
        target,
    };
    assert!(oracle.is_total_at(&cast(CastTarget::U(8), a), entry));
    assert!(oracle.is_total_at(&cast(CastTarget::I(16), a), entry));
    assert!(oracle.is_total_at(&cast(CastTarget::Field, a), entry));
    assert!(oracle.is_total_at(&cast(CastTarget::Nop, a), entry));
    assert!(oracle.is_total_at(&cast(CastTarget::WitnessOf, a), entry));
    assert!(oracle.is_total_at(&cast(CastTarget::ValueOf, wa), entry));
    assert!(!oracle.is_total_at(
        &cast(CastTarget::Map(Box::new(CastTarget::Field)), a),
        entry
    ));
}

#[test]
fn division_requires_a_provably_nonzero_divisor() {
    let (ssa, vals) = main_with_params(&[Type::u(32), Type::u(32)]);
    let (x, d) = (vals[0], vals[1]);
    let c0 = ssa.add_const(Constant::U(32, 0));
    let c2 = ssa.add_const(Constant::U(32, 2));
    let r = ssa.fresh_value();

    let (fid, cc, types) = oracle_env(&ssa);
    let oracle = TotalityOracle::new(&cc, &ssa, fid, types.get_function(fid));
    let entry = ssa.get_unique_entrypoint().get_entry_id();

    use BinaryArithOpKind::*;
    for kind in [Div, Mod] {
        assert!(oracle.is_total_at(&bin(kind, r, x, c2), entry));
        assert!(!oracle.is_total_at(&bin(kind, r, x, c0), entry));
        // No fact about a plain parameter divisor.
        assert!(!oracle.is_total_at(&bin(kind, r, x, d), entry));
    }
}

/// The disequality channel discharges the divisor gate exactly where the `d != 0` arm covers:
/// the false edge of an equality branch, not the entry or the equal arm.
#[test]
fn division_discharged_by_disequality_branch() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (x, d, eq) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let c0 = ssa.add_const(Constant::U(32, 0));
    let r = ssa.fresh_value();

    let f = ssa.get_unique_entrypoint_mut();
    let t = f.add_block();
    let e = f.add_block();
    let entry = f.get_entry_mut();
    entry.push_parameter(x, Type::u(32));
    entry.push_parameter(d, Type::u(32));
    entry.push_test_instruction(OpCode::Cmp {
        kind: CmpKind::Eq,
        result: eq,
        lhs: d,
        rhs: c0,
    });
    entry.set_terminator(Terminator::JmpIf(eq, t, e));
    f.get_block_mut(t)
        .set_terminator(Terminator::Return(vec![]));
    f.get_block_mut(e)
        .set_terminator(Terminator::Return(vec![]));
    let entry_id = f.get_entry_id();

    let (fid, cc, types) = oracle_env(&ssa);
    let oracle = TotalityOracle::new(&cc, &ssa, fid, types.get_function(fid));

    let div = bin(BinaryArithOpKind::Div, r, x, d);
    assert!(oracle.is_total_at(&div, e));
    assert!(!oracle.is_total_at(&div, entry_id));
    assert!(!oracle.is_total_at(&div, t));
}

/// Signed 64-bit division alone can overflow (`i64::MIN / -1` in the VM's `div_s64`): a constant
/// `-1` divisor is refused, a disequality-only fact is insufficient, and other widths/signs are
/// unaffected.
#[test]
fn signed_64_bit_division_minus_one_hazard() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (x, d, eq) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let c0 = ssa.add_const(Constant::I(64, 0));
    let c2 = ssa.add_const(Constant::I(64, 2));
    let cm1 = ssa.add_const(Constant::I(64, u64::MAX as u128));
    let r = ssa.fresh_value();

    let f = ssa.get_unique_entrypoint_mut();
    let t = f.add_block();
    let e = f.add_block();
    let entry = f.get_entry_mut();
    entry.push_parameter(x, Type::i(64));
    entry.push_parameter(d, Type::i(64));
    entry.push_test_instruction(OpCode::Cmp {
        kind: CmpKind::Eq,
        result: eq,
        lhs: d,
        rhs: c0,
    });
    entry.set_terminator(Terminator::JmpIf(eq, t, e));
    f.get_block_mut(t)
        .set_terminator(Terminator::Return(vec![]));
    f.get_block_mut(e)
        .set_terminator(Terminator::Return(vec![]));

    let (fid, cc, types) = oracle_env(&ssa);
    let oracle = TotalityOracle::new(&cc, &ssa, fid, types.get_function(fid));

    use BinaryArithOpKind::*;
    for kind in [Div, Mod] {
        assert!(oracle.is_total_at(&bin(kind, r, x, c2), e));
        assert!(!oracle.is_total_at(&bin(kind, r, x, cm1), e));
        // `d != 0` alone leaves `-1` possible at 64 signed bits.
        assert!(!oracle.is_total_at(&bin(kind, r, x, d), e));
    }
}

/// The all-ones bit pattern is only hazardous as a *signed* 64-bit divisor; as an unsigned
/// divisor it is just a large nonzero constant.
#[test]
fn unsigned_64_bit_all_ones_divisor_is_fine() {
    let (ssa, vals) = main_with_params(&[Type::u(64)]);
    let x = vals[0];
    let ones = ssa.add_const(Constant::U(64, u64::MAX as u128));
    let r = ssa.fresh_value();

    let (fid, cc, types) = oracle_env(&ssa);
    let oracle = TotalityOracle::new(&cc, &ssa, fid, types.get_function(fid));
    let entry = ssa.get_unique_entrypoint().get_entry_id();

    assert!(oracle.is_total_at(&bin(BinaryArithOpKind::Div, r, x, ones), entry));
}

#[test]
fn shifts_require_a_constant_in_range_amount() {
    let (ssa, vals) = main_with_params(&[Type::u(32), Type::u(8)]);
    let (a, dyn_amount) = (vals[0], vals[1]);
    let c5 = ssa.add_const(Constant::U(8, 5));
    let c40 = ssa.add_const(Constant::U(8, 40));
    let r = ssa.fresh_value();

    let (fid, cc, types) = oracle_env(&ssa);
    let oracle = TotalityOracle::new(&cc, &ssa, fid, types.get_function(fid));
    let entry = ssa.get_unique_entrypoint().get_entry_id();

    use BinaryArithOpKind::*;
    for kind in [Shl, Shr] {
        assert!(oracle.is_total_at(&bin(kind, r, a, c5), entry));
        assert!(!oracle.is_total_at(&bin(kind, r, a, c40), entry));
        assert!(!oracle.is_total_at(&bin(kind, r, a, dyn_amount), entry));
    }
}

#[test]
fn array_access_requires_a_constant_in_bounds_index() {
    let (ssa, vals) = main_with_params(&[
        Type::field().array_of(3),
        Type::field().slice_of(),
        Type::u(32),
        Type::field(),
    ]);
    let (arr, slice, dyn_idx, elem) = (vals[0], vals[1], vals[2], vals[3]);
    let c1 = ssa.add_const(Constant::U(32, 1));
    let c5 = ssa.add_const(Constant::U(32, 5));
    let r = ssa.fresh_value();

    let (fid, cc, types) = oracle_env(&ssa);
    let oracle = TotalityOracle::new(&cc, &ssa, fid, types.get_function(fid));
    let entry = ssa.get_unique_entrypoint().get_entry_id();

    let get = |array, index| OpCode::ArrayGet {
        result: r,
        array,
        index,
    };
    assert!(oracle.is_total_at(&get(arr, c1), entry));
    assert!(!oracle.is_total_at(&get(arr, c5), entry));
    assert!(!oracle.is_total_at(&get(arr, dyn_idx), entry));
    // Slices have no static length.
    assert!(!oracle.is_total_at(&get(slice, c1), entry));
    assert!(oracle.is_total_at(
        &OpCode::ArraySet {
            result: r,
            array: arr,
            index: c1,
            value: elem
        },
        entry
    ));
    assert!(!oracle.is_total_at(
        &OpCode::ArraySet {
            result: r,
            array: arr,
            index: c5,
            value: elem
        },
        entry
    ));
}

/// Divisions with any witness-typed operand are never speculated, even with a provably nonzero
/// divisor: the gadget lowering is constraint-emitting.
#[test]
fn witness_typed_division_is_never_total() {
    let (ssa, vals) = main_with_params(&[
        Type::witness_of(Type::field()),
        Type::field(),
        Type::witness_of(Type::u(32)),
    ]);
    let (wx, y, wd) = (vals[0], vals[1], vals[2]);
    let c2f = ssa.add_const(Constant::Field(Field::from(2u64)));
    let r = ssa.fresh_value();

    let (fid, cc, types) = oracle_env(&ssa);
    let oracle = TotalityOracle::new(&cc, &ssa, fid, types.get_function(fid));
    let entry = ssa.get_unique_entrypoint().get_entry_id();

    use BinaryArithOpKind::*;
    // Witness dividend, constant divisor.
    assert!(!oracle.is_total_at(&bin(Div, r, wx, c2f), entry));
    // Pure dividend, witness divisor.
    assert!(!oracle.is_total_at(&bin(Div, r, y, wd), entry));
}

/// A pure `Field` division still needs the divisor gate: the witgen VM defines `x / 0 = 0`, but
/// the R1CS generator's constant evaluation panics on a zero denominator.
#[test]
fn field_division_uses_the_same_divisor_gate() {
    let (ssa, vals) = main_with_params(&[Type::field(), Type::field()]);
    let (x, d) = (vals[0], vals[1]);
    let c0 = ssa.add_const(Constant::Field(Field::zero()));
    let c2 = ssa.add_const(Constant::Field(Field::from(2u64)));
    let r = ssa.fresh_value();

    let (fid, cc, types) = oracle_env(&ssa);
    let oracle = TotalityOracle::new(&cc, &ssa, fid, types.get_function(fid));
    let entry = ssa.get_unique_entrypoint().get_entry_id();

    assert!(oracle.is_total_at(&bin(BinaryArithOpKind::Div, r, x, c2), entry));
    assert!(!oracle.is_total_at(&bin(BinaryArithOpKind::Div, r, x, c0), entry));
    assert!(!oracle.is_total_at(&bin(BinaryArithOpKind::Div, r, x, d), entry));
}

#[test]
fn effectful_and_witness_machinery_ops_are_never_total() {
    let (ssa, vals) = main_with_params(&[Type::u(1), Type::field()]);
    let (cond, a) = (vals[0], vals[1]);
    let r = ssa.fresh_value();

    let (fid, cc, types) = oracle_env(&ssa);
    let oracle = TotalityOracle::new(&cc, &ssa, fid, types.get_function(fid));
    let entry = ssa.get_unique_entrypoint().get_entry_id();

    assert!(!oracle.is_total_at(&OpCode::Assert { value: cond }, entry));
    assert!(!oracle.is_total_at(
        &OpCode::WriteWitness {
            result: Some(r),
            value: a,
            pinned: false
        },
        entry
    ));
    assert!(!oracle.is_total_at(
        &OpCode::Rangecheck {
            value: a,
            max_bits: 32
        },
        entry
    ));
    assert!(!oracle.is_total_at(
        &OpCode::Guard {
            condition: cond,
            inner: Box::new(OpCode::Not {
                result: r,
                value: cond
            })
        },
        entry
    ));
}

// ELIMINATION
// ================================================================================================

/// Sweep-only entry: runs the elimination rewrite without the integrated DCE, so assertions can
/// observe the pre-DCE state exactly (mirrors the SCS test harness's `fold`).
fn eliminate(ssa: &mut HLSSA) {
    eliminate_with_lookup_dedup(ssa, true);
}

fn eliminate_with_lookup_dedup(ssa: &mut HLSSA, deduplicate_lookups: bool) {
    let cc = run_in_test(ssa);
    let flow = FlowAnalysis::run(ssa);
    let types = Types::new().run(ssa, &flow);

    let fids: Vec<_> = ssa.get_function_ids().collect();
    for fid in fids {
        let mut function = ssa.take_function(fid);
        super::eliminate::eliminate_function(
            &mut function,
            &cc,
            fid,
            types.get_function(fid),
            flow.get_function_cfg(fid),
            deduplicate_lookups,
        );
        ssa.put_function(fid, function);
    }
}

/// Motion entry: the elimination sweep followed by the motion stages enabled at `level`, without
/// the integrated DCE (redirected/hoisted-from definitions stay observable).
fn motion(ssa: &mut HLSSA, level: super::MotionLevel) {
    let cc = run_in_test(ssa);
    let flow = FlowAnalysis::run(ssa);
    let types = Types::new().run(ssa, &flow);
    let fids: Vec<_> = ssa.get_function_ids().collect();
    for fid in fids {
        let mut function = ssa.take_function(fid);
        let node_of = super::eliminate::eliminate_function(
            &mut function,
            &cc,
            fid,
            types.get_function(fid),
            flow.get_function_cfg(fid),
            true,
        );
        let oracle = TotalityOracle::new(&cc, ssa, fid, types.get_function(fid));
        super::insert::perform_code_motion(
            ssa,
            &cc,
            fid,
            &mut function,
            types.get_function(fid),
            flow.get_function_cfg(fid),
            &node_of,
            level,
            &oracle,
        );
        ssa.put_function(fid, function);
    }
}

/// [`motion`] at the down-safe loop-hoisting level.
fn hoist(ssa: &mut HLSSA) {
    motion(ssa, super::MotionLevel::LoopHoist);
}

/// [`motion`] at the general join-insertion level.
fn join_insert(ssa: &mut HLSSA) {
    motion(ssa, super::MotionLevel::JoinInsert);
}

/// [`motion`] at the full totality-gated speculation level.
fn speculate(ssa: &mut HLSSA) {
    motion(ssa, super::MotionLevel::Speculate);
}

/// Full-pass entry: the elimination rewrite followed by the integrated DCE, composed exactly as
/// [`PRE::run`] composes them at `EliminateOnly` (no motion stage) but without an `AnalysisStore`
/// (only tests need a store-less entry).
fn run_pass(ssa: &mut HLSSA) {
    eliminate(ssa);
    let flow = FlowAnalysis::run(ssa);
    DCE::new(DceConfig::pre_r1c()).do_run(ssa, &flow);
}

fn add(
    result: crate::compiler::ssa::ValueId,
    lhs: crate::compiler::ssa::ValueId,
    rhs: crate::compiler::ssa::ValueId,
) -> OpCode {
    bin(BinaryArithOpKind::Add, result, lhs, rhs)
}

/// The return values of `main`'s single returning block.
fn return_values(ssa: &HLSSA) -> Vec<crate::compiler::ssa::ValueId> {
    let f = ssa.get_unique_entrypoint();
    for (_, block) in f.get_blocks() {
        if let Some(Terminator::Return(vals)) = block.get_terminator() {
            return vals.clone();
        }
    }
    panic!("no returning block");
}

#[test]
fn same_block_duplicate_is_redirected() {
    let (mut ssa, vals) = main_with_params(&[Type::field(), Type::field()]);
    let (a, b) = (vals[0], vals[1]);
    let (r1, r2) = (ssa.fresh_value(), ssa.fresh_value());
    let f = ssa.get_unique_entrypoint_mut();
    let entry = f.get_entry_mut();
    entry.push_test_instruction(add(r1, a, b));
    entry.push_test_instruction(add(r2, a, b));
    entry.set_terminator(Terminator::Return(vec![r2]));

    eliminate(&mut ssa);

    assert_eq!(return_values(&ssa), vec![r1]);
}

#[test]
fn commutative_operands_match() {
    let (mut ssa, vals) = main_with_params(&[Type::field(), Type::field()]);
    let (a, b) = (vals[0], vals[1]);
    let (r1, r2) = (ssa.fresh_value(), ssa.fresh_value());
    let f = ssa.get_unique_entrypoint_mut();
    let entry = f.get_entry_mut();
    entry.push_test_instruction(add(r1, a, b));
    entry.push_test_instruction(add(r2, b, a));
    entry.set_terminator(Terminator::Return(vec![r2]));

    eliminate(&mut ssa);

    assert_eq!(return_values(&ssa), vec![r1]);
}

#[test]
fn dominated_block_reuses_dominating_definition() {
    let (mut ssa, vals) = main_with_params(&[Type::field(), Type::field()]);
    let (a, b) = (vals[0], vals[1]);
    let (r1, r2) = (ssa.fresh_value(), ssa.fresh_value());
    let f = ssa.get_unique_entrypoint_mut();
    let next = f.add_block();
    let entry = f.get_entry_mut();
    entry.push_test_instruction(add(r1, a, b));
    entry.set_terminator(Terminator::Jmp(next, vec![]));
    let next_block = f.get_block_mut(next);
    next_block.push_test_instruction(add(r2, a, b));
    next_block.set_terminator(Terminator::Return(vec![r2]));

    eliminate(&mut ssa);

    assert_eq!(return_values(&ssa), vec![r1]);
}

/// Neither sibling arm dominates the other: both occurrences must survive untouched — PRE's
/// elimination sweep never moves code (the insertion stages own that).
#[test]
fn sibling_arms_are_not_deduplicated() {
    let (mut ssa, vals) = main_with_params(&[Type::u(1), Type::field(), Type::field()]);
    let (cond, a, b) = (vals[0], vals[1], vals[2]);
    let (r1, r2, p) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let f = ssa.get_unique_entrypoint_mut();
    let t = f.add_block();
    let e = f.add_block();
    let m = f.add_block();
    f.get_entry_mut()
        .set_terminator(Terminator::JmpIf(cond, t, e));
    let tb = f.get_block_mut(t);
    tb.push_test_instruction(add(r1, a, b));
    tb.set_terminator(Terminator::Jmp(m, vec![r1]));
    let eb = f.get_block_mut(e);
    eb.push_test_instruction(add(r2, a, b));
    eb.set_terminator(Terminator::Jmp(m, vec![r2]));
    let mb = f.get_block_mut(m);
    mb.push_parameter(p, Type::field());
    mb.set_terminator(Terminator::Return(vec![p]));

    eliminate(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert!(matches!(
        f.get_block(t).get_terminator(),
        Some(Terminator::Jmp(_, args)) if *args == vec![r1]
    ));
    assert!(matches!(
        f.get_block(e).get_terminator(),
        Some(Terminator::Jmp(_, args)) if *args == vec![r2]
    ));
}

/// `(a + b) + c` and `a + (b + c)` share a flattened chain key even though the trees are not
/// structurally congruent — the CSE-interner equivalence binary value numbering cannot see.
#[test]
fn reassociated_chains_deduplicate() {
    let (mut ssa, vals) = main_with_params(&[Type::field(), Type::field(), Type::field()]);
    let (a, b, c) = (vals[0], vals[1], vals[2]);
    let (t1, t2, s1, s2) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );
    let f = ssa.get_unique_entrypoint_mut();
    let entry = f.get_entry_mut();
    entry.push_test_instruction(add(t1, a, b));
    entry.push_test_instruction(add(t2, t1, c));
    entry.push_test_instruction(add(s1, b, c));
    entry.push_test_instruction(add(s2, a, s1));
    entry.set_terminator(Terminator::Return(vec![t2, s2]));

    eliminate(&mut ssa);

    assert_eq!(return_values(&ssa), vec![t2, t2]);
}

#[test]
fn mul_const_unifies_with_mul() {
    let (mut ssa, vals) = main_with_params(&[Type::field()]);
    let a = vals[0];
    let c2 = ssa.add_const(Constant::Field(Field::from(2u64)));
    let (m1, m2) = (ssa.fresh_value(), ssa.fresh_value());
    let f = ssa.get_unique_entrypoint_mut();
    let entry = f.get_entry_mut();
    entry.push_test_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Mul,
        result: m1,
        lhs: a,
        rhs: c2,
    });
    entry.push_test_instruction(OpCode::MulConst {
        result: m2,
        const_val: c2,
        var: a,
    });
    entry.set_terminator(Terminator::Return(vec![m1, m2]));

    eliminate(&mut ssa);

    assert_eq!(return_values(&ssa), vec![m1, m1]);
}

/// Two parallel loop accumulators are optimistically congruent (AWZ); the duplicated increment in
/// the body redirects to its class leader — a match no expression hashing can find.
#[test]
fn loop_carried_congruent_increments_redirect() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c0 = ssa.add_const(Constant::U(32, 0));
    let c1 = ssa.add_const(Constant::U(32, 1));
    let c10 = ssa.add_const(Constant::U(32, 10));
    let (i, j, cond, i2, j2) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );
    let f = ssa.get_unique_entrypoint_mut();
    let header = f.add_block();
    let body = f.add_block();
    let exit = f.add_block();
    f.get_entry_mut()
        .set_terminator(Terminator::Jmp(header, vec![c0, c0]));
    let hb = f.get_block_mut(header);
    hb.push_parameter(i, Type::u(32));
    hb.push_parameter(j, Type::u(32));
    hb.push_test_instruction(OpCode::Cmp {
        kind: CmpKind::Lt,
        result: cond,
        lhs: i,
        rhs: c10,
    });
    hb.set_terminator(Terminator::JmpIf(cond, body, exit));
    let bb = f.get_block_mut(body);
    bb.push_test_instruction(add(i2, i, c1));
    bb.push_test_instruction(add(j2, j, c1));
    bb.set_terminator(Terminator::Jmp(header, vec![i2, j2]));
    f.get_block_mut(exit)
        .set_terminator(Terminator::Return(vec![i, j]));

    eliminate(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert!(matches!(
        f.get_block(body).get_terminator(),
        Some(Terminator::Jmp(_, args)) if *args == vec![i2, i2]
    ));
}

/// A dominated duplicate `Rangecheck` is dropped; a check of a different width is kept.
#[test]
fn rangecheck_duplicates_are_dropped() {
    let (mut ssa, vals) = main_with_params(&[Type::field()]);
    let v = vals[0];
    let f = ssa.get_unique_entrypoint_mut();
    let entry = f.get_entry_mut();
    entry.push_test_instruction(OpCode::Rangecheck {
        value: v,
        max_bits: 32,
    });
    entry.push_test_instruction(OpCode::Rangecheck {
        value: v,
        max_bits: 32,
    });
    entry.push_test_instruction(OpCode::Rangecheck {
        value: v,
        max_bits: 16,
    });
    entry.set_terminator(Terminator::Return(vec![]));

    eliminate(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert_eq!(f.get_entry().get_instructions().count(), 2);
}

/// Rangechecks over *congruent* (not merely identical) values share a key — stronger than the
/// expression-keyed CSE this sweep replaces.
#[test]
fn rangecheck_over_congruent_values_deduplicates() {
    let (mut ssa, vals) = main_with_params(&[Type::field(), Type::field()]);
    let (a, b) = (vals[0], vals[1]);
    let (x, y) = (ssa.fresh_value(), ssa.fresh_value());
    let f = ssa.get_unique_entrypoint_mut();
    let entry = f.get_entry_mut();
    entry.push_test_instruction(add(x, a, b));
    entry.push_test_instruction(add(y, a, b));
    entry.push_test_instruction(OpCode::Rangecheck {
        value: x,
        max_bits: 32,
    });
    entry.push_test_instruction(OpCode::Rangecheck {
        value: y,
        max_bits: 32,
    });
    entry.set_terminator(Terminator::Return(vec![x, y]));

    eliminate(&mut ssa);

    let f = ssa.get_unique_entrypoint();

    // One of the two rangechecks is gone; both adds remain for DCE.
    assert_eq!(f.get_entry().get_instructions().count(), 3);
}

#[test]
fn lookup_dedup_respects_config() {
    let build = || {
        let (mut ssa, vals) = main_with_params(&[Type::field()]);
        let v = vals[0];
        let flag = ssa.add_const(Constant::U(1, 1));
        let f = ssa.get_unique_entrypoint_mut();
        let entry = f.get_entry_mut();
        for _ in 0..2 {
            entry.push_test_instruction(OpCode::Lookup {
                target: LookupTarget::Rangecheck(8),
                args: vec![v],
                flag,
            });
        }
        entry.set_terminator(Terminator::Return(vec![]));
        ssa
    };

    let mut deduped = build();
    eliminate_with_lookup_dedup(&mut deduped, true);
    assert_eq!(
        deduped
            .get_unique_entrypoint()
            .get_entry()
            .get_instructions()
            .count(),
        1
    );

    let mut kept = build();
    eliminate_with_lookup_dedup(&mut kept, false);
    assert_eq!(
        kept.get_unique_entrypoint()
            .get_entry()
            .get_instructions()
            .count(),
        2
    );
}

/// Two unpinned `WriteWitness` with the same hint share a slot (the second's result redirects);
/// pinned writes never merge.
#[test]
fn unpinned_write_witness_shares_slot_pinned_does_not() {
    let (mut ssa, vals) = main_with_params(&[Type::field()]);
    let v = vals[0];
    let (r1, r2, p1, p2) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );
    let f = ssa.get_unique_entrypoint_mut();
    let entry = f.get_entry_mut();
    for (result, pinned) in [(r1, false), (r2, false), (p1, true), (p2, true)] {
        entry.push_test_instruction(OpCode::WriteWitness {
            result: Some(result),
            value: v,
            pinned,
        });
    }
    entry.set_terminator(Terminator::Return(vec![r1, r2, p1, p2]));

    eliminate(&mut ssa);

    assert_eq!(return_values(&ssa), vec![r1, r1, p1, p2]);
}

#[test]
fn to_bits_deduplicates() {
    let (mut ssa, vals) = main_with_params(&[Type::field()]);
    let v = vals[0];
    let (r1, r2) = (ssa.fresh_value(), ssa.fresh_value());
    let f = ssa.get_unique_entrypoint_mut();
    let entry = f.get_entry_mut();
    for result in [r1, r2] {
        entry.push_test_instruction(OpCode::ToBits {
            result,
            value: v,
            endianness: Endianness::Little,
            count: 8,
        });
    }
    entry.set_terminator(Terminator::Return(vec![r1, r2]));

    eliminate(&mut ssa);

    assert_eq!(return_values(&ssa), vec![r1, r1]);
}

/// The static `Bytes` form of `ToRadix` is keyed and dedups; the `Dyn` form carries a runtime
/// bound and is never keyed.
#[test]
fn to_radix_bytes_deduplicates_dyn_does_not() {
    let (mut ssa, vals) = main_with_params(&[Type::field(), Type::u(32)]);
    let (v, d) = (vals[0], vals[1]);
    let (r1, r2, d1, d2) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );
    let f = ssa.get_unique_entrypoint_mut();
    let entry = f.get_entry_mut();
    for result in [r1, r2] {
        entry.push_test_instruction(OpCode::ToRadix {
            result,
            value: v,
            radix: Radix::Bytes,
            endianness: Endianness::Little,
            count: 8,
        });
    }
    for result in [d1, d2] {
        entry.push_test_instruction(OpCode::ToRadix {
            result,
            value: v,
            radix: Radix::Dyn(d),
            endianness: Endianness::Little,
            count: 8,
        });
    }
    entry.set_terminator(Terminator::Return(vec![r2, d2]));

    eliminate(&mut ssa);

    assert_eq!(return_values(&ssa), vec![r1, d2]);
}

/// A duplicated witness cast and an expression over its results both dedup within one run: the
/// second cast's redirect is visible (chased) when the dependent expressions are keyed.
#[test]
fn witness_cast_and_dependent_expression_dedup_in_one_run() {
    let (mut ssa, vals) = main_with_params(&[Type::field()]);
    let a = vals[0];
    let (c1, c2, s1, s2) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );
    let f = ssa.get_unique_entrypoint_mut();
    let entry = f.get_entry_mut();
    entry.push_test_instruction(OpCode::Cast {
        result: c1,
        value: a,
        target: CastTarget::WitnessOf,
    });
    entry.push_test_instruction(OpCode::Cast {
        result: c2,
        value: a,
        target: CastTarget::WitnessOf,
    });
    entry.push_test_instruction(add(s1, c1, c1));
    entry.push_test_instruction(add(s2, c2, c2));
    entry.set_terminator(Terminator::Return(vec![c2, s2]));

    eliminate(&mut ssa);

    assert_eq!(return_values(&ssa), vec![c1, s1]);
}

/// Aggregate constructors are deliberately not deduplicated (CSE parity).
#[test]
fn aggregate_constructors_are_untouched() {
    let (mut ssa, vals) = main_with_params(&[Type::field(), Type::field()]);
    let (a, b) = (vals[0], vals[1]);
    let (r1, r2) = (ssa.fresh_value(), ssa.fresh_value());
    let f = ssa.get_unique_entrypoint_mut();
    let entry = f.get_entry_mut();
    for result in [r1, r2] {
        entry.push_test_instruction(OpCode::MkSeq {
            result,
            elems: vec![a, b],
            seq_type: SequenceTargetType::Array(2),
            elem_type: Type::field(),
        });
    }
    entry.set_terminator(Terminator::Return(vec![r1, r2]));

    eliminate(&mut ssa);

    assert_eq!(return_values(&ssa), vec![r1, r2]);
}

/// `ArrayGet` — the one aggregate projection the old CSE deduplicated — keeps that behavior.
#[test]
fn array_get_deduplicates() {
    let (mut ssa, vals) = main_with_params(&[Type::field().array_of(3)]);
    let arr = vals[0];
    let c1 = ssa.add_const(Constant::U(32, 1));
    let (g1, g2) = (ssa.fresh_value(), ssa.fresh_value());
    let f = ssa.get_unique_entrypoint_mut();
    let entry = f.get_entry_mut();
    for result in [g1, g2] {
        entry.push_test_instruction(OpCode::ArrayGet {
            result,
            array: arr,
            index: c1,
        });
    }
    entry.set_terminator(Terminator::Return(vec![g1, g2]));

    eliminate(&mut ssa);

    assert_eq!(return_values(&ssa), vec![g1, g1]);
}

#[test]
fn read_global_deduplicates() {
    let mut ssa = HLSSA::with_main("main".to_string());
    ssa.set_global_types(vec![Type::field()]);
    let (r1, r2) = (ssa.fresh_value(), ssa.fresh_value());
    let f = ssa.get_unique_entrypoint_mut();
    let entry = f.get_entry_mut();
    for result in [r1, r2] {
        entry.push_test_instruction(OpCode::ReadGlobal {
            result,
            offset: 0,
            result_type: Type::field(),
        });
    }
    entry.set_terminator(Terminator::Return(vec![r1, r2]));

    eliminate(&mut ssa);

    assert_eq!(return_values(&ssa), vec![r1, r1]);
}

/// The dedup groups are type-scoped: a dominating same-key occurrence of a *different* type neither
/// absorbs a later occurrence nor blocks it from leading its own type's group.
///
/// The shape is synthetic — `ReadGlobal` is keyed by offset alone, the one canonical key that
/// ignores a typed field — but it pins the guard the witness-wrapper asymmetry relies on: the
/// mismatched read `m` stays untouched, while the two same-typed reads below it still dedup with
/// each other.
#[test]
fn mismatched_type_leader_does_not_block_same_typed_dedup() {
    let mut ssa = HLSSA::with_main("main".to_string());
    ssa.set_global_types(vec![Type::field()]);
    let (m, r1, r2) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let f = ssa.get_unique_entrypoint_mut();
    let entry = f.get_entry_mut();
    for (result, result_type) in [(m, Type::u(32)), (r1, Type::field()), (r2, Type::field())] {
        entry.push_test_instruction(OpCode::ReadGlobal {
            result,
            offset: 0,
            result_type,
        });
    }
    entry.set_terminator(Terminator::Return(vec![m, r1, r2]));

    eliminate(&mut ssa);

    assert_eq!(return_values(&ssa), vec![m, r1, r1]);
}

/// The full pass: the redirected duplicate's defining instruction is swept by the integrated DCE.
#[test]
fn integrated_dce_sweeps_redirected_definitions() {
    let (mut ssa, vals) = main_with_params(&[Type::field(), Type::field()]);
    let (a, b) = (vals[0], vals[1]);
    let c5 = ssa.add_const(Constant::Field(Field::from(5u64)));
    let (r1, r2, eq) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let f = ssa.get_unique_entrypoint_mut();
    let entry = f.get_entry_mut();
    entry.push_test_instruction(add(r1, a, b));
    entry.push_test_instruction(add(r2, a, b));
    entry.push_test_instruction(OpCode::Cmp {
        kind: CmpKind::Eq,
        result: eq,
        lhs: r2,
        rhs: c5,
    });
    entry.push_test_instruction(OpCode::Assert { value: eq });
    entry.set_terminator(Terminator::Return(vec![]));

    run_pass(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    let instructions: Vec<_> = f.get_entry().get_instructions().collect();
    assert_eq!(instructions.len(), 3);
    assert!(matches!(
        instructions[1],
        OpCode::Cmp { lhs, .. } if *lhs == r1
    ));
}

// LOOP HOISTING
// ================================================================================================

/// A canonical hoistable loop: `main(a, b)` runs a counted loop whose body computes the invariant
/// `a * b` (kept live by a rangecheck), and the exit path demands the same value — making it
/// anticipated at the header despite the zero-trip exit.
///
/// Returns `(ssa, entry, header, body, exit)`.
fn invariant_loop(
    with_exit_occurrence: bool,
) -> (
    HLSSA,
    crate::compiler::ssa::BlockId,
    crate::compiler::ssa::BlockId,
    crate::compiler::ssa::BlockId,
    crate::compiler::ssa::BlockId,
) {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (a, b) = (ssa.fresh_value(), ssa.fresh_value());
    let c0 = ssa.add_const(Constant::U(32, 0));
    let c1 = ssa.add_const(Constant::U(32, 1));
    let c10 = ssa.add_const(Constant::U(32, 10));
    let (i, cond, inv, i2, inv2) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    let header = f.add_block();
    let body = f.add_block();
    let exit = f.add_block();
    let entry_id = f.get_entry_id();

    let entry = f.get_entry_mut();
    entry.push_parameter(a, Type::field());
    entry.push_parameter(b, Type::field());
    entry.set_terminator(Terminator::Jmp(header, vec![c0]));

    let hb = f.get_block_mut(header);
    hb.push_parameter(i, Type::u(32));
    hb.push_test_instruction(OpCode::Cmp {
        kind: CmpKind::Lt,
        result: cond,
        lhs: i,
        rhs: c10,
    });
    hb.set_terminator(Terminator::JmpIf(cond, body, exit));

    let bb = f.get_block_mut(body);
    bb.push_test_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Mul,
        result: inv,
        lhs: a,
        rhs: b,
    });
    bb.push_test_instruction(OpCode::Rangecheck {
        value: inv,
        max_bits: 32,
    });
    bb.push_test_instruction(add(i2, i, c1));
    bb.set_terminator(Terminator::Jmp(header, vec![i2]));

    let eb = f.get_block_mut(exit);
    if with_exit_occurrence {
        eb.push_test_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Mul,
            result: inv2,
            lhs: a,
            rhs: b,
        });
        eb.set_terminator(Terminator::Return(vec![inv2]));
    } else {
        eb.set_terminator(Terminator::Return(vec![]));
    }

    (ssa, entry_id, header, body, exit)
}

/// The value the single instruction of `block` defines.
fn sole_result(ssa: &HLSSA, block: crate::compiler::ssa::BlockId) -> crate::compiler::ssa::ValueId {
    let f = ssa.get_unique_entrypoint();
    let instrs: Vec<_> = f.get_block(block).get_instructions().collect();
    assert_eq!(instrs.len(), 1, "expected exactly one instruction");
    *instrs[0].get_results().next().unwrap()
}

/// The down-safe shape: the invariant is demanded on the exit path too, so it is anticipated at the
/// header and hoisted onto the entry edge; the in-loop and post-loop evaluations both redirect to
/// the hoisted definition.
#[test]
fn anticipated_invariant_is_hoisted_to_entry_edge() {
    let (mut ssa, entry, header, body, exit) = invariant_loop(true);

    hoist(&mut ssa);

    let hoisted = sole_result(&ssa, entry);
    let f = ssa.get_unique_entrypoint();
    assert!(matches!(
        f.get_block(entry).get_instructions().next().unwrap(),
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Mul,
            ..
        }
    ));
    // The in-loop rangecheck now checks the hoisted value.
    assert!(
        f.get_block(body)
            .get_instructions()
            .any(|i| matches!(i, OpCode::Rangecheck { value, .. } if *value == hoisted))
    );
    // The post-loop return uses it too (its own dead evaluation remains for DCE).
    assert!(matches!(
        f.get_block(exit).get_terminator(),
        Some(Terminator::Return(vals)) if *vals == vec![hoisted]
    ));
    let _ = header;
}

/// The while-loop guard: a body-only invariant is not anticipated at the header (the zero-trip
/// path never computes it), so the down-safe stage must leave it alone.
#[test]
fn body_only_invariant_is_not_hoisted() {
    let (mut ssa, entry, _, body, _) = invariant_loop(false);

    hoist(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert_eq!(f.get_block(entry).get_instructions().count(), 0);
    assert_eq!(f.get_block(body).get_instructions().count(), 3);
}

/// The staging lever's bottom step: at `EliminateOnly` the pass driver never invokes the motion
/// stage, so even the down-safe hoistable shape stays put (the elimination sweep still runs).
#[test]
fn hoisting_requires_the_loop_hoist_level() {
    let (mut ssa, entry, _, body, _) = invariant_loop(true);
    let cc = run_in_test(&ssa);
    let flow = FlowAnalysis::run(&ssa);
    let types = Types::new().run(&ssa, &flow);
    let pre = super::PRE::with_config(super::Config {
        motion: super::MotionLevel::EliminateOnly,
        deduplicate_lookups: true,
        dce: DceConfig::pre_r1c(),
    });

    pre.transform(&mut ssa, &cc, &types, &flow);

    let f = ssa.get_unique_entrypoint();
    assert_eq!(f.get_block(entry).get_instructions().count(), 0);
    assert_eq!(f.get_block(body).get_instructions().count(), 3);
}

/// The pre-untaint configuration's structural contract: the dedup fires, but the canonical diamond
/// geometry survives — the empty arm blocks stay (the integrated DCE preserves blocks), every
/// terminator keeps its shape and targets, and the merge grows no parameter. Untaint's linearizer
/// relies on exactly this geometry (a single jump into the merge from each branch side).
#[test]
fn pre_untaint_config_dedups_without_structural_change() {
    let (mut ssa, vals) = main_with_params(&[Type::u(1), Type::field(), Type::field()]);
    let (cond, a, b) = (vals[0], vals[1], vals[2]);
    let (r1, r2) = (ssa.fresh_value(), ssa.fresh_value());
    let f = ssa.get_unique_entrypoint_mut();
    let t = f.add_block();
    let e = f.add_block();
    let m = f.add_block();
    let entry = f.get_entry_mut();
    entry.push_test_instruction(add(r1, a, b));
    entry.set_terminator(Terminator::JmpIf(cond, t, e));
    f.get_block_mut(t)
        .set_terminator(Terminator::Jmp(m, vec![]));
    f.get_block_mut(e)
        .set_terminator(Terminator::Jmp(m, vec![]));
    let mb = f.get_block_mut(m);
    mb.push_test_instruction(add(r2, a, b));
    // The always-live anchor observing the redirect (the entrypoint's return slots are not
    // seeded live in this harness, so a `Return` operand cannot be the witness).
    mb.push_test_instruction(OpCode::Rangecheck {
        value: r2,
        max_bits: 32,
    });
    mb.set_terminator(Terminator::Return(vec![]));

    // Compose exactly as `PRE::run` does under `Config::pre_untaint`: the transform followed by
    // the integrated DCE with the config's (block-preserving) DCE configuration.
    let config = super::Config::pre_untaint();
    let cc = run_in_test(&ssa);
    let flow = FlowAnalysis::run(&ssa);
    let types = Types::new().run(&ssa, &flow);
    super::PRE::with_config(config).transform(&mut ssa, &cc, &types, &flow);
    let flow = FlowAnalysis::run(&ssa);
    DCE::new(config.dce).do_run(&mut ssa, &flow);

    // The dominated duplicate was redirected to the dominating occurrence (the rangecheck now
    // reads it) and swept by the integrated DCE...
    let f = ssa.get_unique_entrypoint();
    let m_ops: Vec<_> = f.get_block(m).get_instructions().collect();
    assert!(
        matches!(m_ops[..], [OpCode::Rangecheck { value, .. }] if *value == r1),
        "merge should hold exactly the redirected rangecheck: {m_ops:?}"
    );
    assert_eq!(f.get_entry().get_instructions().count(), 1);
    // ...while the diamond is untouched.
    assert_eq!(f.get_blocks().count(), 4);
    assert!(matches!(
        f.get_entry().get_terminator(),
        Some(Terminator::JmpIf(c, tt, ee)) if *c == cond && *tt == t && *ee == e
    ));
    for arm in [t, e] {
        let block = f.get_block(arm);
        assert_eq!(block.get_instructions().count(), 0);
        assert!(matches!(
            block.get_terminator(),
            Some(Terminator::Jmp(target, args)) if *target == m && args.is_empty()
        ));
    }
    assert_eq!(f.get_block(m).get_parameters().count(), 0);
}

/// A counted loop whose only `a * b` evaluation sits *after* it: anticipated at the header (the
/// exit path is bound to compute it), but eliminating nothing per-iteration.
///
/// Returns `(ssa, entry, exit, post)`.
fn post_loop_only_shape() -> (
    HLSSA,
    crate::compiler::ssa::BlockId,
    crate::compiler::ssa::BlockId,
    crate::compiler::ssa::ValueId,
) {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (a, b) = (ssa.fresh_value(), ssa.fresh_value());
    let c0 = ssa.add_const(Constant::U(32, 0));
    let c1 = ssa.add_const(Constant::U(32, 1));
    let c10 = ssa.add_const(Constant::U(32, 10));
    let (i, cond, i2, post) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    let header = f.add_block();
    let body = f.add_block();
    let exit = f.add_block();
    let entry_id = f.get_entry_id();

    let entry = f.get_entry_mut();
    entry.push_parameter(a, Type::field());
    entry.push_parameter(b, Type::field());
    entry.set_terminator(Terminator::Jmp(header, vec![c0]));

    let hb = f.get_block_mut(header);
    hb.push_parameter(i, Type::u(32));
    hb.push_test_instruction(OpCode::Cmp {
        kind: CmpKind::Lt,
        result: cond,
        lhs: i,
        rhs: c10,
    });
    hb.set_terminator(Terminator::JmpIf(cond, body, exit));

    let bb = f.get_block_mut(body);
    bb.push_test_instruction(add(i2, i, c1));
    bb.set_terminator(Terminator::Jmp(header, vec![i2]));

    let eb = f.get_block_mut(exit);
    eb.push_test_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Mul,
        result: post,
        lhs: a,
        rhs: b,
    });
    eb.set_terminator(Terminator::Return(vec![post]));

    (ssa, entry_id, exit, post)
}

/// Profitability: an expression computed only after the loop is anticipated at the header, but
/// hoisting it eliminates nothing per-iteration — it must stay where it is.
#[test]
fn post_loop_only_occurrence_is_not_hoisted() {
    let (mut ssa, entry, exit, post) = post_loop_only_shape();

    hoist(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert_eq!(f.get_block(entry).get_instructions().count(), 0);
    assert!(matches!(
        f.get_block(exit).get_terminator(),
        Some(Terminator::Return(vals)) if *vals == vec![post]
    ));
}

/// Binding stability: an inner-loop "invariant" whose operand chains through the enclosing
/// loop's carried accumulator is rebound to a NEW value per outer iteration — no invariance rule
/// admits `x = acc + 1` over the loop-carried parameter `acc`, so the inner-loop expression over
/// `x` must stay put. (Contrast [`outer_body_pure_operand_hoists_from_inner_loop`], where the
/// outer-body operand recomputes the SAME value per iteration and the hoist fires.)
#[test]
fn loop_carried_operand_blocks_hoist() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c0 = ssa.add_const(Constant::U(32, 0));
    let c1 = ssa.add_const(Constant::U(32, 1));
    let c10 = ssa.add_const(Constant::U(32, 10));
    let c1f = ssa.add_const(Constant::Field(Field::from(1u64)));
    let c2f = ssa.add_const(Constant::Field(Field::from(2u64)));
    let (j, acc, jc, x, k, kc, m1, k2, m2, j2) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    let oh = f.add_block();
    let obody = f.add_block();
    let ih = f.add_block();
    let ibody = f.add_block();
    let iafter = f.add_block();
    let oexit = f.add_block();

    f.get_entry_mut()
        .set_terminator(Terminator::Jmp(oh, vec![c0, c1f]));

    let ohb = f.get_block_mut(oh);
    ohb.push_parameter(j, Type::u(32));
    ohb.push_parameter(acc, Type::field());
    ohb.push_test_instruction(OpCode::Cmp {
        kind: CmpKind::Lt,
        result: jc,
        lhs: j,
        rhs: c10,
    });
    ohb.set_terminator(Terminator::JmpIf(jc, obody, oexit));

    // `x` is rebound every outer iteration.
    let obb = f.get_block_mut(obody);
    obb.push_test_instruction(add(x, acc, c1f));
    obb.set_terminator(Terminator::Jmp(ih, vec![c0]));

    let ihb = f.get_block_mut(ih);
    ihb.push_parameter(k, Type::u(32));
    ihb.push_test_instruction(OpCode::Cmp {
        kind: CmpKind::Lt,
        result: kc,
        lhs: k,
        rhs: c10,
    });
    ihb.set_terminator(Terminator::JmpIf(kc, ibody, iafter));

    let ibb = f.get_block_mut(ibody);
    ibb.push_test_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Mul,
        result: m1,
        lhs: x,
        rhs: c2f,
    });
    ibb.push_test_instruction(OpCode::Rangecheck {
        value: m1,
        max_bits: 32,
    });
    ibb.push_test_instruction(add(k2, k, c1));
    ibb.set_terminator(Terminator::Jmp(ih, vec![k2]));

    let iab = f.get_block_mut(iafter);
    iab.push_test_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Mul,
        result: m2,
        lhs: x,
        rhs: c2f,
    });
    iab.push_test_instruction(OpCode::Rangecheck {
        value: m2,
        max_bits: 32,
    });
    iab.push_test_instruction(add(j2, j, c1));
    iab.set_terminator(Terminator::Jmp(oh, vec![j2, x]));

    f.get_block_mut(oexit)
        .set_terminator(Terminator::Return(vec![]));

    hoist(&mut ssa);

    // The inner preheader (the outer body) still holds exactly its `Add`; nothing was hoisted.
    let f = ssa.get_unique_entrypoint();
    assert_eq!(f.get_block(obody).get_instructions().count(), 1);
}

/// The nested-loop widening shape: `x` is defined in the OUTER loop's body by a pure op over the
/// entry parameters — recomputed per outer iteration but rebinding the same value (invariance rule
/// 3) — and the inner loop evaluates `m1 = x * 2`. `down_safe` adds the inner exit path's
/// re-evaluation that makes the key anticipated at the inner header; `wrapping` builds the `U(32)`
/// twin whose key op the totality oracle refuses (and which, lacking a `Field` result, carries no
/// rangecheck — `Types` refuses integer rangechecks). The twin's `m1` being dead cannot mask that
/// refusal: the hoist rule's `in_loop`/`deeper` gates are deliberately liveness-blind (only the
/// join rule consults the `used` set), so totality is provably the sole gate standing.
fn outer_body_operand_shape(
    down_safe: bool,
    wrapping: bool,
) -> (
    HLSSA,
    crate::compiler::ssa::BlockId,
    crate::compiler::ssa::BlockId,
    crate::compiler::ssa::BlockId,
) {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (a, b) = (ssa.fresh_value(), ssa.fresh_value());
    let c0 = ssa.add_const(Constant::U(32, 0));
    let c1 = ssa.add_const(Constant::U(32, 1));
    let c10 = ssa.add_const(Constant::U(32, 10));
    let c2 = if wrapping {
        ssa.add_const(Constant::U(32, 2))
    } else {
        ssa.add_const(Constant::Field(Field::from(2u64)))
    };
    let (j, jc, x, k, kc, m1, k2, m2, j2) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );
    let operand_type = if wrapping { Type::u(32) } else { Type::field() };

    let f = ssa.get_unique_entrypoint_mut();
    let oh = f.add_block();
    let obody = f.add_block();
    let ih = f.add_block();
    let ibody = f.add_block();
    let iafter = f.add_block();
    let oexit = f.add_block();

    let entry = f.get_entry_mut();
    entry.push_parameter(a, operand_type.clone());
    entry.push_parameter(b, operand_type);
    entry.set_terminator(Terminator::Jmp(oh, vec![c0]));

    let ohb = f.get_block_mut(oh);
    ohb.push_parameter(j, Type::u(32));
    ohb.push_test_instruction(OpCode::Cmp {
        kind: CmpKind::Lt,
        result: jc,
        lhs: j,
        rhs: c10,
    });
    ohb.set_terminator(Terminator::JmpIf(jc, obody, oexit));

    // `x` is recomputed every outer iteration, always to the same value.
    let obb = f.get_block_mut(obody);
    obb.push_test_instruction(mul(x, a, b));
    obb.set_terminator(Terminator::Jmp(ih, vec![c0]));

    let ihb = f.get_block_mut(ih);
    ihb.push_parameter(k, Type::u(32));
    ihb.push_test_instruction(OpCode::Cmp {
        kind: CmpKind::Lt,
        result: kc,
        lhs: k,
        rhs: c10,
    });
    ihb.set_terminator(Terminator::JmpIf(kc, ibody, iafter));

    let ibb = f.get_block_mut(ibody);
    ibb.push_test_instruction(mul(m1, x, c2));
    if !wrapping {
        ibb.push_test_instruction(OpCode::Rangecheck {
            value: m1,
            max_bits: 32,
        });
    }
    ibb.push_test_instruction(add(k2, k, c1));
    ibb.set_terminator(Terminator::Jmp(ih, vec![k2]));

    let iab = f.get_block_mut(iafter);
    if down_safe {
        iab.push_test_instruction(mul(m2, x, c2));
        iab.push_test_instruction(OpCode::Rangecheck {
            value: m2,
            max_bits: 32,
        });
    }
    iab.push_test_instruction(add(j2, j, c1));
    iab.set_terminator(Terminator::Jmp(oh, vec![j2]));

    f.get_block_mut(oexit)
        .set_terminator(Terminator::Return(vec![]));

    (ssa, obody, ibody, iafter)
}

/// The widening the invariance closure buys: the outer-body `x = a * b` is single-valued per
/// invocation (rule 3 over entry parameters), so the inner-loop `x * 2` is binding-stable and its
/// down-safe hoist lands in the inner preheader — the outer body — where it runs once per OUTER
/// iteration instead of once per inner one. The old acyclic-definition gate refused this shape.
#[test]
fn outer_body_pure_operand_hoists_from_inner_loop() {
    let (mut ssa, obody, ibody, iafter) = outer_body_operand_shape(true, false);

    hoist(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    let instrs: Vec<_> = f.get_block(obody).get_instructions().collect();
    assert_eq!(instrs.len(), 2);
    assert!(matches!(
        instrs[1],
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Mul,
            ..
        }
    ));
    let hoisted = *instrs[1].get_results().next().unwrap();
    // Both the per-iteration and the exit-path rangechecks now check the hoisted value.
    for block in [ibody, iafter] {
        assert!(
            f.get_block(block)
                .get_instructions()
                .any(|i| matches!(i, OpCode::Rangecheck { value, .. } if *value == hoisted))
        );
    }
}

/// Zero-trip down-safety is orthogonal to the widening: without the inner exit path demanding the
/// key, the body-only occurrence over the stable outer-body operand may not hoist down-safely.
#[test]
fn outer_body_operand_body_only_is_not_hoisted() {
    let (mut ssa, obody, _, _) = outer_body_operand_shape(false, false);

    hoist(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert_eq!(f.get_block(obody).get_instructions().count(), 1);
}

/// ...but the speculation level moves it: the `Field` `Mul` is total everywhere, and the occurrence
/// sits at a strictly greater loop depth than the insertion point.
#[test]
fn outer_body_operand_body_only_is_speculated() {
    let (mut ssa, obody, ibody, _) = outer_body_operand_shape(false, false);

    speculate(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    let instrs: Vec<_> = f.get_block(obody).get_instructions().collect();
    assert_eq!(instrs.len(), 2);
    let hoisted = *instrs[1].get_results().next().unwrap();
    assert!(
        f.get_block(ibody)
            .get_instructions()
            .any(|i| matches!(i, OpCode::Rangecheck { value, .. } if *value == hoisted))
    );
}

/// The totality license still gates every newly admitted shape: the `U(32)` twin's `Mul` wraps
/// rather than traps, but integer overflow is Noir-semantically an error, so the body-only
/// occurrence stays put even at the speculation level.
#[test]
fn outer_body_operand_wrapping_op_is_not_speculated() {
    let (mut ssa, obody, _, _) = outer_body_operand_shape(false, true);

    speculate(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert_eq!(f.get_block(obody).get_instructions().count(), 1);
}

/// The nested-loop call-result shape: the outer-body operand `x` is the result of a static call to
/// a pure callee over an entry parameter, and the inner loop evaluates `x * 2` down-safely (the
/// inner exit path re-evaluates it). `unconstrained` flips the call's constrained-ness: a
/// constrained call to the deterministic callee is invariance rule 3's det-call form, while an
/// unconstrained call is nondeterministic advice the determinism summaries never bless.
fn call_result_operand_shape(
    unconstrained: bool,
) -> (
    HLSSA,
    crate::compiler::ssa::BlockId,
    crate::compiler::ssa::BlockId,
    crate::compiler::ssa::BlockId,
) {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let g = ssa.add_function("g".to_string());
    let (p, r) = (ssa.fresh_value(), ssa.fresh_value());
    let (a, x) = (ssa.fresh_value(), ssa.fresh_value());
    let c0 = ssa.add_const(Constant::U(32, 0));
    let c1 = ssa.add_const(Constant::U(32, 1));
    let c10 = ssa.add_const(Constant::U(32, 10));
    let c2f = ssa.add_const(Constant::Field(Field::from(2u64)));
    let (j, jc, k, kc, m1, k2, m2, j2) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    {
        let gf = ssa.get_function_mut(g);
        gf.add_return_type(Type::field());
        gf.get_entry_mut().push_parameter(p, Type::field());
        gf.get_entry_mut().push_test_instruction(add(r, p, p));
        gf.get_entry_mut()
            .set_terminator(Terminator::Return(vec![r]));
    }

    let f = ssa.get_function_mut(main_id);
    let oh = f.add_block();
    let obody = f.add_block();
    let ih = f.add_block();
    let ibody = f.add_block();
    let iafter = f.add_block();
    let oexit = f.add_block();

    let entry = f.get_entry_mut();
    entry.push_parameter(a, Type::field());
    entry.set_terminator(Terminator::Jmp(oh, vec![c0]));

    let ohb = f.get_block_mut(oh);
    ohb.push_parameter(j, Type::u(32));
    ohb.push_test_instruction(OpCode::Cmp {
        kind: CmpKind::Lt,
        result: jc,
        lhs: j,
        rhs: c10,
    });
    ohb.set_terminator(Terminator::JmpIf(jc, obody, oexit));

    let obb = f.get_block_mut(obody);
    obb.push_test_instruction(OpCode::Call {
        results: vec![x],
        function: CallTarget::Static(g),
        args: vec![a],
        unconstrained,
    });
    obb.set_terminator(Terminator::Jmp(ih, vec![c0]));

    let ihb = f.get_block_mut(ih);
    ihb.push_parameter(k, Type::u(32));
    ihb.push_test_instruction(OpCode::Cmp {
        kind: CmpKind::Lt,
        result: kc,
        lhs: k,
        rhs: c10,
    });
    ihb.set_terminator(Terminator::JmpIf(kc, ibody, iafter));

    let ibb = f.get_block_mut(ibody);
    ibb.push_test_instruction(mul(m1, x, c2f));
    ibb.push_test_instruction(OpCode::Rangecheck {
        value: m1,
        max_bits: 32,
    });
    ibb.push_test_instruction(add(k2, k, c1));
    ibb.set_terminator(Terminator::Jmp(ih, vec![k2]));

    let iab = f.get_block_mut(iafter);
    iab.push_test_instruction(mul(m2, x, c2f));
    iab.push_test_instruction(OpCode::Rangecheck {
        value: m2,
        max_bits: 32,
    });
    iab.push_test_instruction(add(j2, j, c1));
    iab.set_terminator(Terminator::Jmp(oh, vec![j2]));

    f.get_block_mut(oexit)
        .set_terminator(Terminator::Return(vec![]));

    (ssa, obody, ibody, iafter)
}

/// Invariance rule 3's det-call form: the outer-body operand is a constrained static call to a
/// deterministic callee over entry parameters — single-valued per invocation even though the call
/// itself can never move — so the inner-loop expression over its result hoists.
#[test]
fn det_call_result_operand_hoists_from_inner_loop() {
    let (mut ssa, obody, ibody, iafter) = call_result_operand_shape(false);

    hoist(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    let instrs: Vec<_> = f.get_block(obody).get_instructions().collect();
    assert_eq!(instrs.len(), 2);
    assert!(matches!(instrs[0], OpCode::Call { .. }));
    let hoisted = *instrs[1].get_results().next().unwrap();
    for block in [ibody, iafter] {
        assert!(
            f.get_block(block)
                .get_instructions()
                .any(|i| matches!(i, OpCode::Rangecheck { value, .. } if *value == hoisted))
        );
    }
}

/// The negative twin: an `unconstrained` call is nondeterministic advice — its results are pinned
/// to no function of their arguments, and rule 3's det-call form requires a *constrained* static
/// call — so `x` stays unstable and the inner-loop `x * 2` must stay put. If the gate's determinism
/// read regressed to blessing every call, this hoist would fire.
#[test]
fn nondet_call_result_operand_blocks_hoist() {
    let (mut ssa, obody, ibody, _) = call_result_operand_shape(true);

    hoist(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    // The outer body holds exactly the call; nothing was hoisted into it.
    let instrs: Vec<_> = f.get_block(obody).get_instructions().collect();
    assert_eq!(instrs.len(), 1);
    assert!(matches!(instrs[0], OpCode::Call { .. }));
    // The per-iteration evaluation stays in the inner body (mul + rangecheck + increment).
    assert_eq!(f.get_block(ibody).get_instructions().count(), 3);
}

/// The invariance closure's rule-4 (congruence-tier) form, isolated: the inner-loop key's array
/// operand is the OUTER header's parameter `arr = φ(arr0, arr0)`, which no other rule admits —
/// its lattice constant is an aggregate the scalar-gated rule 1 refuses, its definition sits in a
/// cyclic block (rule 2 — the old acyclic-definition gate refused this shape the same way), and a
/// parameter has no defining op for rule 3. Only the congruence tier admits it: the lattice folds
/// the φ to the same `Blob` constant as the entry-block `arr0 = MkSeq(...)`, the const-seeded
/// partition puts both in one class, and `arr0` — acyclically defined, hence rules-1–3 stable —
/// is the dominating witness. The index must be the NON-constant entry formal: a constant index
/// would let the lattice's `ArrayGet` projection fold the occurrences away entirely.
#[test]
fn aggregate_const_phi_operand_hoists_via_congruence_tier() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let idx = ssa.fresh_value();
    let c0 = ssa.add_const(Constant::U(32, 0));
    let c1 = ssa.add_const(Constant::U(32, 1));
    let c10 = ssa.add_const(Constant::U(32, 10));
    let ce1 = ssa.add_const(Constant::Field(Field::from(7u64)));
    let ce2 = ssa.add_const(Constant::Field(Field::from(9u64)));
    let (arr0, j, arr, jc, k, kc, x1, k2, x2, j2) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    let oh = f.add_block();
    let obody = f.add_block();
    let ih = f.add_block();
    let ibody = f.add_block();
    let iafter = f.add_block();
    let oexit = f.add_block();

    let entry = f.get_entry_mut();
    entry.push_parameter(idx, Type::u(32));
    entry.push_test_instruction(OpCode::MkSeq {
        result: arr0,
        elems: vec![ce1, ce2],
        seq_type: SequenceTargetType::Array(2),
        elem_type: Type::field(),
    });
    entry.set_terminator(Terminator::Jmp(oh, vec![c0, arr0]));

    let ohb = f.get_block_mut(oh);
    ohb.push_parameter(j, Type::u(32));
    ohb.push_parameter(arr, Type::field().array_of(2));
    ohb.push_test_instruction(OpCode::Cmp {
        kind: CmpKind::Lt,
        result: jc,
        lhs: j,
        rhs: c10,
    });
    ohb.set_terminator(Terminator::JmpIf(jc, obody, oexit));

    f.get_block_mut(obody)
        .set_terminator(Terminator::Jmp(ih, vec![c0]));

    let ihb = f.get_block_mut(ih);
    ihb.push_parameter(k, Type::u(32));
    ihb.push_test_instruction(OpCode::Cmp {
        kind: CmpKind::Lt,
        result: kc,
        lhs: k,
        rhs: c10,
    });
    ihb.set_terminator(Terminator::JmpIf(kc, ibody, iafter));

    let ibb = f.get_block_mut(ibody);
    ibb.push_test_instruction(OpCode::ArrayGet {
        result: x1,
        array: arr,
        index: idx,
    });
    ibb.push_test_instruction(OpCode::Rangecheck {
        value: x1,
        max_bits: 32,
    });
    ibb.push_test_instruction(add(k2, k, c1));
    ibb.set_terminator(Terminator::Jmp(ih, vec![k2]));

    let iab = f.get_block_mut(iafter);
    iab.push_test_instruction(OpCode::ArrayGet {
        result: x2,
        array: arr,
        index: idx,
    });
    iab.push_test_instruction(OpCode::Rangecheck {
        value: x2,
        max_bits: 32,
    });
    iab.push_test_instruction(add(j2, j, c1));
    iab.set_terminator(Terminator::Jmp(oh, vec![j2, arr0]));

    f.get_block_mut(oexit)
        .set_terminator(Terminator::Return(vec![]));

    hoist(&mut ssa);

    // The down-safe hoist lands in the inner preheader (the outer body), and both the
    // per-iteration and exit-path rangechecks now read the hoisted access.
    let f = ssa.get_unique_entrypoint();
    let instrs: Vec<_> = f.get_block(obody).get_instructions().collect();
    assert_eq!(instrs.len(), 1);
    assert!(matches!(instrs[0], OpCode::ArrayGet { .. }));
    let hoisted = *instrs[0].get_results().next().unwrap();
    for block in [ibody, iafter] {
        assert!(
            f.get_block(block)
                .get_instructions()
                .any(|i| matches!(i, OpCode::Rangecheck { value, .. } if *value == hoisted))
        );
    }
}

/// A parameterless self-loop reached through a `JmpIf`: the hoist must split the entry edge and
/// place the computation in the split block, leaving the branch's other arm untouched.
#[test]
fn jmp_if_entry_edge_is_split_for_the_hoist() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (c, a, b) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let inv = ssa.fresh_value();

    let f = ssa.get_unique_entrypoint_mut();
    let h = f.add_block();
    let skip = f.add_block();
    let entry_id = f.get_entry_id();

    let entry = f.get_entry_mut();
    entry.push_parameter(c, Type::u(1));
    entry.push_parameter(a, Type::field());
    entry.push_parameter(b, Type::field());
    entry.set_terminator(Terminator::JmpIf(c, h, skip));

    // A parameterless self-loop: the invariant is generated in the header itself, so it is
    // anticipated there regardless of the exit path.
    let hb = f.get_block_mut(h);
    hb.push_test_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Mul,
        result: inv,
        lhs: a,
        rhs: b,
    });
    hb.push_test_instruction(OpCode::Rangecheck {
        value: inv,
        max_bits: 32,
    });
    hb.set_terminator(Terminator::JmpIf(c, h, skip));

    f.get_block_mut(skip)
        .set_terminator(Terminator::Return(vec![]));

    hoist(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    // The entry's true arm now targets a fresh split block holding the hoisted multiply...
    let split = match f.get_block(entry_id).get_terminator() {
        Some(Terminator::JmpIf(_, t, e)) if *t != h && *e == skip => *t,
        other => panic!("entry terminator not rewired: {other:?}"),
    };
    let hoisted = sole_result(&ssa, split);
    assert!(matches!(
        f.get_block(split).get_terminator(),
        Some(Terminator::Jmp(target, args)) if *target == h && args.is_empty()
    ));
    // ...and the in-loop rangecheck consumes the hoisted value.
    assert!(
        f.get_block(h)
            .get_instructions()
            .any(|i| matches!(i, OpCode::Rangecheck { value, .. } if *value == hoisted))
    );
}

/// A header with two entry-side predecessors is out of scope for this stage (it needs the general
/// join-insertion machinery); nothing moves.
#[test]
fn multi_entry_header_is_skipped() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (c, a, b) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let c0 = ssa.add_const(Constant::U(32, 0));
    let c1 = ssa.add_const(Constant::U(32, 1));
    let c10 = ssa.add_const(Constant::U(32, 10));
    let (i, cond, inv, i2, inv2) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    let p1 = f.add_block();
    let p2 = f.add_block();
    let header = f.add_block();
    let body = f.add_block();
    let exit = f.add_block();

    let entry = f.get_entry_mut();
    entry.push_parameter(c, Type::u(1));
    entry.push_parameter(a, Type::field());
    entry.push_parameter(b, Type::field());
    entry.set_terminator(Terminator::JmpIf(c, p1, p2));
    f.get_block_mut(p1)
        .set_terminator(Terminator::Jmp(header, vec![c0]));
    f.get_block_mut(p2)
        .set_terminator(Terminator::Jmp(header, vec![c1]));

    let hb = f.get_block_mut(header);
    hb.push_parameter(i, Type::u(32));
    hb.push_test_instruction(OpCode::Cmp {
        kind: CmpKind::Lt,
        result: cond,
        lhs: i,
        rhs: c10,
    });
    hb.set_terminator(Terminator::JmpIf(cond, body, exit));

    let bb = f.get_block_mut(body);
    bb.push_test_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Mul,
        result: inv,
        lhs: a,
        rhs: b,
    });
    bb.push_test_instruction(OpCode::Rangecheck {
        value: inv,
        max_bits: 32,
    });
    bb.push_test_instruction(add(i2, i, c1));
    bb.set_terminator(Terminator::Jmp(header, vec![i2]));

    let eb = f.get_block_mut(exit);
    eb.push_test_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Mul,
        result: inv2,
        lhs: a,
        rhs: b,
    });
    eb.set_terminator(Terminator::Return(vec![inv2]));

    hoist(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    for pred in [p1, p2] {
        assert_eq!(f.get_block(pred).get_instructions().count(), 0);
    }
}

/// Availability: a definition already dominating the header carries the value; the loop's uses were
/// redirected to it by the elimination sweep and no insertion happens.
#[test]
fn value_available_before_the_loop_is_not_rehoisted() {
    let (mut ssa, entry, _, body, _) = invariant_loop(true);
    // Prepend the same computation before the loop.
    let (a, b): (crate::compiler::ssa::ValueId, crate::compiler::ssa::ValueId) = {
        let f = ssa.get_unique_entrypoint();
        let mut params = f.get_entry().get_parameter_values().copied();
        (params.next().unwrap(), params.next().unwrap())
    };
    let pre = ssa.fresh_value();
    {
        let f = ssa.get_unique_entrypoint_mut();
        let eb = f.get_block_mut(entry);
        let mut instructions = eb.take_instructions();
        instructions.insert(
            0,
            crate::compiler::ssa::Located::new(
                OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::Mul,
                    result: pre,
                    lhs: a,
                    rhs: b,
                },
                SourceLocation::test(),
            ),
        );
        eb.put_instructions(instructions);
    }

    hoist(&mut ssa);

    let f = ssa.get_unique_entrypoint();

    // Entry still holds exactly the one pre-existing multiply; the rangecheck uses it.
    assert_eq!(f.get_block(entry).get_instructions().count(), 1);
    assert!(
        f.get_block(body)
            .get_instructions()
            .any(|i| matches!(i, OpCode::Rangecheck { value, .. } if *value == pre))
    );
}

// JOIN INSERTION
// ================================================================================================

/// `entry(a, b, c): JmpIf(c, left, right)`, both arms jumping to `merge` (which returns nothing
/// yet); the caller fills the arms and the merge.
fn diamond() -> (
    HLSSA,
    crate::compiler::ssa::ValueId,
    crate::compiler::ssa::ValueId,
    crate::compiler::ssa::BlockId,
    crate::compiler::ssa::BlockId,
    crate::compiler::ssa::BlockId,
) {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (a, b, c) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let f = ssa.get_unique_entrypoint_mut();
    let left = f.add_block();
    let right = f.add_block();
    let merge = f.add_block();
    let entry = f.get_entry_mut();
    entry.push_parameter(a, Type::field());
    entry.push_parameter(b, Type::field());
    entry.push_parameter(c, Type::u(1));
    entry.set_terminator(Terminator::JmpIf(c, left, right));
    f.get_block_mut(left)
        .set_terminator(Terminator::Jmp(merge, vec![]));
    f.get_block_mut(right)
        .set_terminator(Terminator::Jmp(merge, vec![]));
    f.get_block_mut(merge)
        .set_terminator(Terminator::Return(vec![]));
    (ssa, a, b, left, right, merge)
}

/// Extend `merge` with `merge: JmpIf(cond, x, y)` where each successor recomputes `a + b` and
/// returns it: two live sibling occurrences the merge dominates (siblings, so the elimination sweep
/// cannot pre-merge them), enough to pass the static-cost gate against one copy.
///
/// Returns `(x, y, t_x, t_y)`.
fn sibling_recomputations(
    ssa: &mut HLSSA,
    merge: crate::compiler::ssa::BlockId,
    cond: crate::compiler::ssa::ValueId,
    a: crate::compiler::ssa::ValueId,
    b: crate::compiler::ssa::ValueId,
    extra_return: Option<crate::compiler::ssa::ValueId>,
) -> (
    crate::compiler::ssa::BlockId,
    crate::compiler::ssa::BlockId,
    crate::compiler::ssa::ValueId,
    crate::compiler::ssa::ValueId,
) {
    let (tx, ty) = (ssa.fresh_value(), ssa.fresh_value());
    let f = ssa.get_unique_entrypoint_mut();
    let x = f.add_block();
    let y = f.add_block();
    f.get_block_mut(merge)
        .set_terminator(Terminator::JmpIf(cond, x, y));
    for (block, t) in [(x, tx), (y, ty)] {
        let bb = f.get_block_mut(block);
        bb.push_test_instruction(add(t, a, b));
        let mut rets = vec![t];
        rets.extend(extra_return);
        bb.set_terminator(Terminator::Return(rets));
    }
    (x, y, tx, ty)
}

/// The profitable diamond: one arm computes the key, and both continuations of the merge recompute
/// it. The join drains the two sibling evaluations through one fresh parameter, paying a single
/// copy materialized on the bare edge (2 eliminated > 1 materialized).
#[test]
fn diamond_partial_redundancy_joins_through_parameter() {
    let (mut ssa, a, b, left, right, merge) = diamond();
    let t1 = ssa.fresh_value();
    let c = {
        let f = ssa.get_unique_entrypoint();
        f.get_entry()
            .get_parameter_values()
            .copied()
            .nth(2)
            .unwrap()
    };
    ssa.get_unique_entrypoint_mut()
        .get_block_mut(left)
        .push_test_instruction(add(t1, a, b));
    let (x, y, _tx, _ty) = sibling_recomputations(&mut ssa, merge, c, a, b, None);

    join_insert(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    let params: Vec<_> = f.get_block(merge).get_parameter_values().copied().collect();
    assert_eq!(params.len(), 1);
    // The computing arm passes its own evaluation...
    assert!(matches!(
        f.get_block(left).get_terminator(),
        Some(Terminator::Jmp(_, args)) if *args == vec![t1]
    ));
    // ...the bare arm materialized a copy on its edge...
    let copies: Vec<_> = f.get_block(right).get_instructions().collect();
    assert_eq!(copies.len(), 1);
    let copy = *copies[0].get_results().next().unwrap();
    assert!(matches!(
        f.get_block(right).get_terminator(),
        Some(Terminator::Jmp(_, args)) if *args == vec![copy]
    ));
    // ...and both sibling recomputations are redirected to the parameter.
    for block in [x, y] {
        assert!(matches!(
            f.get_block(block).get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![params[0]]
        ));
    }
}

/// The static-cost gate: the instruction-neutral diamond — one computing arm, one recomputation at
/// the merge — trades one op for one op plus a parameter and its argument moves, so it is refused.
#[test]
fn instruction_neutral_diamond_is_refused() {
    let (mut ssa, a, b, left, right, merge) = diamond();
    let (t1, t2) = (ssa.fresh_value(), ssa.fresh_value());
    let f = ssa.get_unique_entrypoint_mut();
    f.get_block_mut(left).push_test_instruction(add(t1, a, b));
    let mb = f.get_block_mut(merge);
    mb.push_test_instruction(add(t2, a, b));
    mb.set_terminator(Terminator::Return(vec![t2]));

    join_insert(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert_eq!(f.get_block(merge).get_parameter_values().count(), 0);
    assert_eq!(f.get_block(right).get_instructions().count(), 0);
    assert_eq!(return_values(&ssa), vec![t2]);
}

/// Both arms compute the key: the parameter joins the branch-local evaluations without a single new
/// instruction — the cross-branch shape CSE's dominator rule cannot reach.
#[test]
fn cross_branch_full_availability_joins_without_copies() {
    let (mut ssa, a, b, left, right, merge) = diamond();
    let (t1, t2, t3) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let f = ssa.get_unique_entrypoint_mut();
    f.get_block_mut(left).push_test_instruction(add(t1, a, b));
    f.get_block_mut(right).push_test_instruction(add(t2, a, b));
    let mb = f.get_block_mut(merge);
    mb.push_test_instruction(add(t3, a, b));
    mb.set_terminator(Terminator::Return(vec![t3]));

    join_insert(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert_eq!(f.get_block(left).get_instructions().count(), 1);
    assert_eq!(f.get_block(right).get_instructions().count(), 1);
    let params: Vec<_> = f.get_block(merge).get_parameter_values().copied().collect();
    assert_eq!(params.len(), 1);
    assert!(matches!(
        f.get_block(left).get_terminator(),
        Some(Terminator::Jmp(_, args)) if *args == vec![t1]
    ));
    assert!(matches!(
        f.get_block(right).get_terminator(),
        Some(Terminator::Jmp(_, args)) if *args == vec![t2]
    ));
    assert_eq!(return_values(&ssa), vec![params[0]]);
}

/// A key available on no incoming edge is refused: joining it would compute on every path while
/// eliminating on none.
#[test]
fn key_available_on_no_edge_is_refused() {
    let (mut ssa, a, b, _left, _right, merge) = diamond();
    let t = ssa.fresh_value();
    let f = ssa.get_unique_entrypoint_mut();
    let mb = f.get_block_mut(merge);
    mb.push_test_instruction(add(t, a, b));
    mb.set_terminator(Terminator::Return(vec![t]));

    join_insert(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert_eq!(f.get_block(merge).get_parameter_values().count(), 0);
    assert_eq!(return_values(&ssa), vec![t]);
}

/// Profitability against path growth: the key is anticipated at the inner merge (the join below
/// evaluates it on every continuation) and available on one edge, but its only later occurrence
/// sits at a block the merge does not dominate — redirecting nothing, the insertion would only
/// lengthen the paths entering through the bare edge, so it must be refused.
#[test]
fn merge_without_dominated_occurrence_is_refused() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (a, b, c) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let (t1, t2) = (ssa.fresh_value(), ssa.fresh_value());
    let f = ssa.get_unique_entrypoint_mut();
    let pre = f.add_block();
    let w = f.add_block();
    let left = f.add_block();
    let right = f.add_block();
    let m = f.add_block();
    let x = f.add_block();

    let entry = f.get_entry_mut();
    entry.push_parameter(a, Type::field());
    entry.push_parameter(b, Type::field());
    entry.push_parameter(c, Type::u(1));
    entry.set_terminator(Terminator::JmpIf(c, pre, w));
    f.get_block_mut(pre)
        .set_terminator(Terminator::JmpIf(c, left, right));
    let lb = f.get_block_mut(left);
    lb.push_test_instruction(add(t1, a, b));
    lb.push_test_instruction(OpCode::Rangecheck {
        value: t1,
        max_bits: 32,
    });
    lb.set_terminator(Terminator::Jmp(m, vec![]));
    f.get_block_mut(right)
        .set_terminator(Terminator::Jmp(m, vec![]));
    f.get_block_mut(m)
        .set_terminator(Terminator::Jmp(x, vec![]));
    f.get_block_mut(w)
        .set_terminator(Terminator::Jmp(x, vec![]));
    let xb = f.get_block_mut(x);
    xb.push_test_instruction(add(t2, a, b));
    xb.set_terminator(Terminator::Return(vec![t2]));

    join_insert(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    for block in [m, x] {
        assert_eq!(f.get_block(block).get_parameter_values().count(), 0);
    }
    assert_eq!(return_values(&ssa), vec![t2]);
}

/// A branching predecessor cannot carry a jump argument: its edge into the merge is split, and the
/// split block hosts both the materialized copy and the argument.
#[test]
fn branching_predecessor_edge_is_split_to_carry_the_argument() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (a, b, c) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let t1 = ssa.fresh_value();
    let f = ssa.get_unique_entrypoint_mut();
    let p1 = f.add_block();
    let p2 = f.add_block();
    let m = f.add_block();
    let z = f.add_block();

    let entry = f.get_entry_mut();
    entry.push_parameter(a, Type::field());
    entry.push_parameter(b, Type::field());
    entry.push_parameter(c, Type::u(1));
    entry.set_terminator(Terminator::JmpIf(c, p1, p2));
    let p1b = f.get_block_mut(p1);
    p1b.push_test_instruction(add(t1, a, b));
    p1b.set_terminator(Terminator::Jmp(m, vec![]));
    f.get_block_mut(p2)
        .set_terminator(Terminator::JmpIf(c, m, z));
    f.get_block_mut(z)
        .set_terminator(Terminator::Return(vec![]));
    let (x, y, _tx, _ty) = sibling_recomputations(&mut ssa, m, c, a, b, None);

    join_insert(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    let Some(Terminator::JmpIf(_, s, other)) = f.get_block(p2).get_terminator() else {
        panic!("p2 must still branch");
    };
    let (split, other) = (*s, *other);
    assert_ne!(split, m);
    assert_eq!(other, z);
    let copies: Vec<_> = f.get_block(split).get_instructions().collect();
    assert_eq!(copies.len(), 1);
    let copy = *copies[0].get_results().next().unwrap();
    assert!(matches!(
        f.get_block(split).get_terminator(),
        Some(Terminator::Jmp(t, args)) if *t == m && *args == vec![copy]
    ));
    let params: Vec<_> = f.get_block(m).get_parameter_values().copied().collect();
    assert_eq!(params.len(), 1);
    for block in [x, y] {
        assert!(matches!(
            f.get_block(block).get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![params[0]]
        ));
    }
}

/// One predecessor enters the merge through *both* arms of its `JmpIf`: the wiring splits both
/// arms, the single materialized copy lands at that predecessor's own end (its exit is the edge,
/// dominating both splits), and both split jumps carry the copy as their argument.
#[test]
fn both_arms_predecessor_hosts_one_copy_and_two_arguments() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (a, b, c) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let t1 = ssa.fresh_value();
    let f = ssa.get_unique_entrypoint_mut();
    let left = f.add_block();
    let bi = f.add_block();
    let m = f.add_block();

    let entry = f.get_entry_mut();
    entry.push_parameter(a, Type::field());
    entry.push_parameter(b, Type::field());
    entry.push_parameter(c, Type::u(1));
    entry.set_terminator(Terminator::JmpIf(c, left, bi));
    let lb = f.get_block_mut(left);
    lb.push_test_instruction(add(t1, a, b));
    lb.set_terminator(Terminator::Jmp(m, vec![]));
    // The degenerate branch: both arms enter the merge, two distinct edges from one predecessor.
    f.get_block_mut(bi)
        .set_terminator(Terminator::JmpIf(c, m, m));
    f.get_block_mut(m)
        .set_terminator(Terminator::Return(vec![]));
    let (x, y, _tx, _ty) = sibling_recomputations(&mut ssa, m, c, a, b, None);

    join_insert(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    let params: Vec<_> = f.get_block(m).get_parameter_values().copied().collect();
    assert_eq!(params.len(), 1);
    // The copy lives once, at the both-arms predecessor's own end...
    let copies: Vec<_> = f.get_block(bi).get_instructions().collect();
    assert_eq!(copies.len(), 1);
    let copy = *copies[0].get_results().next().unwrap();
    // ...and each arm was split into its own argument-carrying block.
    let Some(Terminator::JmpIf(_, s_true, s_false)) = f.get_block(bi).get_terminator() else {
        panic!("the both-arms predecessor must still branch");
    };
    let (s_true, s_false) = (*s_true, *s_false);
    assert_ne!(s_true, s_false);
    for split in [s_true, s_false] {
        assert_ne!(split, m);
        assert!(matches!(
            f.get_block(split).get_terminator(),
            Some(Terminator::Jmp(t, args)) if *t == m && *args == vec![copy]
        ));
    }
    // The computing arm passes its own evaluation, and the siblings drain into the parameter.
    assert!(matches!(
        f.get_block(left).get_terminator(),
        Some(Terminator::Jmp(_, args)) if *args == vec![t1]
    ));
    for block in [x, y] {
        assert!(matches!(
            f.get_block(block).get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![params[0]]
        ));
    }
}

/// An unreachable predecessor refuses the join outright: no leader can ever dominate it, yet any
/// planted parameter would still need an argument on its dead jump. The same shape without the
/// dead predecessor fires (see [`diamond_partial_redundancy_joins_through_parameter`]).
#[test]
fn merge_with_unreachable_predecessor_is_refused() {
    let (mut ssa, a, b, left, _right, merge) = diamond();
    let t1 = ssa.fresh_value();
    let c = {
        let f = ssa.get_unique_entrypoint();
        f.get_entry()
            .get_parameter_values()
            .copied()
            .nth(2)
            .unwrap()
    };
    let f = ssa.get_unique_entrypoint_mut();
    f.get_block_mut(left).push_test_instruction(add(t1, a, b));
    // A block no path reaches, jumping into the merge.
    let dead = f.add_block();
    f.get_block_mut(dead)
        .set_terminator(Terminator::Jmp(merge, vec![]));
    let (x, y, tx, ty) = sibling_recomputations(&mut ssa, merge, c, a, b, None);

    join_insert(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert_eq!(f.get_block(merge).get_parameter_values().count(), 0);
    assert!(matches!(
        f.get_block(dead).get_terminator(),
        Some(Terminator::Jmp(_, args)) if args.is_empty()
    ));
    for (block, t) in [(x, tx), (y, ty)] {
        assert!(matches!(
            f.get_block(block).get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![t]
        ));
    }
}

/// A merge that already carries parameters gets the new one appended, with every predecessor's
/// existing arguments left aligned in front of the new ones.
#[test]
fn existing_parameters_and_arguments_stay_aligned() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (a, b, c, q) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );
    let t1 = ssa.fresh_value();
    let f = ssa.get_unique_entrypoint_mut();
    let left = f.add_block();
    let right = f.add_block();
    let m = f.add_block();

    let entry = f.get_entry_mut();
    entry.push_parameter(a, Type::field());
    entry.push_parameter(b, Type::field());
    entry.push_parameter(c, Type::u(1));
    entry.set_terminator(Terminator::JmpIf(c, left, right));
    let lb = f.get_block_mut(left);
    lb.push_test_instruction(add(t1, a, b));
    lb.set_terminator(Terminator::Jmp(m, vec![a]));
    f.get_block_mut(right)
        .set_terminator(Terminator::Jmp(m, vec![b]));
    f.get_block_mut(m).push_parameter(q, Type::field());
    let (x, y, _tx, _ty) = sibling_recomputations(&mut ssa, m, c, a, b, Some(q));

    join_insert(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    let params: Vec<_> = f.get_block(m).get_parameter_values().copied().collect();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0], q);
    let copies: Vec<_> = f.get_block(right).get_instructions().collect();
    assert_eq!(copies.len(), 1);
    let copy = *copies[0].get_results().next().unwrap();
    assert!(matches!(
        f.get_block(left).get_terminator(),
        Some(Terminator::Jmp(_, args)) if *args == vec![a, t1]
    ));
    assert!(matches!(
        f.get_block(right).get_terminator(),
        Some(Terminator::Jmp(_, args)) if *args == vec![b, copy]
    ));
    for block in [x, y] {
        assert!(matches!(
            f.get_block(block).get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![params[1], q]
        ));
    }
}

/// A loop header with two entry-side predecessors is beyond the hoist rule but not the join rule:
/// the in-loop evaluation is the back edge's own leader, so after its redirect the back edge
/// carries the parameter itself — the value survives iterations unchanged (binding stability) and
/// the loop body computes nothing.
#[test]
fn multi_entry_loop_header_joins_with_self_carrying_back_edge() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (a, b, c) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let (t1, tb, te) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let f = ssa.get_unique_entrypoint_mut();
    let e1 = f.add_block();
    let e2 = f.add_block();
    let h = f.add_block();
    let body = f.add_block();
    let exit = f.add_block();

    let entry = f.get_entry_mut();
    entry.push_parameter(a, Type::field());
    entry.push_parameter(b, Type::field());
    entry.push_parameter(c, Type::u(1));
    entry.set_terminator(Terminator::JmpIf(c, e1, e2));
    let e1b = f.get_block_mut(e1);
    e1b.push_test_instruction(add(t1, a, b));
    e1b.set_terminator(Terminator::Jmp(h, vec![]));
    f.get_block_mut(e2)
        .set_terminator(Terminator::Jmp(h, vec![]));
    f.get_block_mut(h)
        .set_terminator(Terminator::JmpIf(c, body, exit));
    // Sibling occurrences below the header: per-iteration in the body, and on the exit path.
    let bb = f.get_block_mut(body);
    bb.push_test_instruction(add(tb, a, b));
    bb.push_test_instruction(OpCode::Rangecheck {
        value: tb,
        max_bits: 32,
    });
    bb.set_terminator(Terminator::Jmp(h, vec![]));
    let eb = f.get_block_mut(exit);
    eb.push_test_instruction(add(te, a, b));
    eb.set_terminator(Terminator::Return(vec![te]));

    join_insert(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    let params: Vec<_> = f.get_block(h).get_parameter_values().copied().collect();
    assert_eq!(params.len(), 1);
    let param = params[0];
    assert!(matches!(
        f.get_block(e1).get_terminator(),
        Some(Terminator::Jmp(_, args)) if *args == vec![t1]
    ));
    let copies: Vec<_> = f.get_block(e2).get_instructions().collect();
    assert_eq!(copies.len(), 1);
    // The back edge's leader was the body's own evaluation; its redirect makes the edge
    // self-carry the parameter, and the per-iteration rangecheck now checks the parameter.
    assert!(matches!(
        f.get_block(body).get_terminator(),
        Some(Terminator::Jmp(_, args)) if *args == vec![param]
    ));
    assert!(
        f.get_block(body)
            .get_instructions()
            .any(|i| matches!(i, OpCode::Rangecheck { value, .. } if *value == param))
    );
    assert_eq!(return_values(&ssa), vec![param]);
}

/// The staging lever: below [`super::MotionLevel::JoinInsert`] the (otherwise profitable) diamond
/// stays untouched.
#[test]
fn join_insertion_requires_the_join_insert_level() {
    let (mut ssa, a, b, left, right, merge) = diamond();
    let t1 = ssa.fresh_value();
    let c = {
        let f = ssa.get_unique_entrypoint();
        f.get_entry()
            .get_parameter_values()
            .copied()
            .nth(2)
            .unwrap()
    };
    ssa.get_unique_entrypoint_mut()
        .get_block_mut(left)
        .push_test_instruction(add(t1, a, b));
    let (x, y, tx, ty) = sibling_recomputations(&mut ssa, merge, c, a, b, None);

    hoist(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert_eq!(f.get_block(merge).get_parameter_values().count(), 0);
    assert_eq!(f.get_block(right).get_instructions().count(), 0);
    for (block, t) in [(x, tx), (y, ty)] {
        assert!(matches!(
            f.get_block(block).get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![t]
        ));
    }
}

// SPECULATIVE HOISTING
// ================================================================================================

/// The while-loop LICM shape down-safety cannot reach: a body-only total invariant moves onto the
/// entry edge at the speculation level, and the per-iteration evaluation dies.
#[test]
fn body_only_invariant_is_speculated_to_entry_edge() {
    let (mut ssa, entry, _header, body, _exit) = invariant_loop(false);

    speculate(&mut ssa);

    let hoisted = sole_result(&ssa, entry);
    let f = ssa.get_unique_entrypoint();
    assert!(matches!(
        f.get_block(entry).get_instructions().next().unwrap(),
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Mul,
            ..
        }
    ));
    // The per-iteration rangecheck now checks the hoisted value.
    assert!(
        f.get_block(body)
            .get_instructions()
            .any(|i| matches!(i, OpCode::Rangecheck { value, .. } if *value == hoisted))
    );
}

/// The staging lever: at [`super::MotionLevel::JoinInsert`] (one level below) the same body-only
/// invariant stays put.
#[test]
fn speculation_requires_the_speculate_level() {
    let (mut ssa, entry, _, body, _) = invariant_loop(false);

    join_insert(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert_eq!(f.get_block(entry).get_instructions().count(), 0);
    assert_eq!(f.get_block(body).get_instructions().count(), 3);
}

/// The totality license: a `U(32)` `Add` wraps rather than traps, but integer overflow is
/// Noir-semantically an error and its witness lowering can emit a rejecting rangecheck, so the
/// oracle refuses it and the body-only occurrence must stay put even at the speculation level.
#[test]
fn wrapping_integer_invariant_is_not_speculated() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (a, b) = (ssa.fresh_value(), ssa.fresh_value());
    let c0 = ssa.add_const(Constant::U(32, 0));
    let c1 = ssa.add_const(Constant::U(32, 1));
    let c10 = ssa.add_const(Constant::U(32, 10));
    let (i, s, cond, u, i2) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    let header = f.add_block();
    let body = f.add_block();
    let exit = f.add_block();
    let entry_id = f.get_entry_id();

    let entry = f.get_entry_mut();
    entry.push_parameter(a, Type::u(32));
    entry.push_parameter(b, Type::u(32));
    entry.set_terminator(Terminator::Jmp(header, vec![c0, c0]));

    let hb = f.get_block_mut(header);
    hb.push_parameter(i, Type::u(32));
    hb.push_parameter(s, Type::u(32));
    hb.push_test_instruction(OpCode::Cmp {
        kind: CmpKind::Lt,
        result: cond,
        lhs: i,
        rhs: c10,
    });
    hb.set_terminator(Terminator::JmpIf(cond, body, exit));

    // The invariant wrapping sum, kept live as a loop-carried argument.
    let bb = f.get_block_mut(body);
    bb.push_test_instruction(add(u, a, b));
    bb.push_test_instruction(add(i2, i, c1));
    bb.set_terminator(Terminator::Jmp(header, vec![i2, u]));

    f.get_block_mut(exit)
        .set_terminator(Terminator::Return(vec![s]));

    speculate(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert_eq!(f.get_block(entry_id).get_instructions().count(), 0);
    assert_eq!(f.get_block(body).get_instructions().count(), 2);
}

/// A body-only division guarded by a dominating `d != 0` branch: the disequality fact holds at the
/// entry predecessor (the branch's false arm), so the totality gate discharges and the division
/// moves onto the guarded entry edge.
#[test]
fn guarded_division_is_speculated_where_the_fact_holds() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (x, d) = (ssa.fresh_value(), ssa.fresh_value());
    let c0 = ssa.add_const(Constant::U(32, 0));
    let c1 = ssa.add_const(Constant::U(32, 1));
    let c10 = ssa.add_const(Constant::U(32, 10));
    let (eq, i, acc, cond, q, i2) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    let skip = f.add_block();
    let pre = f.add_block();
    let header = f.add_block();
    let body = f.add_block();
    let exit = f.add_block();

    let entry = f.get_entry_mut();
    entry.push_parameter(x, Type::u(32));
    entry.push_parameter(d, Type::u(32));
    entry.push_test_instruction(OpCode::Cmp {
        kind: CmpKind::Eq,
        result: eq,
        lhs: d,
        rhs: c0,
    });
    entry.set_terminator(Terminator::JmpIf(eq, skip, pre));

    f.get_block_mut(skip)
        .set_terminator(Terminator::Return(vec![x]));
    f.get_block_mut(pre)
        .set_terminator(Terminator::Jmp(header, vec![c0, c0]));

    let hb = f.get_block_mut(header);
    hb.push_parameter(i, Type::u(32));
    hb.push_parameter(acc, Type::u(32));
    hb.push_test_instruction(OpCode::Cmp {
        kind: CmpKind::Lt,
        result: cond,
        lhs: i,
        rhs: c10,
    });
    hb.set_terminator(Terminator::JmpIf(cond, body, exit));

    let bb = f.get_block_mut(body);
    bb.push_test_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Div,
        result: q,
        lhs: x,
        rhs: d,
    });
    bb.push_test_instruction(add(i2, i, c1));
    bb.set_terminator(Terminator::Jmp(header, vec![i2, q]));

    f.get_block_mut(exit)
        .set_terminator(Terminator::Return(vec![acc]));

    speculate(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    let pre_instrs: Vec<_> = f.get_block(pre).get_instructions().collect();
    assert_eq!(pre_instrs.len(), 1);
    assert!(matches!(
        pre_instrs[0],
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Div,
            ..
        }
    ));
    let hoisted = *pre_instrs[0].get_results().next().unwrap();
    // The back edge now carries the hoisted quotient.
    assert!(matches!(
        f.get_block(body).get_terminator(),
        Some(Terminator::Jmp(_, args)) if *args == vec![i2, hoisted]
    ));
}

/// Without the guard the divisor is not provably nonzero anywhere, and the division must stay in
/// the body (a zero-trip run would come to trap on a hoisted copy).
#[test]
fn unguarded_division_is_not_speculated() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (x, d) = (ssa.fresh_value(), ssa.fresh_value());
    let c0 = ssa.add_const(Constant::U(32, 0));
    let c1 = ssa.add_const(Constant::U(32, 1));
    let c10 = ssa.add_const(Constant::U(32, 10));
    let (i, acc, cond, q, i2) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    let header = f.add_block();
    let body = f.add_block();
    let exit = f.add_block();
    let entry_id = f.get_entry_id();

    let entry = f.get_entry_mut();
    entry.push_parameter(x, Type::u(32));
    entry.push_parameter(d, Type::u(32));
    entry.set_terminator(Terminator::Jmp(header, vec![c0, c0]));

    let hb = f.get_block_mut(header);
    hb.push_parameter(i, Type::u(32));
    hb.push_parameter(acc, Type::u(32));
    hb.push_test_instruction(OpCode::Cmp {
        kind: CmpKind::Lt,
        result: cond,
        lhs: i,
        rhs: c10,
    });
    hb.set_terminator(Terminator::JmpIf(cond, body, exit));

    let bb = f.get_block_mut(body);
    bb.push_test_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Div,
        result: q,
        lhs: x,
        rhs: d,
    });
    bb.push_test_instruction(add(i2, i, c1));
    bb.set_terminator(Terminator::Jmp(header, vec![i2, q]));

    f.get_block_mut(exit)
        .set_terminator(Terminator::Return(vec![acc]));

    speculate(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert_eq!(f.get_block(entry_id).get_instructions().count(), 0);
    assert_eq!(f.get_block(body).get_instructions().count(), 2);
}

/// Speculative profitability: a post-loop-only occurrence eliminates nothing at greater depth, so
/// even with a total op and the speculation level enabled, nothing moves.
#[test]
fn post_loop_only_occurrence_is_not_speculated() {
    let (mut ssa, entry, exit, post) = post_loop_only_shape();

    speculate(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert_eq!(f.get_block(entry).get_instructions().count(), 0);
    assert!(matches!(
        f.get_block(exit).get_terminator(),
        Some(Terminator::Return(vals)) if *vals == vec![post]
    ));
}

// SOURCE LOCATION PROPAGATION
// ================================================================================================

/// The location every occurrence of the propagation tests carries, so the assertion holds
/// whichever occurrence the motion rule picks as its template.
fn test_location() -> SourceLocation {
    SourceLocation::new(
        "pre_test.nr",
        SourcePosition::new(3, 7),
        SourcePosition::new(3, 12),
    )
}

fn mul(
    result: crate::compiler::ssa::ValueId,
    lhs: crate::compiler::ssa::ValueId,
    rhs: crate::compiler::ssa::ValueId,
) -> OpCode {
    bin(BinaryArithOpKind::Mul, result, lhs, rhs)
}

/// A hoisted copy carries its template's source location, so a moved evaluation still attributes
/// its traps. The down-safe LICM shape of [`invariant_loop`] with located occurrences.
#[test]
fn hoisted_copy_carries_the_template_source_location() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (a, b) = (ssa.fresh_value(), ssa.fresh_value());
    let c0 = ssa.add_const(Constant::U(32, 0));
    let c1 = ssa.add_const(Constant::U(32, 1));
    let c10 = ssa.add_const(Constant::U(32, 10));
    let (i, cond, inv, i2, inv2) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    let header = f.add_block();
    let body = f.add_block();
    let exit = f.add_block();
    let entry_id = f.get_entry_id();

    let entry = f.get_entry_mut();
    entry.push_parameter(a, Type::field());
    entry.push_parameter(b, Type::field());
    entry.set_terminator(Terminator::Jmp(header, vec![c0]));

    let hb = f.get_block_mut(header);
    hb.push_parameter(i, Type::u(32));
    hb.push_test_instruction(OpCode::Cmp {
        kind: CmpKind::Lt,
        result: cond,
        lhs: i,
        rhs: c10,
    });
    hb.set_terminator(Terminator::JmpIf(cond, body, exit));

    let bb = f.get_block_mut(body);
    bb.push_instruction_with_source_location(mul(inv, a, b), test_location());
    bb.push_test_instruction(add(i2, i, c1));
    bb.set_terminator(Terminator::Jmp(header, vec![i2]));

    let eb = f.get_block_mut(exit);
    eb.push_instruction_with_source_location(mul(inv2, a, b), test_location());
    eb.set_terminator(Terminator::Return(vec![inv2]));

    hoist(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    let hoisted: Vec<_> = f
        .get_block(entry_id)
        .get_instructions_with_source_locations()
        .collect();
    assert_eq!(hoisted.len(), 1);
    assert_eq!(hoisted[0].1, &test_location());
}

/// A join copy materialized on a leaderless edge carries its template's source location. The
/// profitable diamond of [`diamond_partial_redundancy_joins_through_parameter`] with located
/// occurrences.
#[test]
fn join_copy_carries_the_template_source_location() {
    let (mut ssa, a, b, left, right, merge) = diamond();
    let (t1, tx, ty) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let c = {
        let f = ssa.get_unique_entrypoint();
        f.get_entry()
            .get_parameter_values()
            .copied()
            .nth(2)
            .unwrap()
    };
    let f = ssa.get_unique_entrypoint_mut();
    f.get_block_mut(left)
        .push_instruction_with_source_location(add(t1, a, b), test_location());
    let x = f.add_block();
    let y = f.add_block();
    f.get_block_mut(merge)
        .set_terminator(Terminator::JmpIf(c, x, y));
    for (block, t) in [(x, tx), (y, ty)] {
        let bb = f.get_block_mut(block);
        bb.push_instruction_with_source_location(add(t, a, b), test_location());
        bb.set_terminator(Terminator::Return(vec![t]));
    }

    join_insert(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    let copies: Vec<_> = f
        .get_block(right)
        .get_instructions_with_source_locations()
        .collect();
    assert_eq!(copies.len(), 1);
    assert_eq!(copies[0].1, &test_location());
}
