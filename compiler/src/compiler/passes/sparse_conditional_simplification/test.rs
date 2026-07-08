use super::*;
use crate::compiler::{
    Field,
    analysis::click_cooper::test::run_in_test,
    ssa::hlssa::{BinaryArithOpKind, CastTarget, CmpKind, Constant, SequenceTargetType, Type},
};

// TESTING UTILITIES
// ================================================================================================

/// Fold-only entry: runs the propagation rewrite without the integrated DCE, so assertions can
/// observe the pre-DCE state exactly (the integrated DCE would otherwise prune e.g. the return
/// slots of these caller-less test entrypoints, sweeping the values feeding them).
fn fold(ssa: &mut HLSSA) {
    let cc = run_in_test(ssa);
    let flow = FlowAnalysis::run(ssa);
    let type_info = Types::new().run(ssa, &flow);
    propagate_all(ssa, &cc, &type_info);
}

/// Full pass including the integrated DCE — for the DCE-integration tests below.
fn fold_and_dce(ssa: &mut HLSSA) {
    let cc = run_in_test(ssa);
    SCS::new(Config::preserve_blocks()).do_run(ssa, &cc);
}

// TESTS
// ================================================================================================

/// `2 + 3 == 5` decides the branch: the comparison chain folds away, the `JmpIf` becomes a
/// `Jmp` to the then-block, and the else-block is deleted.
#[test]
fn folds_constants_and_prunes_dead_branch() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c2 = ssa.add_const(Constant::U(32, 2));
    let c3 = ssa.add_const(Constant::U(32, 3));
    let c5 = ssa.add_const(Constant::U(32, 5));
    let (sum, is_five) = (ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    let then_b = f.add_block();
    let else_b = f.add_block();

    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: sum,
            lhs: c2,
            rhs: c3,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Eq,
            result: is_five,
            lhs: sum,
            rhs: c5,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::JmpIf(is_five, then_b, else_b));
    f.get_block_mut(then_b)
        .set_terminator(Terminator::Return(vec![sum]));
    f.get_block_mut(else_b)
        .set_terminator(Terminator::Return(vec![c2]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    // Both instructions folded away.
    assert_eq!(f.get_entry().get_instructions().count(), 0);
    // The branch is decided and the dead block is gone.
    assert!(matches!(
        f.get_entry().get_terminator(),
        Some(Terminator::Jmp(t, args)) if *t == then_b && args.is_empty()
    ));
    assert_eq!(f.get_blocks().count(), 2);
    // The surviving return now names the folded constant.
    assert!(matches!(
        f.get_block(then_b).get_terminator(),
        Some(Terminator::Return(vals)) if *vals == vec![c5]
    ));
}

/// A merge parameter receiving the same constant from both arms of an unknown branch folds to
/// that constant.
#[test]
fn folds_phi_of_agreeing_constants() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c7 = ssa.add_const(Constant::Field(Field::from(7u64)));
    let (cond, m_param) = (ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut().push_parameter(cond, Type::u(1));
    let a = f.add_block();
    let b = f.add_block();
    let merge = f.add_block();

    f.get_entry_mut()
        .set_terminator(Terminator::JmpIf(cond, a, b));
    f.get_block_mut(a)
        .set_terminator(Terminator::Jmp(merge, vec![c7]));
    f.get_block_mut(b)
        .set_terminator(Terminator::Jmp(merge, vec![c7]));
    let merge_block = f.get_block_mut(merge);
    merge_block.push_parameter(m_param, Type::field());
    merge_block.set_terminator(Terminator::Return(vec![m_param]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    // All blocks reachable (unknown condition), but the merge value is the constant.
    assert_eq!(f.get_blocks().count(), 4);
    assert!(matches!(
        f.get_block(merge).get_terminator(),
        Some(Terminator::Return(vals)) if *vals == vec![c7]
    ));
}

/// Constants propagate around a loop with a constant trip decision: the loop body is entered,
/// but the parameter that bottoms out (varies per iteration) is not folded while the invariant
/// one is.
#[test]
fn loop_variant_value_is_not_folded() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c0 = ssa.add_const(Constant::U(32, 0));
    let c1 = ssa.add_const(Constant::U(32, 1));
    let c10 = ssa.add_const(Constant::U(32, 10));
    let (i_param, lt, next) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    let header = f.add_block();
    let body = f.add_block();
    let exit = f.add_block();

    f.get_entry_mut()
        .set_terminator(Terminator::Jmp(header, vec![c0]));
    let header_block = f.get_block_mut(header);
    header_block.push_parameter(i_param, Type::u(32));
    header_block.push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Lt,
            result: lt,
            lhs: i_param,
            rhs: c10,
        },
        SourceLocation::test(),
    ));
    header_block.set_terminator(Terminator::JmpIf(lt, body, exit));
    let body_block = f.get_block_mut(body);
    body_block.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: next,
            lhs: i_param,
            rhs: c1,
        },
        SourceLocation::test(),
    ));
    body_block.set_terminator(Terminator::Jmp(header, vec![next]));
    f.get_block_mut(exit)
        .set_terminator(Terminator::Return(vec![i_param]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    // The loop-carried counter varies: nothing folds, all blocks survive.
    assert_eq!(f.get_blocks().count(), 4);
    assert_eq!(f.get_block(header).get_instructions().count(), 1);
    assert_eq!(f.get_block(body).get_instructions().count(), 1);
}

/// Overflowing integer arithmetic must not be folded: the backends' wrap behavior is not
/// modeled, so the instruction stays.
#[test]
fn does_not_fold_overflow() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c200 = ssa.add_const(Constant::U(8, 200));
    let c100 = ssa.add_const(Constant::U(8, 100));
    let sum = ssa.fresh_value();

    let f = ssa.get_unique_entrypoint_mut();
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: sum,
            lhs: c200,
            rhs: c100,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![sum]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert_eq!(f.get_entry().get_instructions().count(), 1);
    assert!(matches!(
        f.get_entry().get_terminator(),
        Some(Terminator::Return(vals)) if *vals == vec![sum]
    ));
}

/// A cast into the witness domain is never treated as a constant: the witness chain stays
/// intact.
#[test]
fn does_not_fold_witness_casts() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c5 = ssa.add_const(Constant::Field(Field::from(5u64)));
    let (wit, doubled) = (ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::Cast {
            result: wit,
            value: c5,
            target: CastTarget::WitnessOf,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: doubled,
            lhs: wit,
            rhs: wit,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![doubled]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert_eq!(f.get_entry().get_instructions().count(), 2);
}

/// A degenerate loop whose branch condition converges stuck at ⊤ must ICE rather than let the
/// rewrite delete blocks the kept `JmpIf` still targets. The ICE is raised by the ClickCooper
/// solver (`assert_no_stuck_conditions`) while `fold` builds the analysis (`run_in_test`), i.e.
/// before `propagate_all` runs — well ahead of the integrated DCE.
#[test]
#[should_panic(expected = "stuck at ⊤")]
fn degenerate_loop_ices_on_stuck_condition() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let cond = ssa.fresh_value();

    let f = ssa.get_unique_entrypoint_mut();
    let header = f.add_block();
    let body = f.add_block();
    let exit = f.add_block();

    f.get_entry_mut()
        .set_terminator(Terminator::Jmp(header, vec![]));
    f.get_block_mut(header)
        .set_terminator(Terminator::JmpIf(cond, body, exit));
    let body_block = f.get_block_mut(body);
    body_block.push_parameter(cond, Type::u(1));
    body_block.set_terminator(Terminator::Jmp(header, vec![]));
    f.get_block_mut(exit)
        .set_terminator(Terminator::Return(vec![]));

    fold(&mut ssa);
}

/// A select whose condition is constant aliases to the chosen arm even when the arms are not
/// constants.
#[test]
fn select_with_constant_condition_aliases_to_arm() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c_true = ssa.add_const(Constant::U(1, 1));
    let (arm_t, arm_f, sel) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut().push_parameter(arm_t, Type::field());
    f.get_entry_mut().push_parameter(arm_f, Type::field());
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::Select {
            result: sel,
            cond: c_true,
            if_t: arm_t,
            if_f: arm_f,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![sel]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert_eq!(f.get_entry().get_instructions().count(), 0);
    assert!(matches!(
        f.get_entry().get_terminator(),
        Some(Terminator::Return(vals)) if *vals == vec![arm_t]
    ));
}

/// A branch condition is path-sensitive: it is true in blocks reached only through the true
/// edge, even when it is an unknown function argument globally.
#[test]
fn branch_fact_folds_dominated_uses() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c_false = ssa.add_const(Constant::U(1, 0));
    let (cond, arm_t, arm_f, selected, not_cond) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    let then_b = f.add_block();
    let else_b = f.add_block();
    f.get_entry_mut().push_parameter(cond, Type::u(1));
    f.get_entry_mut().push_parameter(arm_t, Type::field());
    f.get_entry_mut().push_parameter(arm_f, Type::field());
    f.get_entry_mut()
        .set_terminator(Terminator::JmpIf(cond, then_b, else_b));

    let then_block = f.get_block_mut(then_b);
    then_block.push_instruction(Located::with(
        OpCode::Select {
            result: selected,
            cond,
            if_t: arm_t,
            if_f: arm_f,
        },
        SourceLocation::test(),
    ));
    then_block.push_instruction(Located::with(
        OpCode::Not {
            result: not_cond,
            value: cond,
        },
        SourceLocation::test(),
    ));
    then_block.set_terminator(Terminator::Return(vec![selected, not_cond]));
    f.get_block_mut(else_b)
        .set_terminator(Terminator::Return(vec![arm_f, c_false]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert_eq!(f.get_block(then_b).get_instructions().count(), 0);
    assert!(matches!(
        f.get_block(then_b).get_terminator(),
        Some(Terminator::Return(vals)) if *vals == vec![arm_t, c_false]
    ));
    assert!(matches!(
        f.get_entry().get_terminator(),
        Some(Terminator::JmpIf(c, t, e)) if *c == cond && *t == then_b && *e == else_b
    ));
}

/// Conflicting incoming predicate facts disappear at a join.
#[test]
fn branch_fact_does_not_cross_conflicting_merge() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (cond, not_cond) = (ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    let then_b = f.add_block();
    let else_b = f.add_block();
    let merge = f.add_block();
    f.get_entry_mut().push_parameter(cond, Type::u(1));
    f.get_entry_mut()
        .set_terminator(Terminator::JmpIf(cond, then_b, else_b));
    f.get_block_mut(then_b)
        .set_terminator(Terminator::Jmp(merge, vec![]));
    f.get_block_mut(else_b)
        .set_terminator(Terminator::Jmp(merge, vec![]));
    let merge_block = f.get_block_mut(merge);
    merge_block.push_instruction(Located::with(
        OpCode::Not {
            result: not_cond,
            value: cond,
        },
        SourceLocation::test(),
    ));
    merge_block.set_terminator(Terminator::Return(vec![not_cond]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert_eq!(f.get_block(merge).get_instructions().count(), 1);
    assert!(matches!(
        f.get_block(merge).get_terminator(),
        Some(Terminator::Return(vals)) if *vals == vec![not_cond]
    ));
}

/// A nested branch on a condition already known in the block is decided immediately.
#[test]
fn branch_fact_prunes_nested_same_condition_branch() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let cond = ssa.fresh_value();

    let f = ssa.get_unique_entrypoint_mut();
    let then_b = f.add_block();
    let else_b = f.add_block();
    let nested_then = f.add_block();
    let nested_else = f.add_block();
    f.get_entry_mut().push_parameter(cond, Type::u(1));
    f.get_entry_mut()
        .set_terminator(Terminator::JmpIf(cond, then_b, else_b));
    f.get_block_mut(then_b)
        .set_terminator(Terminator::JmpIf(cond, nested_then, nested_else));
    f.get_block_mut(else_b)
        .set_terminator(Terminator::Return(vec![]));
    f.get_block_mut(nested_then)
        .set_terminator(Terminator::Return(vec![]));
    f.get_block_mut(nested_else)
        .set_terminator(Terminator::Return(vec![]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    // `nested_else` is unreachable once the nested branch is decided, and is pruned (block
    // preservation only keeps blocks the analysis still reaches).
    assert_eq!(f.get_blocks().count(), 4);
    assert!(matches!(
        f.get_block(then_b).get_terminator(),
        Some(Terminator::Jmp(t, args)) if *t == nested_then && args.is_empty()
    ));
}

/// If both edges go to the same block, the edge itself carries no useful predicate fact.
#[test]
fn branch_with_same_successor_does_not_infer_condition() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (cond, not_cond) = (ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    let join = f.add_block();
    f.get_entry_mut().push_parameter(cond, Type::u(1));
    f.get_entry_mut()
        .set_terminator(Terminator::JmpIf(cond, join, join));
    let join_block = f.get_block_mut(join);
    join_block.push_instruction(Located::with(
        OpCode::Not {
            result: not_cond,
            value: cond,
        },
        SourceLocation::test(),
    ));
    join_block.set_terminator(Terminator::Return(vec![not_cond]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert_eq!(f.get_block(join).get_instructions().count(), 1);
    assert!(matches!(
        f.get_block(join).get_terminator(),
        Some(Terminator::Return(vals)) if *vals == vec![not_cond]
    ));
}

/// Jump arguments are evaluated under the branch facts of the edge carrying them.
#[test]
fn branch_fact_folds_phi_from_one_edge() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c_true = ssa.add_const(Constant::U(1, 1));
    let (cond, phi) = (ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    let then_b = f.add_block();
    let else_b = f.add_block();
    let merge = f.add_block();
    f.get_entry_mut().push_parameter(cond, Type::u(1));
    f.get_entry_mut()
        .set_terminator(Terminator::JmpIf(cond, then_b, else_b));
    f.get_block_mut(then_b)
        .set_terminator(Terminator::Jmp(merge, vec![cond]));
    f.get_block_mut(else_b)
        .set_terminator(Terminator::Return(vec![]));
    let merge_block = f.get_block_mut(merge);
    merge_block.push_parameter(phi, Type::u(1));
    merge_block.set_terminator(Terminator::Return(vec![phi]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert!(matches!(
        f.get_block(then_b).get_terminator(),
        Some(Terminator::Jmp(t, args)) if *t == merge && *args == vec![c_true]
    ));
    assert!(matches!(
        f.get_block(merge).get_terminator(),
        Some(Terminator::Return(vals)) if *vals == vec![c_true]
    ));
}

/// A branch decided by _congruence_ — `CmpEq(a, b)` with `a` and `b` structurally equal but not
/// constant — folds away through the combined-fixpoint writeback: the comparison is dropped, the
/// `JmpIf` becomes a `Jmp` to the then-block, and the dead else-block is deleted. The two
/// (non-constant) adds survive, and the purity assert does not trip on the folded `Cmp`.
#[test]
fn folds_congruence_decided_branch() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c1 = ssa.add_const(Constant::U(32, 1));
    let (x, a, b, eq) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    let then_b = f.add_block();
    let else_b = f.add_block();
    f.get_entry_mut().push_parameter(x, Type::u(32));
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: a,
            lhs: x,
            rhs: c1,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: b,
            lhs: x,
            rhs: c1,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Eq,
            result: eq,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::JmpIf(eq, then_b, else_b));
    f.get_block_mut(then_b)
        .set_terminator(Terminator::Return(vec![a]));
    f.get_block_mut(else_b)
        .set_terminator(Terminator::Return(vec![b]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    // The two adds survive (not constant); only the `Cmp` folded away.
    assert_eq!(f.get_entry().get_instructions().count(), 2);
    // The branch is decided and the dead else-block is gone.
    assert!(matches!(
        f.get_entry().get_terminator(),
        Some(Terminator::Jmp(t, args)) if *t == then_b && args.is_empty()
    ));
    assert_eq!(f.get_blocks().count(), 2);
}

/// A literal `AssertCmp{Eq, v, v}` is a tautology and is dropped (it would otherwise lower to a
/// wasted `Constrain` row).
#[test]
fn redundant_self_equality_assert_is_dropped() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let a = ssa.fresh_value();

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut().push_parameter(a, Type::field());
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: a,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert!(
        !f.get_entry()
            .get_instructions()
            .any(|i| matches!(i, OpCode::AssertCmp { .. })),
        "a tautological `AssertCmp{{Eq, v, v}}` should be dropped"
    );
}

/// An `AssertCmp{Eq, a, b}` whose operands are proven equal by _congruence_ (two identical
/// computations) is redundant and dropped — but the computations themselves survive.
#[test]
fn redundant_congruent_equality_assert_is_dropped() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c1 = ssa.add_const(Constant::U(32, 1));
    let (x, a, b) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut().push_parameter(x, Type::u(32));
    let entry = f.get_entry_mut();
    // Two identical adds: `a` and `b` are congruent (`known_equal`), independent of any assert.
    entry.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: a,
            lhs: x,
            rhs: c1,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: b,
            lhs: x,
            rhs: c1,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![a, b]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert!(
        !f.get_entry()
            .get_instructions()
            .any(|i| matches!(i, OpCode::AssertCmp { .. })),
        "an `AssertCmp{{Eq}}` on congruent operands should be dropped"
    );
    // The two (non-constant) adds are not removed by the congruence-gated drop.
    assert_eq!(
        f.get_entry()
            .get_instructions()
            .filter(|i| matches!(i, OpCode::BinaryArithOp { .. }))
            .count(),
        2
    );
}

/// Guard: an `AssertCmp{Eq, a, b}` on _non_-congruent operands (`known_equal` false) is
/// load-bearing and must be kept — dropping it would weaken the constraint system.
#[test]
fn non_congruent_equality_assert_is_kept() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (a, b) = (ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut().push_parameter(a, Type::field());
    f.get_entry_mut().push_parameter(b, Type::field());
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert!(
        f.get_entry().get_instructions().any(|i| matches!(
            i,
            OpCode::AssertCmp { kind: CmpKind::Eq, lhs, rhs } if *lhs == a && *rhs == b
        )),
        "a non-congruent equality assert must be kept (it is load-bearing)"
    );
}

/// A _witnessed_ comparison of congruent operands folds to a constant, but keeps its `WitnessOf`
/// type: it is redefined in place as a `cast <const> to WitnessOf` rather than aliased to a bare
/// constant, so the witnessed return slot stays correctly typed.
#[test]
fn witnessed_constant_is_cast_to_witness_of() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (w, ww) = (ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    let entry = f.get_entry_mut();
    // An operand is `WitnessOf`, so the comparison result is `WitnessOf(u1)`.
    entry.push_parameter(w, Type::witness_of(Type::u(32)));
    entry.push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Eq,
            result: ww,
            lhs: w,
            rhs: w,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![ww]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert!(
        !f.get_entry()
            .get_instructions()
            .any(|i| matches!(i, OpCode::Cmp { .. })),
        "the witnessed comparison should have folded away"
    );
    assert!(
        f.get_entry().get_instructions().any(|i| matches!(
            i,
            OpCode::Cast { result, target: CastTarget::WitnessOf, .. } if *result == ww
        )),
        "ww should be redefined as a cast to WitnessOf"
    );
    assert!(matches!(
        f.get_entry().get_terminator(),
        Some(Terminator::Return(vals)) if vals.as_slice() == [ww]
    ));
}

/// A constant lookup-table projection folds: `ArrayGet`/`SliceLen` over a constant `MkSeq` are
/// dropped and their uses aliased to the interned scalar constant, while the (never-surfaced)
/// aggregate `MkSeq` itself is left in place by the rewrite (the integrated DCE sweeps it — see
/// `integrated_dce_sweeps_dead_aggregate`).
#[test]
fn folds_constant_aggregate_projections() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c10 = ssa.add_const(Constant::U(32, 10));
    let c20 = ssa.add_const(Constant::U(32, 20));
    let c30 = ssa.add_const(Constant::U(32, 30));
    let idx = ssa.add_const(Constant::U(32, 1));
    let c_len = ssa.add_const(Constant::U(32, 3));
    let (seq, got, len) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    let entry = ssa.get_unique_entrypoint_mut().get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::MkSeq {
            result: seq,
            elems: vec![c10, c20, c30],
            seq_type: SequenceTargetType::Array(3),
            elem_type: Type::u(32),
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::ArrayGet {
            result: got,
            array: seq,
            index: idx,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::SliceLen {
            result: len,
            slice: seq,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![got, len]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    // Both projections folded away; only the (internal, now-dead) aggregate constructor remains.
    assert!(
        !f.get_entry()
            .get_instructions()
            .any(|i| matches!(i, OpCode::ArrayGet { .. } | OpCode::SliceLen { .. })),
        "the constant projections should have folded away"
    );
    assert_eq!(f.get_entry().get_instructions().count(), 1); // the surviving MkSeq
    // Their uses are aliased to the interned scalars: element 1 == 20, length == 3.
    assert!(matches!(
        f.get_entry().get_terminator(),
        Some(Terminator::Return(vals)) if vals.as_slice() == [c20, c_len]
    ));
}

// CONDITIONAL-LAYER TESTS
// --------------------------------------------------------------------------------------------

/// A value pinned to a constant by a dominating `Assert{v}` is substituted at later uses (here,
/// the return), while the establishing assert is kept.
#[test]
fn asserted_const_substituted_after_assert() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c_true = ssa.add_const(Constant::U(1, 1));
    let b = ssa.fresh_value();

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut().push_parameter(b, Type::u(1));
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::Assert { value: b },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![b]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    // The assert survives (it establishes the fact), and the return now names the constant.
    assert!(
        f.get_entry()
            .get_instructions()
            .any(|i| matches!(i, OpCode::Assert { .. })),
        "the establishing assert must survive"
    );
    assert!(matches!(
        f.get_entry().get_terminator(),
        Some(Terminator::Return(vals)) if *vals == vec![c_true]
    ));
}

/// `AssertCmp{Eq, x, c}` pins `x` to the constant `c`; later uses of `x` fold to `c`.
#[test]
fn asserted_const_substituted_after_assert_cmp_eq() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c5 = ssa.add_const(Constant::Field(Field::from(5u64)));
    let x = ssa.fresh_value();

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut().push_parameter(x, Type::field());
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: x,
            rhs: c5,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![x]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert!(matches!(
        f.get_entry().get_terminator(),
        Some(Terminator::Return(vals)) if *vals == vec![c5]
    ));
}

/// Index granularity within the asserting block: a use _after_ the assert folds via the
/// dominance ("already ran") direction, and a use _before_ it folds via the anticipated ("bound
/// to run") direction — straight-line execution reaches the assert on the very same binding.
/// What keeps the assert from folding itself into a tautology is that its _own_ operand is
/// never rewritten: `q == i` matches neither direction's index rule, and Gate 3 bars the
/// anticipated channel from assert inputs entirely.
#[test]
fn same_block_assert_is_index_granular() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c_true = ssa.add_const(Constant::U(1, 1));
    let (b, before, after) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut().push_parameter(b, Type::u(1));
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::Not {
            result: before,
            value: b,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::Assert { value: b },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::Not {
            result: after,
            value: b,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![before, after]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    let nots: Vec<(ValueId, ValueId)> = f
        .get_entry()
        .get_instructions()
        .filter_map(|i| match i {
            OpCode::Not { result, value } => Some((*result, *value)),
            _ => None,
        })
        .collect();
    let before_input = nots.iter().find(|(r, _)| *r == before).unwrap().1;
    let after_input = nots.iter().find(|(r, _)| *r == after).unwrap().1;
    // Both uses read the asserted constant — each through its own direction.
    assert_eq!(before_input, c_true);
    assert_eq!(after_input, c_true);
    // The assert itself still checks the genuine `b`.
    assert!(
        f.get_entry()
            .get_instructions()
            .any(|i| matches!(i, OpCode::Assert { value } if *value == b))
    );
}

/// On the false edge of an equality branch, a re-test `Cmp{Eq, a, b}` folds to `false` via the
/// known disequality.
#[test]
fn known_unequal_folds_cmp_eq_to_false() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c_false = ssa.add_const(Constant::U(1, 0));
    let (a, b, eq, eq2) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    let then_b = f.add_block();
    let else_b = f.add_block();
    f.get_entry_mut().push_parameter(a, Type::field());
    f.get_entry_mut().push_parameter(b, Type::field());
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Eq,
            result: eq,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    f.get_entry_mut()
        .set_terminator(Terminator::JmpIf(eq, then_b, else_b));
    f.get_block_mut(then_b)
        .set_terminator(Terminator::Return(vec![]));
    let else_block = f.get_block_mut(else_b);
    else_block.push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Eq,
            result: eq2,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    else_block.set_terminator(Terminator::Return(vec![eq2]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    // The re-test folded away; its use is the constant false.
    assert_eq!(f.get_block(else_b).get_instructions().count(), 0);
    assert!(matches!(
        f.get_block(else_b).get_terminator(),
        Some(Terminator::Return(vals)) if *vals == vec![c_false]
    ));
    // The establishing equality branch survives unchanged.
    assert!(matches!(
        f.get_entry().get_terminator(),
        Some(Terminator::JmpIf(c, _, _)) if *c == eq
    ));
}

/// Fold + prune: when a conditionally-folded `Cmp{Eq}` result is the block's `JmpIf` condition,
/// the terminator is folded to a `Jmp` on the taken target — never left as a constant-fed `JmpIf`
/// (the dead edge is pruned; the now-orphaned target is reclaimed before the run returns).
#[test]
fn known_unequal_folds_and_prunes_nested_jmpif() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c1 = ssa.add_const(Constant::U(32, 1));
    let c2 = ssa.add_const(Constant::U(32, 2));
    let (a, b, eq, eq2) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    let then_b = f.add_block();
    let else_b = f.add_block();
    let x_b = f.add_block();
    let y_b = f.add_block();
    f.get_entry_mut().push_parameter(a, Type::field());
    f.get_entry_mut().push_parameter(b, Type::field());
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Eq,
            result: eq,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    f.get_entry_mut()
        .set_terminator(Terminator::JmpIf(eq, then_b, else_b));
    f.get_block_mut(then_b)
        .set_terminator(Terminator::Return(vec![]));
    let else_block = f.get_block_mut(else_b);
    else_block.push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Eq,
            result: eq2,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    else_block.set_terminator(Terminator::JmpIf(eq2, x_b, y_b));
    f.get_block_mut(x_b)
        .set_terminator(Terminator::Return(vec![c1]));
    f.get_block_mut(y_b)
        .set_terminator(Terminator::Return(vec![c2]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    // `eq2` is `false` on this edge, so the nested branch is folded to a `Jmp` on the false
    // target — not left as a `JmpIf` on a constant condition.
    assert_eq!(f.get_block(else_b).get_instructions().count(), 0);
    assert!(matches!(
        f.get_block(else_b).get_terminator(),
        Some(Terminator::Jmp(t, args)) if *t == y_b && args.is_empty()
    ));
    // The pruned edge's target is orphaned by the fold and reclaimed within the same run.
    assert!(f.get_blocks().all(|(bid, _)| *bid != x_b));
}

/// A `Cmp{Eq, a, b}` whose operands are proven equal by an earlier same-block `AssertCmp{Eq}`
/// folds to `true` via `asserted_equal` — the conditional-fact mirror of
/// `known_unequal_folds_cmp_eq_to_false`. The establishing assert is kept (it carries the
/// fact), and neither operand is constant, so the equality is conditional rather than a
/// constant pin.
#[test]
fn asserted_equal_folds_cmp_eq_to_true() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c_true = ssa.add_const(Constant::U(1, 1));
    let (a, b, eq) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut().push_parameter(a, Type::field());
    f.get_entry_mut().push_parameter(b, Type::field());
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Eq,
            result: eq,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![eq]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    // The re-test folded away; its use is the constant true.
    assert!(
        !f.get_entry()
            .get_instructions()
            .any(|i| matches!(i, OpCode::Cmp { .. })),
        "the asserted equality should have folded the re-test away"
    );
    // The establishing equality assert survives.
    assert!(
        f.get_entry().get_instructions().any(|i| matches!(
            i,
            OpCode::AssertCmp {
                kind: CmpKind::Eq,
                ..
            }
        )),
        "the establishing AssertCmp must survive"
    );
    assert!(matches!(
        f.get_entry().get_terminator(),
        Some(Terminator::Return(vals)) if *vals == vec![c_true]
    ));
}

/// The witness-typed analog of `asserted_equal_folds_cmp_eq_to_true`: when the conditionally
/// folded `Cmp{Eq}` result is `WitnessOf`-typed it is redefined as a `Cast` to `WitnessOf`
/// (keeping the IR well-typed) rather than aliased to a bare constant. The operands are two
/// distinct witnesses, so the comparison is not congruence-decidable and only folds conditionally.
#[test]
fn witnessed_cmp_eq_folded_conditionally_is_cast_to_witness_of() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (a, b, ww) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut()
        .push_parameter(a, Type::witness_of(Type::field()));
    f.get_entry_mut()
        .push_parameter(b, Type::witness_of(Type::field()));
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    // A witness operand makes the result `WitnessOf(u1)`.
    entry.push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Eq,
            result: ww,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![ww]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert!(
        !f.get_entry()
            .get_instructions()
            .any(|i| matches!(i, OpCode::Cmp { .. })),
        "the witnessed comparison should have folded away"
    );
    assert!(
        f.get_entry().get_instructions().any(|i| matches!(
            i,
            OpCode::Cast { result, target: CastTarget::WitnessOf, .. } if *result == ww
        )),
        "the conditionally folded witnessed result should be recast to WitnessOf"
    );
    assert!(matches!(
        f.get_entry().get_terminator(),
        Some(Terminator::Return(vals)) if vals.as_slice() == [ww]
    ));
}

/// A select whose condition is a constant `false` aliases to the else arm — symmetric to
/// `select_with_constant_condition_aliases_to_arm`, which covers the true arm.
#[test]
fn select_with_constant_false_condition_aliases_to_else_arm() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c_false = ssa.add_const(Constant::U(1, 0));
    let (arm_t, arm_f, sel) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut().push_parameter(arm_t, Type::field());
    f.get_entry_mut().push_parameter(arm_f, Type::field());
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::Select {
            result: sel,
            cond: c_false,
            if_t: arm_t,
            if_f: arm_f,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![sel]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert_eq!(f.get_entry().get_instructions().count(), 0);
    assert!(matches!(
        f.get_entry().get_terminator(),
        Some(Terminator::Return(vals)) if *vals == vec![arm_f]
    ));
}

/// Per-point asserted-constant substitution reaches `Jmp` arguments, not only `Return` values:
/// a value pinned by a dominating `Assert` is substituted where it is passed as a block
/// argument.
#[test]
fn asserted_const_substituted_in_jmp_args() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c_true = ssa.add_const(Constant::U(1, 1));
    let (b, p) = (ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    let target = f.add_block();
    f.get_entry_mut().push_parameter(b, Type::u(1));
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::Assert { value: b },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Jmp(target, vec![b]));
    let target_block = f.get_block_mut(target);
    target_block.push_parameter(p, Type::u(1));
    target_block.set_terminator(Terminator::Return(vec![]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    // The asserted constant is substituted into the jump argument.
    assert!(matches!(
        f.get_entry().get_terminator(),
        Some(Terminator::Jmp(t, args)) if *t == target && args.as_slice() == [c_true]
    ));
}

/// A `WitnessOf`-typed operand pinned to a constant by an `Assert`, used by a _pure value_
/// consumer (here the return), is substituted with a witness-typed constant: a
/// `cast <const> to WitnessOf` is hoisted into the entry block and the use is redirected to it,
/// keeping the IR well-typed. The establishing assert is retained.
#[test]
fn asserted_const_substituted_for_witness_operand_via_cast() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let w = ssa.fresh_value();

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut()
        .push_parameter(w, Type::witness_of(Type::u(1)));
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::Assert { value: w },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![w]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    let entry = f.get_entry();

    // A `cast <const> to WitnessOf` was hoisted to the front of the entry block.
    let cast_result = entry
        .get_instructions()
        .find_map(|i| match i {
            OpCode::Cast {
                result,
                target: CastTarget::WitnessOf,
                ..
            } => Some(*result),
            _ => None,
        })
        .expect("a witness cast should be hoisted into the entry block");

    // The witness-typed return operand is redirected to that witness-typed constant, not `w`.
    assert!(matches!(
        entry.get_terminator(),
        Some(Terminator::Return(vals)) if *vals == vec![cast_result]
    ));

    // The establishing assert is retained (it carries the fact into R1CS).
    assert!(
        entry
            .get_instructions()
            .any(|i| matches!(i, OpCode::Assert { value } if *value == w)),
        "the establishing assert must survive"
    );
}

/// A `WitnessOf`-typed operand pinned to a constant by an `Assert` is _not_ substituted when its
/// consumer is an excluded witness-machinery op (here `WriteWitness`, the honest hint): such a
/// consumer must keep referencing the real witness, so neither the operand is redirected nor a
/// witness cast is materialized.
#[test]
fn asserted_const_not_substituted_for_excluded_witness_consumer() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let w = ssa.fresh_value();

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut()
        .push_parameter(w, Type::witness_of(Type::u(1)));
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::Assert { value: w },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::WriteWitness {
            result: None,
            value: w,
            pinned: false,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    let entry = f.get_entry();
    // The excluded consumer still references the original witness `w`.
    assert!(
        entry
            .get_instructions()
            .any(|i| matches!(i, OpCode::WriteWitness { value, .. } if *value == w)),
        "the excluded consumer must keep referencing the real witness"
    );
    // No witness cast was hoisted, since the only candidate use was excluded.
    assert!(
        !entry.get_instructions().any(|i| matches!(
            i,
            OpCode::Cast {
                target: CastTarget::WitnessOf,
                ..
            }
        )),
        "no witness cast should be materialized for an excluded consumer"
    );
}

/// An operand proven equal to a dominating value by an `AssertCmp{Eq}` is redirected to that
/// value (the class's dominance-root-most member): the return's `b` becomes the param `a`,
/// whose definition dominates `b = a + a`. The establishing assert is kept.
#[test]
fn asserted_equal_copy_propagates_to_dominating_leader() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (a, b) = (ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut().push_parameter(a, Type::field());
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: b,
            lhs: a,
            rhs: a,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![b]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    // `b` is redirected to its dominating asserted-equal representative `a`.
    assert!(matches!(
        f.get_entry().get_terminator(),
        Some(Terminator::Return(vals)) if *vals == vec![a]
    ));
    // The establishing equality assert survives (it carries the fact).
    assert!(
        f.get_entry().get_instructions().any(|i| matches!(
            i,
            OpCode::AssertCmp {
                kind: CmpKind::Eq,
                ..
            }
        )),
        "the establishing AssertCmp must survive"
    );
}

/// Both index directions within the asserting block, copy-prop form: a use _after_ the
/// establishing assert redirects via the dominance ("already ran") direction, and a use before
/// it via the anticipated ("bound to run") direction — the copy-prop counterpart of
/// `same_block_assert_is_index_granular`. The assert itself still checks the genuine `a`, `b`:
/// Gate 3 bars the anticipated channel from assert inputs, which is the redirect channel's only
/// self-protection at the establishing index.
#[test]
fn asserted_equal_copy_prop_redirects_both_directions() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c0 = ssa.add_const(Constant::U(32, 0));
    let (a, b, before, after) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut().push_parameter(a, Type::u(32));
    f.get_entry_mut().push_parameter(b, Type::u(32));
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: before,
            lhs: b,
            rhs: c0,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: after,
            lhs: b,
            rhs: c0,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![before, after]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    let adds: Vec<(ValueId, ValueId)> = f
        .get_entry()
        .get_instructions()
        .filter_map(|i| match i {
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result,
                lhs,
                ..
            } => Some((*result, *lhs)),
            _ => None,
        })
        .collect();
    let before_lhs = adds.iter().find(|(r, _)| *r == before).unwrap().1;
    let after_lhs = adds.iter().find(|(r, _)| *r == after).unwrap().1;
    // Both uses read the leader `a` — each through its own direction.
    assert_eq!(before_lhs, a);
    assert_eq!(after_lhs, a);
    // The establishing assert survives and still checks the genuine `a`, `b`.
    assert!(
        f.get_entry().get_instructions().any(|i| matches!(
            i,
            OpCode::AssertCmp {
                kind: CmpKind::Eq,
                lhs,
                rhs,
            } if *lhs == a && *rhs == b
        )),
        "the establishing AssertCmp must survive on its genuine operands"
    );
}

/// A block-defined leader redirects too: with `x` and `y` both defined by the block's own
/// instructions and asserted equal _later_, a use of `y` between the definitions and the assert
/// is redirected to the earlier-defined `x` (the class's DefKey minimum, valid from its own
/// definition onward).
#[test]
fn anticipated_copy_prop_to_block_defined_leader() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c1 = ssa.add_const(Constant::U(32, 1));
    let c2 = ssa.add_const(Constant::U(32, 2));
    let (a, b, x, y, m) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut().push_parameter(a, Type::u(32));
    f.get_entry_mut().push_parameter(b, Type::u(32));
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            // index 0: def(x)
            kind: BinaryArithOpKind::Add,
            result: x,
            lhs: a,
            rhs: c1,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            // index 1: def(y)
            kind: BinaryArithOpKind::Add,
            result: y,
            lhs: b,
            rhs: c2,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            // index 2: a use of `y` before the assert
            kind: BinaryArithOpKind::Add,
            result: m,
            lhs: y,
            rhs: c1,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::AssertCmp {
            // index 3: x == y
            kind: CmpKind::Eq,
            lhs: x,
            rhs: y,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![m]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    let m_lhs = f
        .get_entry()
        .get_instructions()
        .find_map(|i| match i {
            OpCode::BinaryArithOp { result, lhs, .. } if *result == m => Some(*lhs),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        m_lhs, x,
        "the pre-assert use of `y` redirects to the leader `x`"
    );
    // The establishing assert survives on its genuine operands.
    assert!(
        f.get_entry().get_instructions().any(|i| matches!(
            i,
            OpCode::AssertCmp {
                kind: CmpKind::Eq,
                lhs,
                rhs,
            } if *lhs == x && *rhs == y
        )),
        "the establishing AssertCmp must survive on its genuine operands"
    );
}

/// Guard C: the leader is value-id/structure based, so a redirect across a type mismatch is
/// suppressed (an `AssertCmp{Eq}`'s operand types are not enforced equal). `y: u8` is not
/// redirected to `x: u32`, keeping the IR well-typed.
#[test]
fn asserted_equal_copy_prop_skips_type_mismatch() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (x, y) = (ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut().push_parameter(x, Type::u(32));
    f.get_entry_mut().push_parameter(y, Type::u(8));
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: x,
            rhs: y,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![y]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert!(matches!(
        f.get_entry().get_terminator(),
        Some(Terminator::Return(vals)) if *vals == vec![y]
    ));
}

/// The class is transitive: `c == b == a`, so a use of `c` redirects all the way to `a`, even
/// though no single `AssertCmp` names both `c` and `a`.
#[test]
fn asserted_equal_copy_prop_is_transitive() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (a, b, c) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut().push_parameter(a, Type::field());
    f.get_entry_mut().push_parameter(b, Type::field());
    f.get_entry_mut().push_parameter(c, Type::field());
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: b,
            rhs: c,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![c]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert!(matches!(
        f.get_entry().get_terminator(),
        Some(Terminator::Return(vals)) if *vals == vec![a]
    ));
}

/// The redirect reaches `Jmp` block-arguments, not only `Return` values — the second of the two
/// terminator call sites.
#[test]
fn asserted_equal_copy_prop_reaches_jmp_args() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (a, b, p) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    let target = f.add_block();
    f.get_entry_mut().push_parameter(a, Type::field());
    f.get_entry_mut().push_parameter(b, Type::field());
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Jmp(target, vec![b]));
    let target_block = f.get_block_mut(target);
    target_block.push_parameter(p, Type::field());
    target_block.set_terminator(Terminator::Return(vec![]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert!(matches!(
        f.get_entry().get_terminator(),
        Some(Terminator::Jmp(t, args)) if *t == target && args.as_slice() == [a]
    ));
}

/// Guard A: a function-wide fold beats a conditional redirect. `v = Cmp{Eq, x, y}` folds to the
/// constant `true` (via the dominating `AssertCmp{Eq, x, y}`), and is _also_ placed in an
/// asserted-equal class with `w`. The return's `v` must become the constant `true` — not its
/// class-mate `w` — because the unconditional fold is strictly stronger. A plain
/// unconditionally-constant operand would not exercise this: an equality pair is only filed
/// when _both_ sides are non-constant, so it would already yield no leader.
#[test]
fn asserted_equal_copy_prop_yields_to_function_wide_fold() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c_true = ssa.add_const(Constant::U(1, 1));
    let (x, y, w, v) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut().push_parameter(x, Type::field());
    f.get_entry_mut().push_parameter(y, Type::field());
    f.get_entry_mut().push_parameter(w, Type::u(1));
    let entry = f.get_entry_mut();
    // x == y pins the comparison `v` to `true` at later points.
    entry.push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: x,
            rhs: y,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Eq,
            result: v,
            lhs: x,
            rhs: y,
        },
        SourceLocation::test(),
    ));
    // `v` is also placed in an asserted-equal class with `w`.
    entry.push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: v,
            rhs: w,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![v]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    // The constant fold of `v` wins; `v` is not redirected to `w`.
    assert!(matches!(
        f.get_entry().get_terminator(),
        Some(Terminator::Return(vals)) if *vals == vec![c_true]
    ));
}

/// Equal _witness_ types copy-propagate: unlike bare-constant substitution there is no
/// `is_witness_of` skip, because Guard C already proves the redirect type-safe (both operands
/// are the same `WitnessOf` type).
#[test]
fn asserted_equal_copy_prop_redirects_equal_witnesses() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (wa, wb) = (ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut()
        .push_parameter(wa, Type::witness_of(Type::field()));
    f.get_entry_mut()
        .push_parameter(wb, Type::witness_of(Type::field()));
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: wa,
            rhs: wb,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![wb]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert!(matches!(
        f.get_entry().get_terminator(),
        Some(Terminator::Return(vals)) if *vals == vec![wa]
    ));
}

/// Copy-propagation redirects a witness operand of a _witness-machinery_ consumer (here
/// `WriteWitness`, the honest hint) to its asserted-equal leader — it has no
/// `is_witness_subst_safe_target` allowlist, unlike `substitute_asserted_consts`. This is sound
/// because the operand is redirected to _another real witness_ the kept assert constrains equal
/// (not displaced by a constant, which is why the _constant_ substitution excludes this very
/// consumer — see `asserted_const_not_substituted_for_excluded_witness_consumer`).
#[test]
fn asserted_equal_copy_prop_redirects_witness_machinery_consumer() {
    let mut ssa = HLSSA::with_main("main".to_string());
    // `wa` first ⇒ `wa.0 < wb.0`; both are entry params (same def site), so the dominance-root-
    // most leader is the smaller-id `wa`.
    let (wa, wb) = (ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut()
        .push_parameter(wa, Type::witness_of(Type::field()));
    f.get_entry_mut()
        .push_parameter(wb, Type::witness_of(Type::field()));
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: wa,
            rhs: wb,
        },
        SourceLocation::test(),
    ));

    // A witness-machinery consumer of `wb`: its `value` operand is exposed via `get_inputs_mut`
    // like every witness-machinery operand (`Lookup`/`DLookup` args included), so copy-prop
    // reaches it.
    entry.push_instruction(Located::with(
        OpCode::WriteWitness {
            result: None,
            value: wb,
            pinned: false,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    // The `WriteWitness` value operand is redirected from `wb` to the leader `wa`.
    assert!(
        f.get_entry()
            .get_instructions()
            .any(|i| matches!(i, OpCode::WriteWitness { value, .. } if *value == wa)),
        "the witness-machinery consumer should be redirected to the asserted-equal leader"
    );
    // The establishing assert survives (it carries the equality the redirect relies on).
    assert!(
        f.get_entry().get_instructions().any(|i| matches!(
            i,
            OpCode::AssertCmp {
                kind: CmpKind::Eq,
                ..
            }
        )),
        "the establishing AssertCmp must survive"
    );
}

/// The establishing assert (and the `b = a + a` def it pins live) survive the integrated DCE
/// even after the return's `b` is redirected to `a` — so the equality the redirect relied on is
/// never broken. `AssertCmp` is unconditionally live in DCE and marks its operands live, which
/// is what keeps `b`'s def alive; this guards against a regression that stopped doing so.
#[test]
fn asserted_equal_copy_prop_keeps_establisher_under_dce() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (a, b) = (ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut().push_parameter(a, Type::field());
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: b,
            lhs: a,
            rhs: a,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![b]));

    fold_and_dce(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    assert!(
        f.get_entry().get_instructions().any(|i| matches!(
            i,
            OpCode::AssertCmp {
                kind: CmpKind::Eq,
                ..
            }
        )),
        "the establishing AssertCmp must survive DCE"
    );
    assert!(
        f.get_entry()
            .get_instructions()
            .any(|i| matches!(i, OpCode::BinaryArithOp { result, .. } if *result == b)),
        "the `b` def stays live, pinned live by the assert"
    );
}

/// The redundant-assert drop (4b) and copy-propagation cooperate in one pass: an
/// `AssertCmp{Eq, a, b}` on _congruent_ operands is dropped (4b, `known_equal`), yet the
/// second-walk copy-prop still redirects a use of `b` to the leader `a` — because
/// `asserted_leader` is read from the analysis built on the pristine IR, which still has the
/// assert. This is sound precisely because the drop is congruence-backed: `a` and `b` compute
/// the same thing (`x + 1`), so `a == b` holds in every witness independently of the dropped
/// assert, and the redirect is well-typed (both `u32`). Regression guard for the non-local
/// interaction. (Fold-only: the integrated DCE would prune this caller-less entrypoint's return
/// slot and sweep everything — sound, but it would hide the redirect.)
#[test]
fn congruence_dropped_assert_still_copy_propagates_soundly() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c1 = ssa.add_const(Constant::U(32, 1));
    let (x, a, b) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut().push_parameter(x, Type::u(32));
    let entry = f.get_entry_mut();
    // Two identical adds ⇒ `a` and `b` are congruent (`known_equal`), `a` defined first.
    entry.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: a,
            lhs: x,
            rhs: c1,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: b,
            lhs: x,
            rhs: c1,
        },
        SourceLocation::test(),
    ));
    // Congruence-redundant: 4b drops this in the first walk.
    entry.push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    // A use of `b` the second-walk copy-prop redirects to the leader `a`.
    entry.set_terminator(Terminator::Return(vec![b]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    // 4b dropped the redundant assert.
    assert!(
        !f.get_entry()
            .get_instructions()
            .any(|i| matches!(i, OpCode::AssertCmp { .. })),
        "the congruence-redundant assert should be dropped"
    );
    // The use of `b` was redirected to the leader `a`, even though the connecting assert was
    // dropped in the same pass.
    assert!(matches!(
        f.get_entry().get_terminator(),
        Some(Terminator::Return(vals)) if *vals == vec![a]
    ));
    // `a`'s def survives (it now feeds the return) and is the well-typed `u32` add.
    assert!(
        f.get_entry()
            .get_instructions()
            .any(|i| matches!(i, OpCode::BinaryArithOp { result, .. } if *result == a)),
        "the leader's def must survive to feed the redirected use"
    );
}

/// The integrated DCE runs in the same pass: dead code (here, an unused instruction) is removed
/// without a separate cleanup pass. Fold-only would leave the add (it is not constant); the
/// full pass sweeps it.
#[test]
fn integrated_dce_removes_dead_code() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c1 = ssa.add_const(Constant::U(32, 1));
    let (x, unused) = (ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut().push_parameter(x, Type::u(32));
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: unused,
            lhs: x,
            rhs: c1,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![x]));

    fold_and_dce(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    // The unused add is swept by the integrated DCE.
    assert!(
        !f.get_entry()
            .get_instructions()
            .any(|i| matches!(i, OpCode::BinaryArithOp { .. })),
        "the unused add should be removed by the integrated DCE"
    );
}

/// A fold that _creates_ dead code: once the constant aggregate projections fold away, the
/// (never-surfaced) `MkSeq` is dead, and the integrated DCE sweeps it in the same pass —
/// contrast `folds_constant_aggregate_projections`, where fold-only leaves it.
#[test]
fn integrated_dce_sweeps_dead_aggregate() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c10 = ssa.add_const(Constant::U(32, 10));
    let c20 = ssa.add_const(Constant::U(32, 20));
    let c30 = ssa.add_const(Constant::U(32, 30));
    let idx = ssa.add_const(Constant::U(32, 1));
    let (seq, got, len) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    let entry = ssa.get_unique_entrypoint_mut().get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::MkSeq {
            result: seq,
            elems: vec![c10, c20, c30],
            seq_type: SequenceTargetType::Array(3),
            elem_type: Type::u(32),
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::ArrayGet {
            result: got,
            array: seq,
            index: idx,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::SliceLen {
            result: len,
            slice: seq,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![got, len]));

    fold_and_dce(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    // The projections fold and the now-dead aggregate is swept: nothing remains.
    assert_eq!(f.get_entry().get_instructions().count(), 0);
}

/// Regression guard for the orphan a conditional `Cmp{Eq}`->`JmpIf` fold leaves behind.
/// `eq2 = (a == b)` is proven only by the dominating `AssertCmp{Eq, a, b}` — a _conditional_
/// fold (`a`, `b` are distinct params, never congruent/constant), so the analysis never pruned
/// the branch and step 1 cannot delete the block the fold strands. Folding
/// `JmpIf(eq2, then_b, else_b)` to `Jmp(then_b)` therefore orphans `else_b` together with its
/// constraint-bearing `AssertCmp{Eq, p, q}`, which the integrated DCE would keep under
/// `preserve_blocks()` (DCE seeds every `Return` block live).
///
/// The pass must reclaim the orphan _within the same run_ (step 10): the passes that may follow
/// an SCS placement walk every block and type instructions against reachable-only `TypeInfo`
/// (the `PointsTo` build, `InstructionLowering`'s value-range analysis), so an escaped orphan is
/// an ICE — and its constraint must never leak into the circuit. This test pins the same-run
/// reclamation of the block and its constraint.
#[test]
fn conditional_jmpif_fold_orphan_is_reclaimed_same_run() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (a, b, p, q, eq2) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    let then_b = f.add_block();
    let else_b = f.add_block();
    f.get_entry_mut().push_parameter(a, Type::field());
    f.get_entry_mut().push_parameter(b, Type::field());
    f.get_entry_mut().push_parameter(p, Type::field());
    f.get_entry_mut().push_parameter(q, Type::field());
    let entry = f.get_entry_mut();
    // Dominating assert establishes `a == b` conditionally (distinct params, so never a
    // congruence/constant fact); the re-test below folds to `true` only via `asserted_equal`.
    entry.push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Eq,
            result: eq2,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::JmpIf(eq2, then_b, else_b));
    f.get_block_mut(then_b)
        .set_terminator(Terminator::Return(vec![]));
    // `else_b`'s sole predecessor is the entry, so folding the branch orphans it. Its distinct,
    // load-bearing constraint must not leak into the circuit.
    let else_block = f.get_block_mut(else_b);
    else_block.push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: p,
            rhs: q,
        },
        SourceLocation::test(),
    ));
    else_block.set_terminator(Terminator::Return(vec![]));

    // The orphan's constraint, found anywhere in the function.
    let orphan_constraint_present = |s: &HLSSA| {
        s.get_unique_entrypoint().get_blocks().any(|(_, block)| {
            block.get_instructions().any(|i| {
                matches!(
                    i,
                    OpCode::AssertCmp { kind: CmpKind::Eq, lhs, rhs } if *lhs == p && *rhs == q
                )
            })
        })
    };

    // One pass: the conditional fold fires, and the orphan — constraint and all — is reclaimed
    // before the run returns.
    let cc = run_in_test(&ssa);
    SCS::new(Config::preserve_blocks()).do_run(&mut ssa, &cc);
    assert!(
        matches!(
            ssa.get_unique_entrypoint().get_entry().get_terminator(),
            Some(Terminator::Jmp(t, args)) if *t == then_b && args.is_empty()
        ),
        "the conditional `Cmp{{Eq}}`->`JmpIf` fold should rewrite the branch to a `Jmp`"
    );
    assert!(
        !orphan_constraint_present(&ssa),
        "the orphaned constraint must not survive the run that folded the branch"
    );
    assert!(
        ssa.get_unique_entrypoint()
            .get_blocks()
            .all(|(bid, _)| *bid != else_b),
        "the orphaned block itself must be reclaimed within the same run"
    );
}

// ANTICIPATED-LAYER TESTS
// --------------------------------------------------------------------------------------------

/// A constant pinned by a _post-dominating_ assert substitutes into a use that runs before the
/// assert, while the assert itself keeps its genuine operand (Gate 3).
#[test]
fn anticipated_const_folds_earlier_use() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c1 = ssa.add_const(Constant::U(32, 1));
    let c5 = ssa.add_const(Constant::U(32, 5));
    let (x, r) = (ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    let tail = f.add_block();
    f.get_entry_mut().push_parameter(x, Type::u(32));
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: r,
            lhs: x,
            rhs: c1,
        },
        SourceLocation::test(),
    ));
    f.get_entry_mut()
        .set_terminator(Terminator::Jmp(tail, vec![]));
    f.get_block_mut(tail).push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: x,
            rhs: c5,
        },
        SourceLocation::test(),
    ));
    f.get_block_mut(tail)
        .set_terminator(Terminator::Return(vec![r]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    // The `Add`'s `x` operand — which runs _before_ the assert — is substituted with `5`.
    assert!(matches!(
        f.get_entry().get_instructions().next(),
        Some(OpCode::BinaryArithOp { lhs, .. }) if *lhs == c5
    ));
    // The assert still checks the genuine `x`: Gate 3 keeps anticipated facts out of it.
    assert!(matches!(
        f.get_block(tail).get_instructions().next(),
        Some(OpCode::AssertCmp { kind: CmpKind::Eq, lhs, .. }) if *lhs == x
    ));
}

/// A `Cmp{Eq}` proven by a _bound-to-run_ assert folds to `true` and decides its own block's
/// `JmpIf` — while the folded `Cmp` is kept (an excluded assert could still reference it) and
/// the justifying assert keeps its genuine operands.
#[test]
fn anticipated_cmp_fold_decides_branch() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (a, b, w) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    let then_b = f.add_block();
    let else_b = f.add_block();
    let merge = f.add_block();
    f.get_entry_mut().push_parameter(a, Type::u(32));
    f.get_entry_mut().push_parameter(b, Type::u(32));
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Eq,
            result: w,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    f.get_entry_mut()
        .set_terminator(Terminator::JmpIf(w, then_b, else_b));
    f.get_block_mut(then_b)
        .set_terminator(Terminator::Jmp(merge, vec![]));
    f.get_block_mut(else_b)
        .set_terminator(Terminator::Jmp(merge, vec![]));
    f.get_block_mut(merge).push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    f.get_block_mut(merge)
        .set_terminator(Terminator::Return(vec![]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    // The branch is decided on the anticipated fold of `w` (sound: both targets reach the
    // justifying assert — `merge` post-dominates the branch block)...
    assert!(matches!(
        f.get_entry().get_terminator(),
        Some(Terminator::Jmp(t, args)) if *t == then_b && args.is_empty()
    ));
    // ...the folded `Cmp` survives the rewrite...
    assert!(matches!(
        f.get_entry().get_instructions().next(),
        Some(OpCode::Cmp {
            kind: CmpKind::Eq,
            ..
        })
    ));
    // ...and the justifying assert keeps its genuine operands.
    assert!(matches!(
        f.get_block(merge).get_instructions().next(),
        Some(OpCode::AssertCmp { kind: CmpKind::Eq, lhs, rhs }) if *lhs == a && *rhs == b
    ));
}

/// THE Gate-3 regression — mutual erasure: two identical checks of one fact, each in the
/// other's fan-out. The dominance fact from the first legitimately folds the second's operand
/// into a droppable tautology; without Gate 3 the _anticipated_ fact from the second would
/// symmetrically vacate the first, and `v == 5` would silently vanish from the circuit.
/// Exactly one genuine check must remain, no matter how many passes run.
#[test]
fn mutual_erasure_keeps_one_assert() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c5 = ssa.add_const(Constant::U(32, 5));
    let v = ssa.fresh_value();

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut().push_parameter(v, Type::u(32));
    for _ in 0..2 {
        f.get_entry_mut().push_instruction(Located::with(
            OpCode::AssertCmp {
                kind: CmpKind::Eq,
                lhs: v,
                rhs: c5,
            },
            SourceLocation::test(),
        ));
    }
    f.get_entry_mut().set_terminator(Terminator::Return(vec![]));

    // Two rounds, each with a fresh analysis: the first substitutes what it can, the second
    // acts on the rewritten IR (dropping the tautologized duplicate).
    fold(&mut ssa);
    fold(&mut ssa);

    let genuine_checks = ssa
        .get_unique_entrypoint()
        .get_entry()
        .get_instructions()
        .filter(|i| {
            matches!(
                i,
                OpCode::AssertCmp { kind: CmpKind::Eq, lhs, rhs }
                    if (*lhs == v && *rhs == c5) || (*lhs == c5 && *rhs == v)
            )
        })
        .count();
    assert_eq!(
        genuine_checks, 1,
        "exactly the earliest assert survives with its genuine operand — losing both would \
         silently drop the `v == 5` constraint"
    );
}

/// The stability gate end-to-end: a loop-carried value asserted _after_ the loop is not
/// rewritten inside it — pinning the counter to its final value would collapse the loop and
/// turn accepting runs rejecting.
#[test]
fn loop_variant_value_not_rewritten_by_later_assert() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c0 = ssa.add_const(Constant::U(32, 0));
    let c1 = ssa.add_const(Constant::U(32, 1));
    let c10 = ssa.add_const(Constant::U(32, 10));
    let (v, used, lt, v1) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    let header = f.add_block();
    let body = f.add_block();
    let after = f.add_block();
    f.get_entry_mut()
        .set_terminator(Terminator::Jmp(header, vec![c0]));
    let header_block = f.get_block_mut(header);
    header_block.push_parameter(v, Type::u(32));
    header_block.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: used,
            lhs: v,
            rhs: c1,
        },
        SourceLocation::test(),
    ));
    header_block.push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Lt,
            result: lt,
            lhs: v,
            rhs: c10,
        },
        SourceLocation::test(),
    ));
    header_block.set_terminator(Terminator::JmpIf(lt, body, after));
    let body_block = f.get_block_mut(body);
    body_block.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: v1,
            lhs: v,
            rhs: c1,
        },
        SourceLocation::test(),
    ));
    body_block.set_terminator(Terminator::Jmp(header, vec![v1]));
    f.get_block_mut(after).push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: v,
            rhs: c10,
        },
        SourceLocation::test(),
    ));
    f.get_block_mut(after)
        .set_terminator(Terminator::Return(vec![used]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    // The in-loop use keeps the genuine loop-carried `v` — no anticipated substitution fired.
    assert!(matches!(
        f.get_block(header).get_instructions().next(),
        Some(OpCode::BinaryArithOp { lhs, .. }) if *lhs == v
    ));
    // And the assert keeps checking the genuine final value.
    assert!(matches!(
        f.get_block(after).get_instructions().next(),
        Some(OpCode::AssertCmp { kind: CmpKind::Eq, lhs, .. }) if *lhs == v
    ));
}

/// The binding-finality half of the stability gate end-to-end: a loop-carried accumulator
/// asserted after its loop _is_ rewritten in the post-loop region (no path from there re-enters
/// the loop, so the binding is the final one the assert checks), while the justifying assert
/// keeps its genuine operand.
#[test]
fn anticipated_const_folds_post_loop_use() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c0 = ssa.add_const(Constant::U(32, 0));
    let c1 = ssa.add_const(Constant::U(32, 1));
    let c10 = ssa.add_const(Constant::U(32, 10));
    let (n, v, lt, v1, r) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    let header = f.add_block();
    let body = f.add_block();
    let after = f.add_block();
    let tail = f.add_block();
    f.get_entry_mut().push_parameter(n, Type::u(32));
    f.get_entry_mut()
        .set_terminator(Terminator::Jmp(header, vec![c0]));
    let header_block = f.get_block_mut(header);
    header_block.push_parameter(v, Type::u(32));
    header_block.push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Lt,
            result: lt,
            lhs: v,
            rhs: c10,
        },
        SourceLocation::test(),
    ));
    header_block.set_terminator(Terminator::JmpIf(lt, body, after));
    let body_block = f.get_block_mut(body);
    body_block.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: v1,
            lhs: v,
            rhs: c1,
        },
        SourceLocation::test(),
    ));
    body_block.set_terminator(Terminator::Jmp(header, vec![v1]));
    f.get_block_mut(after)
        .push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: r,
                lhs: v,
                rhs: n,
            },
            SourceLocation::test(),
        ));
    f.get_block_mut(after)
        .set_terminator(Terminator::Jmp(tail, vec![]));
    f.get_block_mut(tail).push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: v,
            rhs: c10,
        },
        SourceLocation::test(),
    ));
    f.get_block_mut(tail)
        .set_terminator(Terminator::Return(vec![r]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    // The post-loop use takes the pinned constant: `r = 10 + n`.
    assert!(matches!(
        f.get_block(after).get_instructions().next(),
        Some(OpCode::BinaryArithOp { lhs, .. }) if *lhs == c10
    ));
    // The in-loop use keeps the genuine loop-carried `v`...
    assert!(matches!(
        f.get_block(body).get_instructions().next(),
        Some(OpCode::BinaryArithOp { lhs, .. }) if *lhs == v
    ));
    // ...and so does the justifying assert (Gate 3).
    assert!(matches!(
        f.get_block(tail).get_instructions().next(),
        Some(OpCode::AssertCmp { kind: CmpKind::Eq, lhs, .. }) if *lhs == v
    ));
}

/// Gate 3 for the opaque-assert shape (`rewrite_asserts` leaves `Assert{w}` over a boolean
/// chain untouched): the anticipated channel folds `w = Cmp{Eq, a, b}` — proven by the later
/// `AssertCmp{Eq, a, b}` — but the fold must not reach the `Assert`'s _input_, and the `Cmp`
/// must stay alive for it. The union leader table does redirect the `Cmp`'s own operands
/// (`b → a`, the before-the-assert direction), so the opaque re-check weakens to a tautology
/// and may collapse on a later pass — sound _because_ the establishing `AssertCmp` is what Gate
/// 3 keeps genuine: the constraint's earliest surviving check is never weakened, so the
/// acceptance set is unchanged.
#[test]
fn bare_assert_over_cmp_chain_protected() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (a, b, w) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut().push_parameter(a, Type::u(32));
    f.get_entry_mut().push_parameter(b, Type::u(32));
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Eq,
            result: w,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    f.get_entry_mut()
        .push_instruction(Located::with(
            OpCode::Assert { value: w },
            SourceLocation::test(),
        ));
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    f.get_entry_mut().set_terminator(Terminator::Return(vec![]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    let entry = f.get_entry();
    // The anticipated channel must not fold the `Assert`'s input: it still references the
    // live comparison, not a constant.
    assert!(
        entry
            .get_instructions()
            .any(|i| matches!(i, OpCode::Assert { value } if *value == w)),
        "the opaque `Assert` must keep referencing the comparison"
    );
    assert!(
        entry.get_instructions().any(|i| matches!(
            i,
            OpCode::Cmp { kind: CmpKind::Eq, result, .. } if *result == w
        )),
        "the comparison feeding the opaque assert must stay alive"
    );

    fold(&mut ssa);

    // Whatever the redundant opaque chain collapses to, the establishing equality assert —
    // the fact's earliest surviving check — must remain on its genuine operands.
    let f = ssa.get_unique_entrypoint();
    assert!(
        f.get_entry().get_instructions().any(|i| matches!(
            i,
            OpCode::AssertCmp { kind: CmpKind::Eq, lhs, rhs } if *lhs == a && *rhs == b
        )),
        "the equality assert survives untouched"
    );
}

/// The witness-typed analog of `anticipated_cmp_fold_decides_branch`'s fold: a `WitnessOf`
/// `Cmp{Eq}` proven by a _bound-to-run_ assert takes the keep-compatible recast — the `Cmp` is
/// _kept_ (contrast the dominance path, which replaces it in place) and the use is redirected
/// to a `cast true to WitnessOf` hoisted into the entry block, while the justifying assert
/// keeps its genuine operands.
#[test]
fn anticipated_witnessed_cmp_fold_redirects_use_via_hoisted_cast() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c_true = ssa.add_const(Constant::U(1, 1));
    let (a, b, ww) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut()
        .push_parameter(a, Type::witness_of(Type::field()));
    f.get_entry_mut()
        .push_parameter(b, Type::witness_of(Type::field()));
    let entry = f.get_entry_mut();

    // A witness operand makes the result `WitnessOf(u1)`; the assert comes _after_, so only
    // the anticipated ("bound to run") direction proves the fold.
    entry.push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Eq,
            result: ww,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![ww]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    let entry = f.get_entry();
    // The comparison is kept: an excluded assert could still reference its result.
    assert!(
        entry.get_instructions().any(|i| matches!(
            i,
            OpCode::Cmp { kind: CmpKind::Eq, result, .. } if *result == ww
        )),
        "the folded witnessed comparison must be kept"
    );
    // A fresh `cast true to WitnessOf` was hoisted into the entry block...
    let cast_result = entry
        .get_instructions()
        .find_map(|i| match i {
            OpCode::Cast {
                result,
                value,
                target: CastTarget::WitnessOf,
            } if *value == c_true => Some(*result),
            _ => None,
        })
        .expect("a witness cast of `true` should be hoisted into the entry block");
    assert_ne!(cast_result, ww, "the recast must not redefine the kept Cmp");
    // ...and the pure use is redirected to it.
    assert!(matches!(
        entry.get_terminator(),
        Some(Terminator::Return(vals)) if *vals == vec![cast_result]
    ));
    // The justifying assert keeps its genuine operands.
    assert!(matches!(
        entry.get_instructions().find(|i| i.is_assert()),
        Some(OpCode::AssertCmp { kind: CmpKind::Eq, lhs, rhs }) if *lhs == a && *rhs == b
    ));
}

/// Gate 3 for the witness fold, via the allowlist: an opaque `Assert{ww}` over the witnessed
/// comparison is not an `is_witness_subst_safe_target`, so it keeps referencing the genuine
/// result and the `Cmp` stays alive for it.
#[test]
fn anticipated_witnessed_cmp_fold_keeps_excluded_assert_use() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (a, b, ww) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut()
        .push_parameter(a, Type::witness_of(Type::field()));
    f.get_entry_mut()
        .push_parameter(b, Type::witness_of(Type::field()));
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Eq,
            result: ww,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::Assert { value: ww },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    let entry = f.get_entry();
    assert!(
        entry
            .get_instructions()
            .any(|i| matches!(i, OpCode::Assert { value } if *value == ww)),
        "the opaque `Assert` must keep referencing the genuine comparison"
    );
    assert!(
        entry.get_instructions().any(|i| matches!(
            i,
            OpCode::Cmp { kind: CmpKind::Eq, result, .. } if *result == ww
        )),
        "the comparison feeding the opaque assert must stay alive"
    );
    assert!(
        entry.get_instructions().any(|i| matches!(
            i,
            OpCode::AssertCmp { kind: CmpKind::Eq, lhs, rhs } if *lhs == a && *rhs == b
        )),
        "the establishing equality assert keeps its genuine operands"
    );
}

/// The witness fold's cross-block direction: the establishing assert sits in a strictly
/// _post-dominating_ block, and the redirected consumer is an allowlisted instruction (a Select
/// condition), not a terminator.
#[test]
fn anticipated_witnessed_cmp_fold_from_post_dominating_assert() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c_true = ssa.add_const(Constant::U(1, 1));
    let (a, b, ww, s) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    let tail = f.add_block();
    f.get_entry_mut()
        .push_parameter(a, Type::witness_of(Type::field()));
    f.get_entry_mut()
        .push_parameter(b, Type::witness_of(Type::field()));
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Eq,
            result: ww,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::Select {
            result: s,
            cond: ww,
            if_t: a,
            if_f: b,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Jmp(tail, vec![]));
    f.get_block_mut(tail).push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    f.get_block_mut(tail)
        .set_terminator(Terminator::Return(vec![s]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    let entry = f.get_entry();
    let cast_result = entry
        .get_instructions()
        .find_map(|i| match i {
            OpCode::Cast {
                result,
                value,
                target: CastTarget::WitnessOf,
            } if *value == c_true => Some(*result),
            _ => None,
        })
        .expect("a witness cast of `true` should be hoisted into the entry block");
    // The allowlisted `Select` condition is redirected to the recast constant.
    assert!(
        entry.get_instructions().any(|i| matches!(
            i,
            OpCode::Select { cond, .. } if *cond == cast_result
        )),
        "the Select condition should be redirected to the hoisted cast"
    );
    // The justifying assert in the post-dominating block keeps its genuine operands.
    assert!(matches!(
        f.get_block(tail).get_instructions().next(),
        Some(OpCode::AssertCmp { kind: CmpKind::Eq, lhs, rhs }) if *lhs == a && *rhs == b
    ));
}

/// The witness fold and the witness-typed asserted-_constant_ substitution share one hoisted
/// cast per constant: both channels pin `true` here, and exactly one `cast true to WitnessOf`
/// serves both redirected uses.
#[test]
fn anticipated_witness_cast_shared_with_asserted_const_channel() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c_true = ssa.add_const(Constant::U(1, 1));
    let (a, b, w, ww) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut()
        .push_parameter(a, Type::witness_of(Type::field()));
    f.get_entry_mut()
        .push_parameter(b, Type::witness_of(Type::field()));
    f.get_entry_mut()
        .push_parameter(w, Type::witness_of(Type::u(1)));
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Eq,
            result: ww,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    // Pins `w` to `true` for the dominance-channel constant substitution in the return.
    entry.push_instruction(Located::with(
        OpCode::Assert { value: w },
        SourceLocation::test(),
    ));
    // Proves the fold of `ww` via the anticipated channel.
    entry.push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![ww, w]));

    fold(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    let entry = f.get_entry();
    let casts: Vec<ValueId> = entry
        .get_instructions()
        .filter_map(|i| match i {
            OpCode::Cast {
                result,
                value,
                target: CastTarget::WitnessOf,
            } if *value == c_true => Some(*result),
            _ => None,
        })
        .collect();
    assert_eq!(
        casts.len(),
        1,
        "both channels must share a single hoisted cast per constant"
    );
    let shared = casts[0];
    assert!(matches!(
        entry.get_terminator(),
        Some(Terminator::Return(vals)) if *vals == vec![shared, shared]
    ));
}

/// When every rewritable use of the folded witnessed comparison is excluded (only the opaque
/// assert references it), the speculatively hoisted cast has no uses and the integrated DCE
/// sweeps it in the same pass — while the assert-liveness chain keeps the `Cmp` and both
/// asserts.
#[test]
fn anticipated_witness_cast_unused_is_swept_by_dce() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (a, b, ww) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut()
        .push_parameter(a, Type::witness_of(Type::field()));
    f.get_entry_mut()
        .push_parameter(b, Type::witness_of(Type::field()));
    let entry = f.get_entry_mut();
    entry.push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Eq,
            result: ww,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::Assert { value: ww },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![]));

    fold_and_dce(&mut ssa);

    let f = ssa.get_unique_entrypoint();
    let entry = f.get_entry();
    assert!(
        !entry.get_instructions().any(|i| matches!(
            i,
            OpCode::Cast {
                target: CastTarget::WitnessOf,
                ..
            }
        )),
        "the unused hoisted cast must be swept by the integrated DCE"
    );
    assert!(
        entry.get_instructions().any(|i| matches!(
            i,
            OpCode::Cmp { kind: CmpKind::Eq, result, .. } if *result == ww
        )) && entry
            .get_instructions()
            .any(|i| matches!(i, OpCode::Assert { value } if *value == ww))
            && entry.get_instructions().any(|i| matches!(
                i,
                OpCode::AssertCmp { kind: CmpKind::Eq, lhs, rhs } if *lhs == a && *rhs == b
            )),
        "the comparison and both asserts must survive"
    );
}
