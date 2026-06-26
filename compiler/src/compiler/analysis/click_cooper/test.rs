use super::{ClickCooper, Context, FlowAnalysis};
use crate::compiler::{
    Field,
    analysis::types::Types,
    ssa::{
        Terminator,
        hlssa::{
            BinaryArithOpKind, Blob, CallTarget, CastTarget, CmpKind, Constant, HLSSA, OpCode,
            ScalarFold, SequenceTargetType, SliceOpDir, Type,
        },
    },
};

/// Build the analysis for `ssa` with freshly-computed dependencies, for test use only.
pub(crate) fn run_in_test(ssa: &HLSSA) -> ClickCooper {
    let flow = FlowAnalysis::run(ssa);
    let types = Types::new().run(ssa, &flow);
    ClickCooper::run(ssa, &flow, &types)
}

/// `OpCode::scalar_fold` is the single source of truth for "foldable *scalar* op", and value
/// numbering is a strict superset of it.
///
/// This locks the projections that read both: `is_pure_scalar_fold` agrees with
/// `scalar_fold().is_some()`; value numbering (`pure_op_operands`, backed by `op_signature`)
/// fires on every foldable scalar op *except* witness casts, **and additionally** on the pure
/// sequence ops, which are not scalar-foldable. Both deliberate asymmetries are asserted below
/// so neither can be silently dropped.
#[test]
fn scalar_fold_is_the_single_classifier() {
    use super::congruence::pure_op_operands;

    let ssa = HLSSA::with_main("main".to_string());
    let main = ssa.get_unique_entrypoint_id();
    let v = || ssa.fresh_value();

    // One instance of every foldable scalar op, each paired with the `ScalarFold` variant it
    // must decompose to.
    let foldable: Vec<OpCode> = vec![
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: v(),
            lhs: v(),
            rhs: v(),
        },
        OpCode::Cmp {
            kind: CmpKind::Eq,
            result: v(),
            lhs: v(),
            rhs: v(),
        },
        OpCode::MulConst {
            result: v(),
            const_val: v(),
            var: v(),
        },
        OpCode::Cast {
            result: v(),
            value: v(),
            target: CastTarget::Field,
        },
        OpCode::SExt {
            result: v(),
            value: v(),
            from_bits: 8,
            to_bits: 32,
        },
        OpCode::BitRange {
            result: v(),
            value: v(),
            offset: 0,
            width: 8,
        },
        OpCode::Not {
            result: v(),
            value: v(),
        },
        OpCode::Select {
            result: v(),
            cond: v(),
            if_t: v(),
            if_f: v(),
        },
    ];
    for instr in &foldable {
        assert!(instr.is_pure_scalar_fold(), "{instr:?} should be foldable");
        assert_eq!(
            instr.is_pure_scalar_fold(),
            instr.scalar_fold().is_some(),
            "is_pure_scalar_fold must equal scalar_fold().is_some() for {instr:?}"
        );
        // Every foldable op except a witness cast is value-numbered.
        assert!(
            pure_op_operands(instr).is_some(),
            "{instr:?} should be value-numbered"
        );
    }

    // The deliberate asymmetry: a witness cast IS a scalar fold but is NOT value-numbered.
    let witness_cast = OpCode::Cast {
        result: v(),
        value: v(),
        target: CastTarget::WitnessOf,
    };
    assert!(witness_cast.is_pure_scalar_fold());
    assert!(matches!(
        witness_cast.scalar_fold(),
        Some(ScalarFold::Cast { .. })
    ));
    assert!(pure_op_operands(&witness_cast).is_none());

    // The other deliberate asymmetry: pure sequence ops ARE value-numbered but are NOT
    // scalar-foldable (their aggregate results never enter the constant lattice).
    let elem = Type::field();
    let sequence_ops: Vec<OpCode> = vec![
        OpCode::ArrayGet {
            result: v(),
            array: v(),
            index: v(),
        },
        OpCode::ArraySet {
            result: v(),
            array: v(),
            index: v(),
            value: v(),
        },
        OpCode::SliceLen {
            result: v(),
            slice: v(),
        },
        OpCode::MkSeq {
            result: v(),
            elems: vec![v(), v()],
            seq_type: SequenceTargetType::Array(2),
            elem_type: elem.clone(),
        },
        OpCode::MkRepeated {
            result: v(),
            element: v(),
            seq_type: SequenceTargetType::Slice,
            count: 3,
            elem_type: elem.clone(),
        },
        OpCode::MkSeqOfBlob {
            result: v(),
            element_type: elem.clone(),
            blob: v(),
        },
        OpCode::SlicePush {
            dir: SliceOpDir::Back,
            result: v(),
            slice: v(),
            values: vec![v()],
        },
    ];
    for instr in &sequence_ops {
        assert!(
            !instr.is_pure_scalar_fold(),
            "{instr:?} should not be scalar-foldable"
        );
        assert!(instr.scalar_fold().is_none());
        assert!(
            pure_op_operands(instr).is_some(),
            "{instr:?} should be value-numbered"
        );
    }

    // Representative non-folds: not foldable, not value-numbered, no `ScalarFold`.
    let non_folds: Vec<OpCode> = vec![
        OpCode::Assert { value: v() },
        OpCode::Store {
            ptr: v(),
            value: v(),
        },
        OpCode::Load {
            result: v(),
            ptr: v(),
        },
        OpCode::Call {
            results: vec![v()],
            function: CallTarget::Static(main),
            args: vec![v()],
            unconstrained: false,
        },
    ];
    for instr in &non_folds {
        assert!(
            !instr.is_pure_scalar_fold(),
            "{instr:?} should not be foldable"
        );
        assert!(instr.scalar_fold().is_none());
        assert!(
            pure_op_operands(instr).is_none(),
            "{instr:?} should not be value-numbered"
        );
    }
}

/// Two values that fold to the *same* constant are congruent, even when computed differently —
/// the const → congruence coupling.
#[test]
fn equal_constants_are_congruent() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c1 = ssa.add_const(Constant::U(32, 1));
    let c2 = ssa.add_const(Constant::U(32, 2));
    let c3 = ssa.add_const(Constant::U(32, 3));
    let c4 = ssa.add_const(Constant::U(32, 4));
    let (a, b) = (ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    let entry = f.get_entry_mut();
    entry.push_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Add,
        result: a,
        lhs: c2,
        rhs: c3,
    });
    entry.push_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Add,
        result: b,
        lhs: c1,
        rhs: c4,
    });
    entry.set_terminator(Terminator::Return(vec![a, b]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    // 2+3 and 1+4 both equal 5.
    assert!(cc.known_equal(fid, a, b));
    // But not congruent to a different constant.
    assert!(!cc.known_equal(fid, a, c2));
}

/// The same expression over the same operands is congruent, commutatively; different operators
/// or operands are not.
#[test]
fn structural_congruence_is_commutative() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (x, y) = (ssa.fresh_value(), ssa.fresh_value());
    let (a, b, c, d) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut().push_parameter(x, Type::u(32));
    f.get_entry_mut().push_parameter(y, Type::u(32));
    let entry = f.get_entry_mut();
    entry.push_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Add,
        result: a,
        lhs: x,
        rhs: y,
    });
    entry.push_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Add,
        result: b,
        lhs: x,
        rhs: y,
    });
    entry.push_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Add,
        result: c,
        lhs: y,
        rhs: x,
    });
    entry.push_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Sub,
        result: d,
        lhs: x,
        rhs: y,
    });
    entry.set_terminator(Terminator::Return(vec![a, b, c, d]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    assert!(cc.known_equal(fid, a, b));
    assert!(cc.known_equal(fid, a, c)); // x+y ≡ y+x
    assert!(!cc.known_equal(fid, a, d)); // x+y ≢ x-y
    assert!(!cc.known_equal(fid, x, y)); // distinct opaque values

    let mut expected = vec![a, b, c];
    expected.sort();
    assert_eq!(cc.congruence_class(fid, a), expected);
}

/// Pure sequence ops are value-numbered: two `ArrayGet`s of the same array at the same index
/// are congruent (a different index is not), `MkSeq`s congruent elementwise are congruent, and
/// the sequence shape (`seq_type`, `elem_type`) carried in the key keeps differently-shaped
/// `MkSeq`s apart.
#[test]
fn array_ops_are_value_numbered() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (arr, i, j) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let (g1, g2, g3) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let (s1, s2, s3, s4) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut()
        .push_parameter(arr, Type::field().array_of(4));
    f.get_entry_mut().push_parameter(i, Type::u(32));
    f.get_entry_mut().push_parameter(j, Type::u(32));
    let entry = f.get_entry_mut();
    // g1, g2 are arr[i]; g3 is arr[j].
    entry.push_instruction(OpCode::ArrayGet {
        result: g1,
        array: arr,
        index: i,
    });
    entry.push_instruction(OpCode::ArrayGet {
        result: g2,
        array: arr,
        index: i,
    });
    entry.push_instruction(OpCode::ArrayGet {
        result: g3,
        array: arr,
        index: j,
    });
    // s1 and s2 are the same array `[arr[i], arr[j]]` built twice — congruent elementwise.
    entry.push_instruction(OpCode::MkSeq {
        result: s1,
        elems: vec![g1, g3],
        seq_type: SequenceTargetType::Array(2),
        elem_type: Type::field(),
    });
    entry.push_instruction(OpCode::MkSeq {
        result: s2,
        elems: vec![g2, g3],
        seq_type: SequenceTargetType::Array(2),
        elem_type: Type::field(),
    });
    // s3 differs only in sequence kind (Slice), s4 only in element type (u32): neither merges.
    entry.push_instruction(OpCode::MkSeq {
        result: s3,
        elems: vec![g1, g3],
        seq_type: SequenceTargetType::Slice,
        elem_type: Type::field(),
    });
    entry.push_instruction(OpCode::MkSeq {
        result: s4,
        elems: vec![g1, g3],
        seq_type: SequenceTargetType::Array(2),
        elem_type: Type::u(32),
    });
    entry.set_terminator(Terminator::Return(vec![g1, g2, g3, s1, s2, s3, s4]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    assert!(cc.known_equal(fid, g1, g2)); // arr[i] ≡ arr[i]
    assert!(!cc.known_equal(fid, g1, g3)); // arr[i] ≢ arr[j]
    assert!(cc.known_equal(fid, s1, s2)); // elementwise-congruent arrays
    assert!(!cc.known_equal(fid, s1, s3)); // different seq_type
    assert!(!cc.known_equal(fid, s1, s4)); // different elem_type
}

/// φ-operands come from *executable* edges only: a parameter whose value differs solely on a
/// dead in-edge is still congruent to one that agrees on the live edge.
#[test]
fn phi_congruence_excludes_dead_edges() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c_true = ssa.add_const(Constant::U(1, 1));
    let (x, y, p, q) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    let live = f.add_block();
    let dead = f.add_block();
    let merge = f.add_block();
    f.get_entry_mut().push_parameter(x, Type::field());
    f.get_entry_mut().push_parameter(y, Type::field());
    f.get_entry_mut()
        .set_terminator(Terminator::JmpIf(c_true, live, dead));
    // Live edge agrees on both params; the dead edge would have forced them apart.
    f.get_block_mut(live)
        .set_terminator(Terminator::Jmp(merge, vec![x, x]));
    f.get_block_mut(dead)
        .set_terminator(Terminator::Jmp(merge, vec![x, y]));
    let merge_block = f.get_block_mut(merge);
    merge_block.push_parameter(p, Type::field());
    merge_block.push_parameter(q, Type::field());
    merge_block.set_terminator(Terminator::Return(vec![p, q]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    assert!(cc.known_equal(fid, p, q));
}

/// The same merge with *both* edges live keeps the parameters apart — congruence is genuinely
/// reachability-sensitive.
#[test]
fn phi_distinguished_when_both_edges_live() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (cond, x, y, p, q) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    let e1 = f.add_block();
    let e2 = f.add_block();
    let merge = f.add_block();
    f.get_entry_mut().push_parameter(cond, Type::u(1));
    f.get_entry_mut().push_parameter(x, Type::field());
    f.get_entry_mut().push_parameter(y, Type::field());
    f.get_entry_mut()
        .set_terminator(Terminator::JmpIf(cond, e1, e2));
    f.get_block_mut(e1)
        .set_terminator(Terminator::Jmp(merge, vec![x, x]));
    f.get_block_mut(e2)
        .set_terminator(Terminator::Jmp(merge, vec![x, y]));
    let merge_block = f.get_block_mut(merge);
    merge_block.push_parameter(p, Type::field());
    merge_block.push_parameter(q, Type::field());
    merge_block.set_terminator(Terminator::Return(vec![p, q]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    assert!(!cc.known_equal(fid, p, q));
}

/// The optimistic win pessimistic value numbering cannot reach: two parallel induction variables
/// that start equal and step identically are congruent across the loop back-edge.
#[test]
fn loop_carried_parallel_induction_is_congruent() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c0 = ssa.add_const(Constant::U(32, 0));
    let c1 = ssa.add_const(Constant::U(32, 1));
    let c10 = ssa.add_const(Constant::U(32, 10));
    let (i, j, lt, i2, j2) = (
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
    let header_block = f.get_block_mut(header);
    header_block.push_parameter(i, Type::u(32));
    header_block.push_parameter(j, Type::u(32));
    header_block.push_instruction(OpCode::Cmp {
        kind: CmpKind::Lt,
        result: lt,
        lhs: i,
        rhs: c10,
    });
    header_block.set_terminator(Terminator::JmpIf(lt, body, exit));
    let body_block = f.get_block_mut(body);
    body_block.push_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Add,
        result: i2,
        lhs: i,
        rhs: c1,
    });
    body_block.push_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Add,
        result: j2,
        lhs: j,
        rhs: c1,
    });
    body_block.set_terminator(Terminator::Jmp(header, vec![i2, j2]));
    f.get_block_mut(exit)
        .set_terminator(Terminator::Return(vec![i, j]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    assert!(cc.known_equal(fid, i, j)); // loop-carried congruence
    assert!(cc.known_equal(fid, i2, j2)); // and their parallel updates

    // The dominance-aware leader is the member at the dominating definition: the two header
    // parameters share a definition site (block entry), so the lower-id one leads; in the body,
    // the earlier `i2` leads the later `j2`.
    assert_eq!(cc.leader(fid, i), Some(i));
    assert_eq!(cc.leader(fid, j), Some(i));
    assert_eq!(cc.leader(fid, i2), Some(i2));
    assert_eq!(cc.leader(fid, j2), Some(i2));
}

/// A congruent definition that dominates another is its leader (a legal redirect target); the
/// dominated occurrence redirects to the dominating one, and the dominating one leads itself.
#[test]
fn leader_is_dominating_definition() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (x, y, a, b) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    let next = f.add_block();
    f.get_entry_mut().push_parameter(x, Type::field());
    f.get_entry_mut().push_parameter(y, Type::field());
    // `a = x + y` in the entry, recomputed as `b = x + y` in a strictly dominated block.
    f.get_entry_mut().push_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Add,
        result: a,
        lhs: x,
        rhs: y,
    });
    f.get_entry_mut()
        .set_terminator(Terminator::Jmp(next, vec![]));
    let next_block = f.get_block_mut(next);
    next_block.push_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Add,
        result: b,
        lhs: x,
        rhs: y,
    });
    next_block.set_terminator(Terminator::Return(vec![a, b]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    assert!(cc.known_equal(fid, a, b));
    assert_eq!(cc.leader(fid, a), Some(a)); // dominating definition leads itself
    assert_eq!(cc.leader(fid, b), Some(a)); // dominated occurrence redirects to it
}

/// Two congruent ops in one block: the earlier one (by instruction index) leads the later.
#[test]
fn leader_is_earlier_in_block() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (x, y, a, b) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut().push_parameter(x, Type::field());
    f.get_entry_mut().push_parameter(y, Type::field());
    let entry = f.get_entry_mut();
    entry.push_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Add,
        result: a,
        lhs: x,
        rhs: y,
    });
    entry.push_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Add,
        result: b,
        lhs: x,
        rhs: y,
    });
    entry.set_terminator(Terminator::Return(vec![a, b]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    assert!(cc.known_equal(fid, a, b));
    assert_eq!(cc.leader(fid, a), Some(a));
    assert_eq!(cc.leader(fid, b), Some(a));
}

/// Congruent occurrences in two incomparable branches (and one at the merge) have no single
/// dominating member, so each is its own leader — the leader is never a non-dominating member,
/// so no illegal cross-branch redirect is offered.
#[test]
fn leader_never_crosses_incomparable_branches() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (cond, x, y, a, b, c) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    let e1 = f.add_block();
    let e2 = f.add_block();
    let merge = f.add_block();
    f.get_entry_mut().push_parameter(cond, Type::u(1));
    f.get_entry_mut().push_parameter(x, Type::field());
    f.get_entry_mut().push_parameter(y, Type::field());
    f.get_entry_mut()
        .set_terminator(Terminator::JmpIf(cond, e1, e2));
    // `x + y` recomputed in both incomparable branches and again at the merge.
    let e1_block = f.get_block_mut(e1);
    e1_block.push_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Add,
        result: a,
        lhs: x,
        rhs: y,
    });
    e1_block.set_terminator(Terminator::Jmp(merge, vec![]));
    let e2_block = f.get_block_mut(e2);
    e2_block.push_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Add,
        result: b,
        lhs: x,
        rhs: y,
    });
    e2_block.set_terminator(Terminator::Jmp(merge, vec![]));
    let merge_block = f.get_block_mut(merge);
    merge_block.push_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Add,
        result: c,
        lhs: x,
        rhs: y,
    });
    merge_block.set_terminator(Terminator::Return(vec![a, b, c]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    // All three are congruent...
    assert!(cc.known_equal(fid, a, b));
    assert!(cc.known_equal(fid, a, c));
    // ...but none dominates another, so each leads itself (no cross-branch redirect).
    assert_eq!(cc.leader(fid, a), Some(a));
    assert_eq!(cc.leader(fid, b), Some(b));
    assert_eq!(cc.leader(fid, c), Some(c));
}

/// A computed value proven constant is congruent to the interned constant of the same value, and
/// that constant — available throughout the function — is its leader.
#[test]
fn leader_of_constant_class_is_the_interned_constant() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c2 = ssa.add_const(Constant::U(32, 2));
    let c3 = ssa.add_const(Constant::U(32, 3));
    let c5 = ssa.add_const(Constant::U(32, 5));
    let a = ssa.fresh_value();

    let f = ssa.get_unique_entrypoint_mut();
    let entry = f.get_entry_mut();
    // `a = 2 + 3` folds to 5, congruent to the interned constant `c5`.
    entry.push_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Add,
        result: a,
        lhs: c2,
        rhs: c3,
    });
    entry.set_terminator(Terminator::Return(vec![a, c5]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    assert!(cc.known_equal(fid, a, c5));
    assert_eq!(cc.leader(fid, a), Some(c5)); // the constant dominates everywhere
    assert_eq!(cc.leader(fid, c5), Some(c5));
}

/// Assert-vacuum soundness: `assert(x == 5)` pins `x` to 5 *conditionally* in dominated blocks,
/// but never unconditionally — so a global fold (SCCP) can't fold `x` and vacuum the assert.
#[test]
fn assert_eq_const_is_conditional_not_unconditional() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c5 = ssa.add_const(Constant::U(32, 5));
    let x = ssa.fresh_value();

    let f = ssa.get_unique_entrypoint_mut();
    let entry_id = f.get_entry_id();
    let after = f.add_block();
    f.get_entry_mut().push_parameter(x, Type::u(32));
    f.get_entry_mut().push_instruction(OpCode::AssertCmp {
        kind: CmpKind::Eq,
        lhs: x,
        rhs: c5,
    });
    f.get_entry_mut()
        .set_terminator(Terminator::Jmp(after, vec![]));
    f.get_block_mut(after)
        .set_terminator(Terminator::Return(vec![]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);

    // Unconditionally `x` is unknown — the assert never enters the global lattice.
    assert_eq!(cc.const_of(fid, x), None);
    assert!(cc.new_const_values(fid).iter().all(|(v, _)| *v != x));
    // Conditionally `x == 5` at every block the assert *strictly* dominates ...
    assert_eq!(
        cc.asserted_const(fid, after, x).as_deref(),
        Some(&Constant::U(32, 5))
    );
    // ... but not in the asserting block itself (block-entry granularity).
    assert_eq!(cc.asserted_const(fid, entry_id, x), None);
}

/// `assert(b)` proves `b == true` conditionally in dominated blocks, never unconditionally.
#[test]
fn assert_bool_is_conditional() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let b = ssa.fresh_value();

    let f = ssa.get_unique_entrypoint_mut();
    let after = f.add_block();
    f.get_entry_mut().push_parameter(b, Type::u(1));
    f.get_entry_mut()
        .push_instruction(OpCode::Assert { value: b });
    f.get_entry_mut()
        .set_terminator(Terminator::Jmp(after, vec![]));
    f.get_block_mut(after)
        .set_terminator(Terminator::Return(vec![]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    assert_eq!(cc.const_of(fid, b), None);
    assert_eq!(
        cc.asserted_const(fid, after, b).as_deref(),
        Some(&Constant::U(1, 1))
    );
}

/// `assert(x == y)` with neither side constant records a *conditional* equality — distinct from
/// structural congruence, which (unconditionally) keeps the two parameters apart.
#[test]
fn assert_eq_pure_equality_is_conditional() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (x, y) = (ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    let entry_id = f.get_entry_id();
    let after = f.add_block();
    f.get_entry_mut().push_parameter(x, Type::u(32));
    f.get_entry_mut().push_parameter(y, Type::u(32));
    f.get_entry_mut().push_instruction(OpCode::AssertCmp {
        kind: CmpKind::Eq,
        lhs: x,
        rhs: y,
    });
    f.get_entry_mut()
        .set_terminator(Terminator::Jmp(after, vec![]));
    f.get_block_mut(after)
        .set_terminator(Terminator::Return(vec![]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    assert!(cc.asserted_equal(fid, after, x, y));
    assert!(cc.asserted_equal(fid, after, y, x)); // symmetric
    assert!(!cc.asserted_equal(fid, entry_id, x, y)); // not in the asserting block
    // Structural congruence (unconditional) does not see the assert.
    assert!(!cc.known_equal(fid, x, y));
}

/// The false edge of `if x == y` proves `x != y` at its target and the blocks that target
/// dominates — and nowhere else.
#[test]
fn disequality_from_false_edge() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (x, y, eq) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    let entry_id = f.get_entry_id();
    let then_b = f.add_block();
    let else_b = f.add_block();
    let after = f.add_block();
    f.get_entry_mut().push_parameter(x, Type::u(32));
    f.get_entry_mut().push_parameter(y, Type::u(32));
    f.get_entry_mut().push_instruction(OpCode::Cmp {
        kind: CmpKind::Eq,
        result: eq,
        lhs: x,
        rhs: y,
    });
    f.get_entry_mut()
        .set_terminator(Terminator::JmpIf(eq, then_b, else_b));
    f.get_block_mut(then_b)
        .set_terminator(Terminator::Return(vec![]));
    f.get_block_mut(else_b)
        .set_terminator(Terminator::Jmp(after, vec![]));
    f.get_block_mut(after)
        .set_terminator(Terminator::Return(vec![]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    // x != y on the false-edge target and blocks it dominates.
    assert!(cc.known_unequal(fid, else_b, x, y));
    assert!(cc.known_unequal(fid, else_b, y, x)); // symmetric
    assert!(cc.known_unequal(fid, after, x, y));
    // Not on the true edge, nor before the branch.
    assert!(!cc.known_unequal(fid, then_b, x, y));
    assert!(!cc.known_unequal(fid, entry_id, x, y));
}

/// An unpinned witness write forwards `r → v`; distinct witnesses get distinct forwards and are
/// never congruent; a *pinned* write (a real constraint) forwards nothing.
#[test]
fn unpinned_witness_forwards_distinct_not_merged() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (v1, v2) = (ssa.fresh_value(), ssa.fresh_value());
    let (r1, r2, rp) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut().push_parameter(v1, Type::u(32));
    f.get_entry_mut().push_parameter(v2, Type::u(32));
    let entry = f.get_entry_mut();
    entry.push_instruction(OpCode::WriteWitness {
        result: Some(r1),
        value: v1,
        pinned: false,
    });
    entry.push_instruction(OpCode::WriteWitness {
        result: Some(r2),
        value: v2,
        pinned: false,
    });
    entry.push_instruction(OpCode::WriteWitness {
        result: Some(rp),
        value: v1,
        pinned: true,
    });
    entry.set_terminator(Terminator::Return(vec![]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    assert_eq!(cc.witness_forward(fid, r1), [v1].as_slice());
    assert_eq!(cc.witness_forward(fid, r2), [v2].as_slice());
    // A pinned write carries a real `r == v` constraint — it is not a free witness, no forward.
    assert!(cc.witness_forward(fid, rp).is_empty());
    // Two distinct free witnesses are never unified (the hard non-merge prohibition).
    assert!(!cc.known_equal(fid, r1, r2));
}

/// `witness_forward` is the sorted union of *both* readings of the one witness↔value
/// correspondence: the `WriteWitness` hint (`r → v`) and every `value_of(r)` read (`r → w`).
/// Distinct witnesses keep disjoint sets — no cross-witness union.
#[test]
fn witness_forward_unions_hint_and_value_of_reads() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (v, v2) = (ssa.fresh_value(), ssa.fresh_value()); // honest hint payloads
    let (r, r2) = (ssa.fresh_value(), ssa.fresh_value()); // witness handles
    let (w1, w2) = (ssa.fresh_value(), ssa.fresh_value()); // two distinct `value_of(r)` reads

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut().push_parameter(v, Type::u(32));
    f.get_entry_mut().push_parameter(v2, Type::u(32));
    let entry = f.get_entry_mut();
    entry.push_instruction(OpCode::WriteWitness {
        result: Some(r),
        value: v,
        pinned: false,
    });
    // `value_of(r)` strips r's witness wrapper: w1, w2 are honestly equal to r. Two separate
    // reads stay distinct (ValueOf is excluded from value-numbering), so both join r's set.
    entry.push_instruction(OpCode::Cast {
        result: w1,
        value: r,
        target: CastTarget::ValueOf,
    });
    entry.push_instruction(OpCode::Cast {
        result: w2,
        value: r,
        target: CastTarget::ValueOf,
    });
    entry.push_instruction(OpCode::WriteWitness {
        result: Some(r2),
        value: v2,
        pinned: false,
    });
    entry.set_terminator(Terminator::Return(vec![]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);

    // r's honest-value set is the sorted union of the hint and both `value_of` reads.
    let mut expected = vec![v, w1, w2];
    expected.sort_unstable();
    assert_eq!(cc.witness_forward(fid, r), expected.as_slice());

    // The unrelated witness keeps a disjoint, single-element set — no cross-witness union.
    assert_eq!(cc.witness_forward(fid, r2), [v2].as_slice());
    assert!(!cc.known_equal(fid, r, r2));
}

/// An assert placed in the *last* block proves nothing by dominance (nothing follows it), but it
/// post-dominates everything upstream — so on accepting runs its fact holds at those earlier
/// blocks. The asserted value (`x`, a parameter) is in scope throughout.
#[test]
fn assert_below_use_holds_via_post_dominance() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c5 = ssa.add_const(Constant::U(32, 5));
    let x = ssa.fresh_value();

    let f = ssa.get_unique_entrypoint_mut();
    let entry_id = f.get_entry_id();
    let mid = f.add_block();
    let tail = f.add_block();
    f.get_entry_mut().push_parameter(x, Type::u(32));
    f.get_entry_mut()
        .set_terminator(Terminator::Jmp(mid, vec![]));
    f.get_block_mut(mid)
        .set_terminator(Terminator::Jmp(tail, vec![]));
    f.get_block_mut(tail).push_instruction(OpCode::AssertCmp {
        kind: CmpKind::Eq,
        lhs: x,
        rhs: c5,
    });
    f.get_block_mut(tail)
        .set_terminator(Terminator::Return(vec![]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    // `tail` post-dominates `mid` and `entry`, so `x == 5` holds at both — a fact pure dominance
    // (the assert is last) would miss entirely.
    assert_eq!(
        cc.asserted_const(fid, mid, x).as_deref(),
        Some(&Constant::U(32, 5))
    );
    assert_eq!(
        cc.asserted_const(fid, entry_id, x).as_deref(),
        Some(&Constant::U(32, 5))
    );
    // Still never recorded at the asserting block's own entry (block-entry granularity).
    assert_eq!(cc.asserted_const(fid, tail, x), None);
}

/// Post-dominance fan-out is gated on the asserted value being in scope: a value defined *below*
/// the target block must not have the fact attributed to that target (the guard dominance
/// otherwise supplies for free).
#[test]
fn post_dominance_respects_value_scope() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c10 = ssa.add_const(Constant::U(32, 10));
    let p = ssa.fresh_value();
    let x = ssa.fresh_value();

    let f = ssa.get_unique_entrypoint_mut();
    let entry_id = f.get_entry_id();
    let mid = f.add_block();
    let tail = f.add_block();
    f.get_entry_mut().push_parameter(p, Type::u(32));
    f.get_entry_mut()
        .set_terminator(Terminator::Jmp(mid, vec![]));
    // `x` is defined in `mid`, *below* `entry`.
    f.get_block_mut(mid)
        .push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: x,
            lhs: p,
            rhs: p,
        });
    f.get_block_mut(mid)
        .set_terminator(Terminator::Jmp(tail, vec![]));
    f.get_block_mut(tail).push_instruction(OpCode::AssertCmp {
        kind: CmpKind::Eq,
        lhs: x,
        rhs: c10,
    });
    f.get_block_mut(tail)
        .set_terminator(Terminator::Return(vec![]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    // At `mid`, `x` is in scope and `tail` post-dominates it ⇒ the fact holds.
    assert_eq!(
        cc.asserted_const(fid, mid, x).as_deref(),
        Some(&Constant::U(32, 10))
    );
    // At `entry`, `x` is not yet defined — the in-scope guard withholds the fact even though
    // `tail` post-dominates `entry`.
    assert_eq!(cc.asserted_const(fid, entry_id, x), None);
}

/// An assert on only one arm of a branch neither dominates nor post-dominates the blocks around
/// the branch, so its fact must not leak there — the other arm reaches `exit` without it.
#[test]
fn assert_on_one_branch_does_not_post_dominate() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c5 = ssa.add_const(Constant::U(32, 5));
    let cond = ssa.fresh_value();
    let x = ssa.fresh_value();

    let f = ssa.get_unique_entrypoint_mut();
    let entry_id = f.get_entry_id();
    let then_b = f.add_block();
    let else_b = f.add_block();
    let merge = f.add_block();
    f.get_entry_mut().push_parameter(cond, Type::u(1));
    f.get_entry_mut().push_parameter(x, Type::u(32));
    f.get_entry_mut()
        .set_terminator(Terminator::JmpIf(cond, then_b, else_b));
    // The assert lives only on the `then` branch.
    f.get_block_mut(then_b).push_instruction(OpCode::AssertCmp {
        kind: CmpKind::Eq,
        lhs: x,
        rhs: c5,
    });
    f.get_block_mut(then_b)
        .set_terminator(Terminator::Jmp(merge, vec![]));
    f.get_block_mut(else_b)
        .set_terminator(Terminator::Jmp(merge, vec![]));
    f.get_block_mut(merge)
        .set_terminator(Terminator::Return(vec![]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    // `x` is in scope everywhere, so only the missing dominance/post-dominance keeps the fact
    // out of `entry` and `merge` (the `else` path skips the assert).
    assert_eq!(cc.asserted_const(fid, entry_id, x), None);
    assert_eq!(cc.asserted_const(fid, merge, x), None);
    // And never at the asserting block's own entry.
    assert_eq!(cc.asserted_const(fid, then_b, x), None);
}

/// A post-dominating *pure* equality (`assert(x == y)`, neither side constant) requires *both*
/// sides in scope at the target.
#[test]
fn post_dominating_assert_eq_needs_both_sides_in_scope() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let x = ssa.fresh_value();
    let y = ssa.fresh_value();
    let p = ssa.fresh_value();

    let f = ssa.get_unique_entrypoint_mut();
    let entry_id = f.get_entry_id();
    let mid = f.add_block();
    let tail = f.add_block();
    f.get_entry_mut().push_parameter(x, Type::u(32));
    f.get_entry_mut().push_parameter(p, Type::u(32));
    f.get_entry_mut()
        .set_terminator(Terminator::Jmp(mid, vec![]));
    // `y` is defined in `mid`, below `entry`.
    f.get_block_mut(mid)
        .push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: y,
            lhs: p,
            rhs: p,
        });
    f.get_block_mut(mid)
        .set_terminator(Terminator::Jmp(tail, vec![]));
    f.get_block_mut(tail).push_instruction(OpCode::AssertCmp {
        kind: CmpKind::Eq,
        lhs: x,
        rhs: y,
    });
    f.get_block_mut(tail)
        .set_terminator(Terminator::Return(vec![]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    // Both sides live at `mid` ⇒ the post-dominating equality holds, symmetrically.
    assert!(cc.asserted_equal(fid, mid, x, y));
    assert!(cc.asserted_equal(fid, mid, y, x));
    // `y` is undefined at `entry`, so the equality is withheld there.
    assert!(!cc.asserted_equal(fid, entry_id, x, y));
}

/// A callee that returns a constant: the call result folds interprocedurally in the caller's
/// context, while the SCCP-visible intraprocedural view leaves it `Bottom`.
#[test]
fn interproc_constant_return_folds_at_call_site() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let helper = ssa.add_function("helper".to_string());
    let c5 = ssa.add_const(Constant::Field(Field::from(5u64)));
    let r = ssa.fresh_value();

    {
        let hf = ssa.get_function_mut(helper);
        hf.add_return_type(Type::field());
        hf.get_entry_mut()
            .set_terminator(Terminator::Return(vec![c5]));
    }
    {
        let mf = ssa.get_function_mut(main_id);
        mf.add_return_type(Type::field());
        let entry = mf.get_entry_mut();
        entry.push_instruction(OpCode::Call {
            results: vec![r],
            function: CallTarget::Static(helper),
            args: vec![],
            unconstrained: false,
        });
        entry.set_terminator(Terminator::Return(vec![r]));
    }

    let cc = run_in_test(&ssa);
    assert_eq!(
        cc.const_of_in(main_id, &Context::empty(), r).as_deref(),
        Some(&Constant::Field(Field::from(5u64)))
    );
    // The intraprocedural view (what SCCP reads) never folds a call result.
    assert_eq!(cc.const_of(main_id, r), None);
}

/// A pass-through callee (returns its parameter): the result takes the argument's constant at
/// the call site, and the callee's parameter is seeded to it per context.
#[test]
fn interproc_passthrough_seeds_param_and_result() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let id = ssa.add_function("id".to_string());
    let p = ssa.fresh_value();
    let c7 = ssa.add_const(Constant::Field(Field::from(7u64)));
    let r = ssa.fresh_value();

    {
        let hf = ssa.get_function_mut(id);
        hf.add_return_type(Type::field());
        hf.get_entry_mut().push_parameter(p, Type::field());
        hf.get_entry_mut()
            .set_terminator(Terminator::Return(vec![p]));
    }
    {
        let mf = ssa.get_function_mut(main_id);
        mf.add_return_type(Type::field());
        let entry = mf.get_entry_mut();
        entry.push_instruction(OpCode::Call {
            results: vec![r],
            function: CallTarget::Static(id),
            args: vec![c7],
            unconstrained: false,
        });
        entry.set_terminator(Terminator::Return(vec![r]));
    }

    let cc = run_in_test(&ssa);
    assert_eq!(
        cc.const_of_in(main_id, &Context::empty(), r).as_deref(),
        Some(&Constant::Field(Field::from(7u64)))
    );
    let ctxs = cc.contexts_of(id);
    assert_eq!(ctxs.len(), 1);
    assert_eq!(
        cc.const_of_in(id, &ctxs[0], p).as_deref(),
        Some(&Constant::Field(Field::from(7u64)))
    );
}

/// Interprocedural writeback through a summary: a callee that *returns* a comparison of
/// congruent operands gets a `Const` summary return-jump (the writeback runs in the polymorphic
/// summary solve), so the call result folds in the caller's context. Without it the return jump
/// is `Bottom` and the call result is unknown.
#[test]
fn interproc_writeback_folds_congruent_comparison_return() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let g = ssa.add_function("g".to_string());
    let c1 = ssa.add_const(Constant::U(32, 1));
    let (x, a, b, eq) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );
    let (x0, r) = (ssa.fresh_value(), ssa.fresh_value());

    // g(x) = ((x + 1) == (x + 1)) — a vacuous comparison of congruent operands.
    {
        let gf = ssa.get_function_mut(g);
        gf.add_return_type(Type::u(1));
        gf.get_entry_mut().push_parameter(x, Type::u(32));
        let entry = gf.get_entry_mut();
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: a,
            lhs: x,
            rhs: c1,
        });
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: b,
            lhs: x,
            rhs: c1,
        });
        entry.push_instruction(OpCode::Cmp {
            kind: CmpKind::Eq,
            result: eq,
            lhs: a,
            rhs: b,
        });
        entry.set_terminator(Terminator::Return(vec![eq]));
    }
    {
        let mf = ssa.get_function_mut(main_id);
        mf.add_return_type(Type::u(1));
        mf.get_entry_mut().push_parameter(x0, Type::u(32));
        let entry = mf.get_entry_mut();
        entry.push_instruction(OpCode::Call {
            results: vec![r],
            function: CallTarget::Static(g),
            args: vec![x0],
            unconstrained: false,
        });
        entry.set_terminator(Terminator::Return(vec![r]));
    }

    let cc = run_in_test(&ssa);
    // The summary return-jump is `Const(true)`, so the call result folds in the caller context.
    assert_eq!(
        cc.const_of_in(main_id, &Context::empty(), r).as_deref(),
        Some(&Constant::U(1, 1))
    );
    // The intraprocedural view (what SCCP reads) never folds a call result.
    assert_eq!(cc.const_of(main_id, r), None);
}

/// Interprocedural writeback in a specialized context: a comparison of congruent operands kept
/// *internal* to a callee folds in that callee's per-context facts (read via `const_of_in`).
/// This exercises the writeback in `specialize`'s per-context solve specifically — the fold is
/// observable only through `contexts`, which the summary solve does not populate.
#[test]
fn interproc_writeback_folds_congruent_comparison_in_context() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let g = ssa.add_function("g".to_string());
    let c1 = ssa.add_const(Constant::U(32, 1));
    let (x, a, b, eq) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );
    let (x0, r) = (ssa.fresh_value(), ssa.fresh_value());

    // g(x) computes `eq = (x + 1) == (x + 1)` but returns `x`, so `eq` is observable only
    // through g's per-context facts.
    {
        let gf = ssa.get_function_mut(g);
        gf.add_return_type(Type::u(32));
        gf.get_entry_mut().push_parameter(x, Type::u(32));
        let entry = gf.get_entry_mut();
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: a,
            lhs: x,
            rhs: c1,
        });
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: b,
            lhs: x,
            rhs: c1,
        });
        entry.push_instruction(OpCode::Cmp {
            kind: CmpKind::Eq,
            result: eq,
            lhs: a,
            rhs: b,
        });
        entry.set_terminator(Terminator::Return(vec![x]));
    }
    {
        let mf = ssa.get_function_mut(main_id);
        mf.add_return_type(Type::u(32));
        mf.get_entry_mut().push_parameter(x0, Type::u(32));
        let entry = mf.get_entry_mut();
        entry.push_instruction(OpCode::Call {
            results: vec![r],
            function: CallTarget::Static(g),
            args: vec![x0],
            unconstrained: false,
        });
        entry.set_terminator(Terminator::Return(vec![r]));
    }

    let cc = run_in_test(&ssa);
    let ctxs = cc.contexts_of(g);
    assert_eq!(ctxs.len(), 1);
    // The internal comparison folds within g's single context.
    assert_eq!(
        cc.const_of_in(g, &ctxs[0], eq).as_deref(),
        Some(&Constant::U(1, 1))
    );
    // Intraprocedurally (no context) the call result in main is still not folded.
    assert_eq!(cc.const_of(main_id, r), None);
}

/// The interprocedural writeback terminates under recursion: a self-recursive callee with a
/// vacuous comparison in its body still reaches a fixpoint (the summary solve stays monotone),
/// and the comparison folds in every context.
#[test]
fn interproc_writeback_terminates_under_recursion() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let g = ssa.add_function("g".to_string());
    let c1 = ssa.add_const(Constant::U(32, 1));
    let (x, a, b, eq, t) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );
    let (x0, r) = (ssa.fresh_value(), ssa.fresh_value());

    // g(x) = { eq = (x + 1) == (x + 1); let _ = g(x); eq } — self-recursive.
    {
        let gf = ssa.get_function_mut(g);
        gf.add_return_type(Type::u(1));
        gf.get_entry_mut().push_parameter(x, Type::u(32));
        let entry = gf.get_entry_mut();
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: a,
            lhs: x,
            rhs: c1,
        });
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: b,
            lhs: x,
            rhs: c1,
        });
        entry.push_instruction(OpCode::Cmp {
            kind: CmpKind::Eq,
            result: eq,
            lhs: a,
            rhs: b,
        });
        entry.push_instruction(OpCode::Call {
            results: vec![t],
            function: CallTarget::Static(g),
            args: vec![x],
            unconstrained: false,
        });
        entry.set_terminator(Terminator::Return(vec![eq]));
    }
    {
        let mf = ssa.get_function_mut(main_id);
        mf.add_return_type(Type::u(1));
        mf.get_entry_mut().push_parameter(x0, Type::u(32));
        let entry = mf.get_entry_mut();
        entry.push_instruction(OpCode::Call {
            results: vec![r],
            function: CallTarget::Static(g),
            args: vec![x0],
            unconstrained: false,
        });
        entry.set_terminator(Terminator::Return(vec![r]));
    }

    let cc = run_in_test(&ssa);
    let ctxs = cc.contexts_of(g);
    assert!(!ctxs.is_empty());
    for ctx in &ctxs {
        assert_eq!(
            cc.const_of_in(g, ctx, eq).as_deref(),
            Some(&Constant::U(1, 1))
        );
    }
}

/// The context-parameterized conditional queries. The branch-fact family (`const_in_block_in`)
/// is context-*precise* — it sees the per-context parameter constant the intraprocedural query
/// cannot — while the assert/witness family forwards to the (sound, context-independent)
/// intraprocedural facts.
#[test]
fn context_parameterized_conditional_queries() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let id = ssa.add_function("id".to_string());
    let p = ssa.fresh_value();
    let c7 = ssa.add_const(Constant::Field(Field::from(7u64)));
    let r = ssa.fresh_value();

    {
        let hf = ssa.get_function_mut(id);
        hf.add_return_type(Type::field());
        hf.get_entry_mut().push_parameter(p, Type::field());
        hf.get_entry_mut()
            .set_terminator(Terminator::Return(vec![p]));
    }
    {
        let mf = ssa.get_function_mut(main_id);
        mf.add_return_type(Type::field());
        let entry = mf.get_entry_mut();
        entry.push_instruction(OpCode::Call {
            results: vec![r],
            function: CallTarget::Static(id),
            args: vec![c7],
            unconstrained: false,
        });
        entry.set_terminator(Terminator::Return(vec![r]));
    }

    let entry_id = ssa.get_function(id).get_entry_id();
    let cc = run_in_test(&ssa);
    let ctxs = cc.contexts_of(id);
    assert_eq!(ctxs.len(), 1);
    let ctx = &ctxs[0];

    // Intraprocedurally `p` is unconstrained; under the single call context it is the constant 7.
    assert_eq!(cc.const_in_block(id, entry_id, p), None);
    assert_eq!(
        cc.const_in_block_in(id, ctx, entry_id, p).as_deref(),
        Some(&Constant::Field(Field::from(7u64)))
    );
}

/// Two call sites of one helper get distinct contexts with distinct per-context parameter
/// constants — the 1-CFA win.
#[test]
fn interproc_two_call_sites_are_distinguished() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let id = ssa.add_function("id".to_string());
    let p = ssa.fresh_value();
    let c5 = ssa.add_const(Constant::Field(Field::from(5u64)));
    let c7 = ssa.add_const(Constant::Field(Field::from(7u64)));
    let (r5, r7) = (ssa.fresh_value(), ssa.fresh_value());

    {
        let hf = ssa.get_function_mut(id);
        hf.add_return_type(Type::field());
        hf.get_entry_mut().push_parameter(p, Type::field());
        hf.get_entry_mut()
            .set_terminator(Terminator::Return(vec![p]));
    }
    {
        let mf = ssa.get_function_mut(main_id);
        mf.add_return_type(Type::field());
        let entry = mf.get_entry_mut();
        entry.push_instruction(OpCode::Call {
            results: vec![r5],
            function: CallTarget::Static(id),
            args: vec![c5],
            unconstrained: false,
        });
        entry.push_instruction(OpCode::Call {
            results: vec![r7],
            function: CallTarget::Static(id),
            args: vec![c7],
            unconstrained: false,
        });
        entry.set_terminator(Terminator::Return(vec![r5, r7]));
    }

    let cc = run_in_test(&ssa);
    assert_eq!(
        cc.const_of_in(main_id, &Context::empty(), r5).as_deref(),
        Some(&Constant::Field(Field::from(5u64)))
    );
    assert_eq!(
        cc.const_of_in(main_id, &Context::empty(), r7).as_deref(),
        Some(&Constant::Field(Field::from(7u64)))
    );
    let ctxs = cc.contexts_of(id);
    assert_eq!(ctxs.len(), 2, "two call sites → two contexts");
    let seen: Vec<Constant> = ctxs
        .iter()
        .filter_map(|ctx| cc.const_of_in(id, ctx, p).map(|c| (*c).clone()))
        .collect();
    assert_eq!(seen.len(), 2);
    assert!(seen.contains(&Constant::Field(Field::from(5u64))));
    assert!(seen.contains(&Constant::Field(Field::from(7u64))));
}

/// An unconstrained call's result is advice, not circuit-constrained: it never folds (even when
/// the callee returns a constant) and stays an opaque singleton.
#[test]
fn unconstrained_call_result_is_opaque() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let helper = ssa.add_function("helper".to_string());
    let c5 = ssa.add_const(Constant::Field(Field::from(5u64)));
    let (r1, r2) = (ssa.fresh_value(), ssa.fresh_value());

    {
        let hf = ssa.get_function_mut(helper);
        hf.add_return_type(Type::field());
        hf.get_entry_mut()
            .set_terminator(Terminator::Return(vec![c5]));
    }
    {
        let mf = ssa.get_function_mut(main_id);
        mf.add_return_type(Type::field());
        mf.add_return_type(Type::field());
        let entry = mf.get_entry_mut();
        entry.push_instruction(OpCode::Call {
            results: vec![r1],
            function: CallTarget::Static(helper),
            args: vec![],
            unconstrained: true,
        });
        entry.push_instruction(OpCode::Call {
            results: vec![r2],
            function: CallTarget::Static(helper),
            args: vec![],
            unconstrained: true,
        });
        entry.set_terminator(Terminator::Return(vec![r1, r2]));
    }

    let cc = run_in_test(&ssa);
    // Even interprocedurally, an unconstrained result is never folded ...
    assert_eq!(cc.const_of_in(main_id, &Context::empty(), r1), None);
    assert_eq!(cc.const_of(main_id, r1), None);
    // ... and two such results are not merged.
    assert!(!cc.known_equal(main_id, r1, r2));
}

/// Cross-call congruence: a callee whose return is a deterministic function of its arguments is
/// value-numbered cross-call, so two calls with congruent arguments yield congruent results —
/// and a call with a non-congruent argument does not.
#[test]
fn cross_call_congruence_for_deterministic_callee() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let dbl = ssa.add_function("dbl".to_string());
    let p = ssa.fresh_value();
    let psum = ssa.fresh_value();

    // dbl(p) = p + p — deterministic in its argument.
    {
        let hf = ssa.get_function_mut(dbl);
        hf.add_return_type(Type::field());
        hf.get_entry_mut().push_parameter(p, Type::field());
        hf.get_entry_mut().push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: psum,
            lhs: p,
            rhs: p,
        });
        hf.get_entry_mut()
            .set_terminator(Terminator::Return(vec![psum]));
    }

    let c1 = ssa.add_const(Constant::Field(Field::from(1u64)));
    let (x, y) = (ssa.fresh_value(), ssa.fresh_value());
    let (a, b) = (ssa.fresh_value(), ssa.fresh_value());
    let (r1, r2, r3) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    {
        let mf = ssa.get_function_mut(main_id);
        mf.add_return_type(Type::field());
        mf.add_return_type(Type::field());
        mf.add_return_type(Type::field());
        mf.get_entry_mut().push_parameter(x, Type::field());
        mf.get_entry_mut().push_parameter(y, Type::field());
        let entry = mf.get_entry_mut();
        // `a` and `b` are structurally congruent (both `x + 1`); `y` is distinct.
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: a,
            lhs: x,
            rhs: c1,
        });
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: b,
            lhs: x,
            rhs: c1,
        });
        entry.push_instruction(OpCode::Call {
            results: vec![r1],
            function: CallTarget::Static(dbl),
            args: vec![a],
            unconstrained: false,
        });
        entry.push_instruction(OpCode::Call {
            results: vec![r2],
            function: CallTarget::Static(dbl),
            args: vec![b],
            unconstrained: false,
        });
        entry.push_instruction(OpCode::Call {
            results: vec![r3],
            function: CallTarget::Static(dbl),
            args: vec![y],
            unconstrained: false,
        });
        entry.set_terminator(Terminator::Return(vec![r1, r2, r3]));
    }

    let cc = run_in_test(&ssa);
    assert!(cc.known_equal(main_id, r1, r2)); // congruent args ⇒ congruent results
    assert!(!cc.known_equal(main_id, r1, r3)); // distinct arg ⇒ distinct result
}

/// A callee whose return carries a fresh witness is *not* a deterministic function of its
/// arguments, so its results are never numbered cross-call — two such calls stay distinct (the
/// determinism gate that protects the free-witness non-merge prohibition).
#[test]
fn cross_call_no_congruence_for_nondeterministic_callee() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let rnd = ssa.add_function("rnd".to_string());
    let w = ssa.fresh_value();

    {
        let hf = ssa.get_function_mut(rnd);
        hf.add_return_type(Type::field());
        hf.get_entry_mut().push_instruction(OpCode::FreshWitness {
            result: w,
            result_type: Type::field(),
        });
        hf.get_entry_mut()
            .set_terminator(Terminator::Return(vec![w]));
    }
    let (r1, r2) = (ssa.fresh_value(), ssa.fresh_value());
    {
        let mf = ssa.get_function_mut(main_id);
        mf.add_return_type(Type::field());
        mf.add_return_type(Type::field());
        let entry = mf.get_entry_mut();
        entry.push_instruction(OpCode::Call {
            results: vec![r1],
            function: CallTarget::Static(rnd),
            args: vec![],
            unconstrained: false,
        });
        entry.push_instruction(OpCode::Call {
            results: vec![r2],
            function: CallTarget::Static(rnd),
            args: vec![],
            unconstrained: false,
        });
        entry.set_terminator(Terminator::Return(vec![r1, r2]));
    }

    let cc = run_in_test(&ssa);
    assert!(!cc.known_equal(main_id, r1, r2));
}

/// Determinism is interprocedural: `outer` is deterministic because the `inner` it calls is, so
/// its results are numbered cross-call too.
#[test]
fn cross_call_congruence_is_transitive() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let inner = ssa.add_function("inner".to_string());
    let outer = ssa.add_function("outer".to_string());

    let ip = ssa.fresh_value();
    let isum = ssa.fresh_value();
    {
        let hf = ssa.get_function_mut(inner);
        hf.add_return_type(Type::field());
        hf.get_entry_mut().push_parameter(ip, Type::field());
        hf.get_entry_mut().push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: isum,
            lhs: ip,
            rhs: ip,
        });
        hf.get_entry_mut()
            .set_terminator(Terminator::Return(vec![isum]));
    }
    let op = ssa.fresh_value();
    let ores = ssa.fresh_value();
    {
        let hf = ssa.get_function_mut(outer);
        hf.add_return_type(Type::field());
        hf.get_entry_mut().push_parameter(op, Type::field());
        hf.get_entry_mut().push_instruction(OpCode::Call {
            results: vec![ores],
            function: CallTarget::Static(inner),
            args: vec![op],
            unconstrained: false,
        });
        hf.get_entry_mut()
            .set_terminator(Terminator::Return(vec![ores]));
    }

    let c1 = ssa.add_const(Constant::Field(Field::from(1u64)));
    let x = ssa.fresh_value();
    let (a, b) = (ssa.fresh_value(), ssa.fresh_value());
    let (r1, r2) = (ssa.fresh_value(), ssa.fresh_value());
    {
        let mf = ssa.get_function_mut(main_id);
        mf.add_return_type(Type::field());
        mf.add_return_type(Type::field());
        mf.get_entry_mut().push_parameter(x, Type::field());
        let entry = mf.get_entry_mut();
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: a,
            lhs: x,
            rhs: c1,
        });
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: b,
            lhs: x,
            rhs: c1,
        });
        entry.push_instruction(OpCode::Call {
            results: vec![r1],
            function: CallTarget::Static(outer),
            args: vec![a],
            unconstrained: false,
        });
        entry.push_instruction(OpCode::Call {
            results: vec![r2],
            function: CallTarget::Static(outer),
            args: vec![b],
            unconstrained: false,
        });
        entry.set_terminator(Terminator::Return(vec![r1, r2]));
    }

    let cc = run_in_test(&ssa);
    assert!(cc.known_equal(main_id, r1, r2));
}

/// A callee that returns a pure transform of an array parameter (`p[0]`) is a deterministic
/// function of its argument now that sequence ops are value-numbered, so two calls with
/// congruent array arguments yield congruent results. Before this change the `ArrayGet` tainted
/// the return as non-deterministic and the calls stayed distinct.
#[test]
fn cross_call_congruence_for_array_returning_callee() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let pick = ssa.add_function("pick".to_string());
    let p = ssa.fresh_value();
    let elem = ssa.fresh_value();
    let c0 = ssa.add_const(Constant::U(32, 0));

    // pick(p) = p[0] — deterministic in its array argument.
    {
        let hf = ssa.get_function_mut(pick);
        hf.add_return_type(Type::field());
        hf.get_entry_mut()
            .push_parameter(p, Type::field().array_of(1));
        hf.get_entry_mut().push_instruction(OpCode::ArrayGet {
            result: elem,
            array: p,
            index: c0,
        });
        hf.get_entry_mut()
            .set_terminator(Terminator::Return(vec![elem]));
    }

    let (x, y) = (ssa.fresh_value(), ssa.fresh_value());
    let (a, b, d) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let (r1, r2, r3) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    {
        let mf = ssa.get_function_mut(main_id);
        mf.add_return_type(Type::field());
        mf.add_return_type(Type::field());
        mf.add_return_type(Type::field());
        mf.get_entry_mut().push_parameter(x, Type::field());
        mf.get_entry_mut().push_parameter(y, Type::field());
        let entry = mf.get_entry_mut();
        // `a` and `b` are congruent single-element arrays (`[x]`); `d` is `[y]`.
        entry.push_instruction(OpCode::MkSeq {
            result: a,
            elems: vec![x],
            seq_type: SequenceTargetType::Array(1),
            elem_type: Type::field(),
        });
        entry.push_instruction(OpCode::MkSeq {
            result: b,
            elems: vec![x],
            seq_type: SequenceTargetType::Array(1),
            elem_type: Type::field(),
        });
        entry.push_instruction(OpCode::MkSeq {
            result: d,
            elems: vec![y],
            seq_type: SequenceTargetType::Array(1),
            elem_type: Type::field(),
        });
        entry.push_instruction(OpCode::Call {
            results: vec![r1],
            function: CallTarget::Static(pick),
            args: vec![a],
            unconstrained: false,
        });
        entry.push_instruction(OpCode::Call {
            results: vec![r2],
            function: CallTarget::Static(pick),
            args: vec![b],
            unconstrained: false,
        });
        entry.push_instruction(OpCode::Call {
            results: vec![r3],
            function: CallTarget::Static(pick),
            args: vec![d],
            unconstrained: false,
        });
        entry.set_terminator(Terminator::Return(vec![r1, r2, r3]));
    }

    let cc = run_in_test(&ssa);
    assert!(cc.known_equal(main_id, r1, r2)); // congruent array args ⇒ congruent results
    assert!(!cc.known_equal(main_id, r1, r3)); // distinct array arg ⇒ distinct result
}

/// The headline combined-fixpoint writeback: `CmpEq(a, b)` with `a` and `b` proven congruent
/// folds to `true` even though neither is constant, and the false edge of the branch it decides
/// is pruned. Neither SCCP (operands not constant) nor plain GVN (does not fold comparisons)
/// reaches this alone.
#[test]
fn cmp_eq_of_congruent_operands_folds_and_prunes_dead_edge() {
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
    // a = x + 1 and b = x + 1 are structurally congruent but not constant.
    entry.push_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Add,
        result: a,
        lhs: x,
        rhs: c1,
    });
    entry.push_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Add,
        result: b,
        lhs: x,
        rhs: c1,
    });
    entry.push_instruction(OpCode::Cmp {
        kind: CmpKind::Eq,
        result: eq,
        lhs: a,
        rhs: b,
    });
    entry.set_terminator(Terminator::JmpIf(eq, then_b, else_b));
    f.get_block_mut(then_b)
        .set_terminator(Terminator::Return(vec![a]));
    f.get_block_mut(else_b)
        .set_terminator(Terminator::Return(vec![b]));

    let fid = ssa.get_unique_entrypoint_id();
    let entry_id = ssa.get_unique_entrypoint().get_entry_id();
    let cc = run_in_test(&ssa);

    assert!(cc.known_equal(fid, a, b));
    // The comparison is now an unconditional constant `true`.
    assert_eq!(cc.const_of(fid, eq).as_deref(), Some(&Constant::U(1, 1)));
    // ...so the branch is decided: only the then-edge is executable.
    assert!(cc.is_executable_edge(fid, entry_id, then_b));
    assert!(!cc.is_executable_edge(fid, entry_id, else_b));
    assert!(cc.is_reachable(fid, then_b));
    assert!(!cc.is_reachable(fid, else_b));
}

/// The writeback cascades: deciding a congruence-derived branch prunes an edge, leaving a merge
/// parameter with a single constant in-edge — a constant SCCP alone (which sees both edges live)
/// cannot derive.
#[test]
fn writeback_cascades_to_downstream_constant() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c1 = ssa.add_const(Constant::U(32, 1));
    let c5 = ssa.add_const(Constant::U(32, 5));
    let c7 = ssa.add_const(Constant::U(32, 7));
    let (x, a, b, eq, p) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    let then_b = f.add_block();
    let else_b = f.add_block();
    let merge = f.add_block();
    f.get_entry_mut().push_parameter(x, Type::u(32));
    let entry = f.get_entry_mut();
    entry.push_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Add,
        result: a,
        lhs: x,
        rhs: c1,
    });
    entry.push_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Add,
        result: b,
        lhs: x,
        rhs: c1,
    });
    entry.push_instruction(OpCode::Cmp {
        kind: CmpKind::Eq,
        result: eq,
        lhs: a,
        rhs: b,
    });
    entry.set_terminator(Terminator::JmpIf(eq, then_b, else_b));
    // The dead else-edge would carry a different constant, forcing the merge to ⊥ under SCCP
    // alone.
    f.get_block_mut(then_b)
        .set_terminator(Terminator::Jmp(merge, vec![c5]));
    f.get_block_mut(else_b)
        .set_terminator(Terminator::Jmp(merge, vec![c7]));
    let merge_block = f.get_block_mut(merge);
    merge_block.push_parameter(p, Type::u(32));
    merge_block.set_terminator(Terminator::Return(vec![p]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);

    assert!(!cc.is_reachable(fid, else_b));
    assert_eq!(cc.const_of(fid, p).as_deref(), Some(&Constant::U(32, 5)));
}

/// Guard: a comparison of values that are *not* congruent stays unfolded and both branch targets
/// stay reachable.
#[test]
fn cmp_eq_of_noncongruent_operands_is_not_folded() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (x, y, eq) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    let then_b = f.add_block();
    let else_b = f.add_block();
    f.get_entry_mut().push_parameter(x, Type::u(32));
    f.get_entry_mut().push_parameter(y, Type::u(32));
    let entry = f.get_entry_mut();
    entry.push_instruction(OpCode::Cmp {
        kind: CmpKind::Eq,
        result: eq,
        lhs: x,
        rhs: y,
    });
    entry.set_terminator(Terminator::JmpIf(eq, then_b, else_b));
    f.get_block_mut(then_b)
        .set_terminator(Terminator::Return(vec![x]));
    f.get_block_mut(else_b)
        .set_terminator(Terminator::Return(vec![y]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);

    assert!(!cc.known_equal(fid, x, y));
    assert_eq!(cc.const_of(fid, eq), None);
    assert!(cc.is_reachable(fid, then_b));
    assert!(cc.is_reachable(fid, else_b));
}

/// Soundness boundary: two free witnesses are never congruent, so a comparison of them is never
/// folded — the writeback cannot fabricate an equality an adversarial witness could break.
#[test]
fn free_witnesses_are_not_congruent_so_comparison_not_folded() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (w1, w2, eq) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    let entry = f.get_entry_mut();
    entry.push_instruction(OpCode::FreshWitness {
        result: w1,
        result_type: Type::field(),
    });
    entry.push_instruction(OpCode::FreshWitness {
        result: w2,
        result_type: Type::field(),
    });
    entry.push_instruction(OpCode::Cmp {
        kind: CmpKind::Eq,
        result: eq,
        lhs: w1,
        rhs: w2,
    });
    entry.set_terminator(Terminator::Return(vec![eq]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);

    assert!(!cc.known_equal(fid, w1, w2));
    assert_eq!(cc.const_of(fid, eq), None);
}

/// The writeback adds nothing when no comparison has congruent operands: a plain non-constant
/// computation yields no new constants (so a function without such a comparison is unchanged).
#[test]
fn writeback_is_noop_without_congruent_comparison() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c1 = ssa.add_const(Constant::U(32, 1));
    let (x, a) = (ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    let entry = f.get_entry_mut();
    entry.push_parameter(x, Type::u(32));
    entry.push_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Add,
        result: a,
        lhs: x,
        rhs: c1,
    });
    entry.set_terminator(Terminator::Return(vec![a]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);

    assert_eq!(cc.const_of(fid, a), None);
    assert!(cc.new_const_values(fid).is_empty());
}

/// `CmpLt(a, b)` with `a` and `b` congruent folds to `false` — `x < x` is never true.
#[test]
fn cmp_lt_of_congruent_operands_folds_false() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c1 = ssa.add_const(Constant::U(32, 1));
    let (x, a, b, lt) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut().push_parameter(x, Type::u(32));
    let entry = f.get_entry_mut();
    entry.push_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Add,
        result: a,
        lhs: x,
        rhs: c1,
    });
    entry.push_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Add,
        result: b,
        lhs: x,
        rhs: c1,
    });
    entry.push_instruction(OpCode::Cmp {
        kind: CmpKind::Lt,
        result: lt,
        lhs: a,
        rhs: b,
    });
    entry.set_terminator(Terminator::Return(vec![lt]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);

    assert!(cc.known_equal(fid, a, b));
    assert_eq!(cc.const_of(fid, lt).as_deref(), Some(&Constant::U(1, 0)));
}

/// A comparison of congruent operands folds to a constant `true` even when it is *witnessed*
/// (its result is `WitnessOf`-typed, as emitted by witness spilling / AD lowering) — the
/// must-equal fact holds either way. Keeping the substituted constant witness-typed is the
/// consumer's job; see the SCCP test `witnessed_constant_is_cast_to_witness_of`.
#[test]
fn witnessed_comparison_of_congruent_operands_is_promoted() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (w, ww, n, nn) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    let entry = f.get_entry_mut();
    entry.push_parameter(w, Type::witness_of(Type::u(32)));
    entry.push_parameter(n, Type::u(32));
    // Witnessed comparison: an operand is `WitnessOf`, so the result is `WitnessOf(u1)`.
    entry.push_instruction(OpCode::Cmp {
        kind: CmpKind::Eq,
        result: ww,
        lhs: w,
        rhs: w,
    });
    // Plain comparison: result is `u1`.
    entry.push_instruction(OpCode::Cmp {
        kind: CmpKind::Eq,
        result: nn,
        lhs: n,
        rhs: n,
    });
    entry.set_terminator(Terminator::Return(vec![ww, nn]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);

    // Both compare congruent (identical) operands, so both fold to `true` — the witnessed one
    // too. (SCCP then keeps `ww` witness-typed via a cast; the analysis fact is the same.)
    assert!(cc.known_equal(fid, w, w));
    assert!(cc.known_equal(fid, n, n));
    assert_eq!(cc.const_of(fid, ww).as_deref(), Some(&Constant::U(1, 1)));
    assert_eq!(cc.const_of(fid, nn).as_deref(), Some(&Constant::U(1, 1)));
}

/// Combined-analysis win unreachable to either factor alone: two loop-carried parallel induction
/// variables are congruent (optimistic GVN), so `i == j` folds to `true` even though neither `i`
/// nor `j` is constant.
#[test]
fn loop_carried_congruent_comparison_folds_true() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c0 = ssa.add_const(Constant::U(32, 0));
    let c1 = ssa.add_const(Constant::U(32, 1));
    let c10 = ssa.add_const(Constant::U(32, 10));
    let (i, j, lt, eq, i2, j2) = (
        ssa.fresh_value(),
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
    let header_block = f.get_block_mut(header);
    header_block.push_parameter(i, Type::u(32));
    header_block.push_parameter(j, Type::u(32));
    header_block.push_instruction(OpCode::Cmp {
        kind: CmpKind::Eq,
        result: eq,
        lhs: i,
        rhs: j,
    });
    header_block.push_instruction(OpCode::Cmp {
        kind: CmpKind::Lt,
        result: lt,
        lhs: i,
        rhs: c10,
    });
    header_block.set_terminator(Terminator::JmpIf(lt, body, exit));
    let body_block = f.get_block_mut(body);
    body_block.push_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Add,
        result: i2,
        lhs: i,
        rhs: c1,
    });
    body_block.push_instruction(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Add,
        result: j2,
        lhs: j,
        rhs: c1,
    });
    body_block.set_terminator(Terminator::Jmp(header, vec![i2, j2]));
    f.get_block_mut(exit)
        .set_terminator(Terminator::Return(vec![i, j]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);

    assert!(cc.known_equal(fid, i, j));
    assert_eq!(cc.const_of(fid, eq).as_deref(), Some(&Constant::U(1, 1)));
}

// AGGREGATE CONSTANT FOLDING
// ============================================================================================

/// `MkSeq` of constant elements folds to an *internal* aggregate; `ArrayGet` at a constant
/// index projects a scalar constant out of it. The aggregate itself is never surfaced to
/// consumers.
#[test]
fn const_array_get_folds_to_scalar() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c0 = ssa.add_const(Constant::U(32, 10));
    let c1 = ssa.add_const(Constant::U(32, 20));
    let c2 = ssa.add_const(Constant::U(32, 30));
    let idx = ssa.add_const(Constant::U(32, 1));
    let (seq, got) = (ssa.fresh_value(), ssa.fresh_value());

    let entry = ssa.get_unique_entrypoint_mut().get_entry_mut();
    entry.push_instruction(OpCode::MkSeq {
        result: seq,
        elems: vec![c0, c1, c2],
        seq_type: SequenceTargetType::Array(3),
        elem_type: Type::u(32),
    });
    entry.push_instruction(OpCode::ArrayGet {
        result: got,
        array: seq,
        index: idx,
    });
    entry.set_terminator(Terminator::Return(vec![got]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);

    // The projection is a surfaced scalar constant.
    assert_eq!(cc.const_of(fid, got).as_deref(), Some(&Constant::U(32, 20)));
    assert!(
        cc.new_const_values(fid)
            .iter()
            .any(|(v, c)| *v == got && **c == Constant::U(32, 20))
    );
    // The aggregate stays internal: never surfaced as a constant.
    assert_eq!(cc.const_of(fid, seq), None);
    assert!(cc.new_const_values(fid).iter().all(|(v, _)| *v != seq));
}

/// `MkRepeated` of a constant folds, and both `ArrayGet` and `SliceLen` project scalars out of
/// the resulting aggregate (the length is always `u32`).
#[test]
fn const_repeated_array_get_and_slice_len() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let elem = ssa.add_const(Constant::U(32, 7));
    let idx = ssa.add_const(Constant::U(32, 2));
    let (seq, got, len) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    let entry = ssa.get_unique_entrypoint_mut().get_entry_mut();
    entry.push_instruction(OpCode::MkRepeated {
        result: seq,
        element: elem,
        seq_type: SequenceTargetType::Array(4),
        count: 4,
        elem_type: Type::u(32),
    });
    entry.push_instruction(OpCode::ArrayGet {
        result: got,
        array: seq,
        index: idx,
    });
    entry.push_instruction(OpCode::SliceLen {
        result: len,
        slice: seq,
    });
    entry.set_terminator(Terminator::Return(vec![got, len]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);

    assert_eq!(cc.const_of(fid, got).as_deref(), Some(&Constant::U(32, 7)));
    assert_eq!(cc.const_of(fid, len).as_deref(), Some(&Constant::U(32, 4)));
}

/// `ArraySet` of a constant aggregate is itself a constant aggregate: a later `ArrayGet` sees
/// the updated cell at the set index and the original value elsewhere.
#[test]
fn const_array_set_then_get() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c0 = ssa.add_const(Constant::U(32, 10));
    let c1 = ssa.add_const(Constant::U(32, 20));
    let c2 = ssa.add_const(Constant::U(32, 30));
    let new_val = ssa.add_const(Constant::U(32, 99));
    let idx0 = ssa.add_const(Constant::U(32, 0));
    let idx1 = ssa.add_const(Constant::U(32, 1));
    let (seq, seq2, at_set, at_orig) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let entry = ssa.get_unique_entrypoint_mut().get_entry_mut();
    entry.push_instruction(OpCode::MkSeq {
        result: seq,
        elems: vec![c0, c1, c2],
        seq_type: SequenceTargetType::Array(3),
        elem_type: Type::u(32),
    });
    entry.push_instruction(OpCode::ArraySet {
        result: seq2,
        array: seq,
        index: idx1,
        value: new_val,
    });
    entry.push_instruction(OpCode::ArrayGet {
        result: at_set,
        array: seq2,
        index: idx1,
    });
    entry.push_instruction(OpCode::ArrayGet {
        result: at_orig,
        array: seq2,
        index: idx0,
    });
    entry.set_terminator(Terminator::Return(vec![at_set, at_orig]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);

    assert_eq!(
        cc.const_of(fid, at_set).as_deref(),
        Some(&Constant::U(32, 99))
    );
    assert_eq!(
        cc.const_of(fid, at_orig).as_deref(),
        Some(&Constant::U(32, 10))
    );
}

/// `SlicePush` extends a constant aggregate at the front or back, preserving element order.
#[test]
fn const_slice_push_front_and_back() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let mid = ssa.add_const(Constant::U(32, 1));
    let front = ssa.add_const(Constant::U(32, 0));
    let back = ssa.add_const(Constant::U(32, 2));
    let idx0 = ssa.add_const(Constant::U(32, 0));
    let idx1 = ssa.add_const(Constant::U(32, 1));
    let (base, pushed_front, pushed_back, head, tail) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let entry = ssa.get_unique_entrypoint_mut().get_entry_mut();
    entry.push_instruction(OpCode::MkSeq {
        result: base,
        elems: vec![mid],
        seq_type: SequenceTargetType::Slice,
        elem_type: Type::u(32),
    });
    // Front: [front, mid]; element 0 is the pushed value.
    entry.push_instruction(OpCode::SlicePush {
        dir: SliceOpDir::Front,
        result: pushed_front,
        slice: base,
        values: vec![front],
    });
    // Back: [mid, back]; element 1 is the pushed value.
    entry.push_instruction(OpCode::SlicePush {
        dir: SliceOpDir::Back,
        result: pushed_back,
        slice: base,
        values: vec![back],
    });
    entry.push_instruction(OpCode::ArrayGet {
        result: head,
        array: pushed_front,
        index: idx0,
    });
    entry.push_instruction(OpCode::ArrayGet {
        result: tail,
        array: pushed_back,
        index: idx1,
    });
    entry.set_terminator(Terminator::Return(vec![head, tail]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);

    assert_eq!(cc.const_of(fid, head).as_deref(), Some(&Constant::U(32, 0)));
    assert_eq!(cc.const_of(fid, tail).as_deref(), Some(&Constant::U(32, 2)));
}

/// `MkSeqOfBlob` re-views a constant blob as a sequence; projecting a constant index folds.
#[test]
fn const_mk_seq_of_blob_folds() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let blob = ssa.add_const(Constant::Blob(Blob::new(
        Type::u(32),
        vec![Constant::U(32, 100), Constant::U(32, 200)],
    )));
    let idx = ssa.add_const(Constant::U(32, 1));
    let (seq, got) = (ssa.fresh_value(), ssa.fresh_value());

    let entry = ssa.get_unique_entrypoint_mut().get_entry_mut();
    entry.push_instruction(OpCode::MkSeqOfBlob {
        result: seq,
        element_type: Type::u(32),
        blob,
    });
    entry.push_instruction(OpCode::ArrayGet {
        result: got,
        array: seq,
        index: idx,
    });
    entry.set_terminator(Terminator::Return(vec![got]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);

    assert_eq!(
        cc.const_of(fid, got).as_deref(),
        Some(&Constant::U(32, 200))
    );
    // The aggregate stays internal.
    assert_eq!(cc.const_of(fid, seq), None);
}

/// Aggregate folding refuses (stays `Bottom`, i.e. `const_of == None`) on an out-of-bounds
/// constant index, a non-constant element, and an over-cap `MkRepeated`.
#[test]
fn aggregate_folding_negative_cases() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c0 = ssa.add_const(Constant::U(32, 10));
    let c1 = ssa.add_const(Constant::U(32, 20));
    let idx0 = ssa.add_const(Constant::U(32, 0));
    let idx_oob = ssa.add_const(Constant::U(32, 5));
    let p = ssa.fresh_value();
    let (seq, g_oob, seq_nc, g_nc, big, g_big) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    f.get_entry_mut().push_parameter(p, Type::u(32));
    let entry = f.get_entry_mut();
    // Out of bounds: index 5 into a length-2 array.
    entry.push_instruction(OpCode::MkSeq {
        result: seq,
        elems: vec![c0, c1],
        seq_type: SequenceTargetType::Array(2),
        elem_type: Type::u(32),
    });
    entry.push_instruction(OpCode::ArrayGet {
        result: g_oob,
        array: seq,
        index: idx_oob,
    });
    // Non-constant element keeps the whole aggregate non-constant.
    entry.push_instruction(OpCode::MkSeq {
        result: seq_nc,
        elems: vec![c0, p],
        seq_type: SequenceTargetType::Array(2),
        elem_type: Type::u(32),
    });
    entry.push_instruction(OpCode::ArrayGet {
        result: g_nc,
        array: seq_nc,
        index: idx0,
    });
    // Over-cap repeat is refused before materialising the aggregate.
    entry.push_instruction(OpCode::MkRepeated {
        result: big,
        element: c0,
        seq_type: SequenceTargetType::Array((1 << 16) + 1),
        count: (1 << 16) + 1,
        elem_type: Type::u(32),
    });
    entry.push_instruction(OpCode::ArrayGet {
        result: g_big,
        array: big,
        index: idx0,
    });
    entry.set_terminator(Terminator::Return(vec![g_oob, g_nc, g_big]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);

    assert_eq!(cc.const_of(fid, g_oob), None);
    assert_eq!(cc.const_of(fid, g_nc), None);
    assert_eq!(cc.const_of(fid, g_big), None);
}

/// The remaining refusal paths, complementing `aggregate_folding_negative_cases`: an
/// out-of-bounds `ArraySet`, an over-cap `SlicePush`, and an over-cap `MkSeq` all stay
/// `Bottom`. Each is observed through a later `ArrayGet`, which bottoms out over the refused
/// (non-constant) aggregate.
#[test]
fn aggregate_folding_refuses_oob_set_and_over_cap_constructors() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c0 = ssa.add_const(Constant::U(32, 10));
    let c1 = ssa.add_const(Constant::U(32, 20));
    let idx0 = ssa.add_const(Constant::U(32, 0));
    let idx_oob = ssa.add_const(Constant::U(32, 5));
    let over_cap = (1usize << 12) + 1; // AGGREGATE_FOLD_CAP + 1

    let (base, set_oob, at_set_oob) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let (one, pushed, at_pushed) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let (big_seq, at_big) = (ssa.fresh_value(), ssa.fresh_value());

    let entry = ssa.get_unique_entrypoint_mut().get_entry_mut();

    // Out-of-bounds `ArraySet` (index 5 into a length-2 array): refused, so the get sees Bottom.
    entry.push_instruction(OpCode::MkSeq {
        result: base,
        elems: vec![c0, c1],
        seq_type: SequenceTargetType::Array(2),
        elem_type: Type::u(32),
    });
    entry.push_instruction(OpCode::ArraySet {
        result: set_oob,
        array: base,
        index: idx_oob,
        value: c0,
    });
    entry.push_instruction(OpCode::ArrayGet {
        result: at_set_oob,
        array: set_oob,
        index: idx0,
    });

    // `SlicePush` whose result would exceed the cap (1 + CAP values): refused.
    entry.push_instruction(OpCode::MkSeq {
        result: one,
        elems: vec![c0],
        seq_type: SequenceTargetType::Slice,
        elem_type: Type::u(32),
    });
    entry.push_instruction(OpCode::SlicePush {
        dir: SliceOpDir::Back,
        result: pushed,
        slice: one,
        values: vec![c0; 1 << 12],
    });
    entry.push_instruction(OpCode::ArrayGet {
        result: at_pushed,
        array: pushed,
        index: idx0,
    });

    // Over-cap `MkSeq`: refused before materialising the aggregate.
    entry.push_instruction(OpCode::MkSeq {
        result: big_seq,
        elems: vec![c0; over_cap],
        seq_type: SequenceTargetType::Array(over_cap),
        elem_type: Type::u(32),
    });
    entry.push_instruction(OpCode::ArrayGet {
        result: at_big,
        array: big_seq,
        index: idx0,
    });

    entry.set_terminator(Terminator::Return(vec![at_set_oob, at_pushed, at_big]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);

    assert_eq!(cc.const_of(fid, at_set_oob), None);
    assert_eq!(cc.const_of(fid, at_pushed), None);
    assert_eq!(cc.const_of(fid, at_big), None);
}
