use crate::compiler::{
    Field,
    analysis::{
        click_cooper::ClickCooper, flow_analysis::FlowAnalysis, shared::call_string::Context,
        types::Types,
    },
    ssa::{
        FunctionId, Located, ProgramPoint, SourceLocation, Terminator,
        hlssa::{
            BinaryArithOpKind, Blob, CallTarget, CastTarget, CmpKind, Constant, HLSSA, OpCode,
            ScalarFold, SequenceTargetType, SliceOpDir, Type,
        },
    },
};

// UTILITIES
// ================================================================================================

/// Build the analysis for `ssa` with freshly-computed dependencies, for test use only.
pub(crate) fn run_in_test(ssa: &HLSSA) -> ClickCooper {
    let flow = FlowAnalysis::run(ssa);
    let types = Types::new().run(ssa, &flow);
    ClickCooper::run(ssa, &flow, &types)
}

/// The single context `f` was specialized in, asserting there is exactly one.
fn sole_context(cc: &ClickCooper, f: FunctionId) -> Context {
    let ctxs = cc.contexts_of(f);
    assert_eq!(ctxs.len(), 1);
    ctxs.into_iter().next().unwrap()
}

// TESTS
// ================================================================================================

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
    entry.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: a,
            lhs: c2,
            rhs: c3,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: b,
            lhs: c1,
            rhs: c4,
        },
        SourceLocation::test(),
    ));
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
    entry.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: a,
            lhs: x,
            rhs: y,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: b,
            lhs: x,
            rhs: y,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: c,
            lhs: y,
            rhs: x,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Sub,
            result: d,
            lhs: x,
            rhs: y,
        },
        SourceLocation::test(),
    ));
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
    entry.push_instruction(Located::with(
        OpCode::ArrayGet {
            result: g1,
            array: arr,
            index: i,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::ArrayGet {
            result: g2,
            array: arr,
            index: i,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::ArrayGet {
            result: g3,
            array: arr,
            index: j,
        },
        SourceLocation::test(),
    ));
    // s1 and s2 are the same array `[arr[i], arr[j]]` built twice — congruent elementwise.
    entry.push_instruction(Located::with(
        OpCode::MkSeq {
            result: s1,
            elems: vec![g1, g3],
            seq_type: SequenceTargetType::Array(2),
            elem_type: Type::field(),
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::MkSeq {
            result: s2,
            elems: vec![g2, g3],
            seq_type: SequenceTargetType::Array(2),
            elem_type: Type::field(),
        },
        SourceLocation::test(),
    ));
    // s3 differs only in sequence kind (Slice), s4 only in element type (u32): neither merges.
    entry.push_instruction(Located::with(
        OpCode::MkSeq {
            result: s3,
            elems: vec![g1, g3],
            seq_type: SequenceTargetType::Slice,
            elem_type: Type::field(),
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::MkSeq {
            result: s4,
            elems: vec![g1, g3],
            seq_type: SequenceTargetType::Array(2),
            elem_type: Type::u(32),
        },
        SourceLocation::test(),
    ));
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
    header_block.push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Lt,
            result: lt,
            lhs: i,
            rhs: c10,
        },
        SourceLocation::test(),
    ));
    header_block.set_terminator(Terminator::JmpIf(lt, body, exit));
    let body_block = f.get_block_mut(body);
    body_block.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: i2,
            lhs: i,
            rhs: c1,
        },
        SourceLocation::test(),
    ));
    body_block.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: j2,
            lhs: j,
            rhs: c1,
        },
        SourceLocation::test(),
    ));
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
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: a,
            lhs: x,
            rhs: y,
        },
        SourceLocation::test(),
    ));
    f.get_entry_mut()
        .set_terminator(Terminator::Jmp(next, vec![]));
    let next_block = f.get_block_mut(next);
    next_block.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: b,
            lhs: x,
            rhs: y,
        },
        SourceLocation::test(),
    ));
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
    entry.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: a,
            lhs: x,
            rhs: y,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: b,
            lhs: x,
            rhs: y,
        },
        SourceLocation::test(),
    ));
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
    e1_block.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: a,
            lhs: x,
            rhs: y,
        },
        SourceLocation::test(),
    ));
    e1_block.set_terminator(Terminator::Jmp(merge, vec![]));
    let e2_block = f.get_block_mut(e2);
    e2_block.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: b,
            lhs: x,
            rhs: y,
        },
        SourceLocation::test(),
    ));
    e2_block.set_terminator(Terminator::Jmp(merge, vec![]));
    let merge_block = f.get_block_mut(merge);
    merge_block.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: c,
            lhs: x,
            rhs: y,
        },
        SourceLocation::test(),
    ));
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
    entry.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: a,
            lhs: c2,
            rhs: c3,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![a, c5]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    assert!(cc.known_equal(fid, a, c5));
    assert_eq!(cc.leader(fid, a), Some(c5)); // the constant dominates everywhere
    assert_eq!(cc.leader(fid, c5), Some(c5));
}

/// Assert-vacuum soundness: `assert(x == 5)` pins `x` to 5 *conditionally* in dominated blocks,
/// but never unconditionally — so a global fold (SCS) can't fold `x` and vacuum the assert.
#[test]
fn assert_eq_const_is_conditional_not_unconditional() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c5 = ssa.add_const(Constant::U(32, 5));
    let x = ssa.fresh_value();

    let f = ssa.get_unique_entrypoint_mut();
    let entry_id = f.get_entry_id();
    let after = f.add_block();
    f.get_entry_mut().push_parameter(x, Type::u(32));
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: x,
            rhs: c5,
        },
        SourceLocation::test(),
    ));
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
        cc.asserted_const(fid, ProgramPoint::new(after, 0), x)
            .as_deref(),
        Some(&Constant::U(32, 5))
    );
    // ... and, at index granularity, in the asserting block itself *after* the assert (the assert is
    // instruction 0, so the terminator at index 1 sees the pin) ...
    assert_eq!(
        cc.asserted_const(fid, ProgramPoint::new(entry_id, 1), x)
            .as_deref(),
        Some(&Constant::U(32, 5))
    );
    // ... but never at the assert's own index, where folding `x` would vacuum the constraint.
    assert_eq!(
        cc.asserted_const(fid, ProgramPoint::new(entry_id, 0), x),
        None
    );
}

/// `assert(b)` proves `b == true` conditionally in dominated blocks, never unconditionally.
#[test]
fn assert_bool_is_conditional() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let b = ssa.fresh_value();

    let f = ssa.get_unique_entrypoint_mut();
    let entry_id = f.get_entry_id();
    let after = f.add_block();
    f.get_entry_mut().push_parameter(b, Type::u(1));
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::Assert { value: b },
        SourceLocation::test(),
    ));
    f.get_entry_mut()
        .set_terminator(Terminator::Jmp(after, vec![]));
    f.get_block_mut(after)
        .set_terminator(Terminator::Return(vec![]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    assert_eq!(cc.const_of(fid, b), None);
    assert_eq!(
        cc.asserted_const(fid, ProgramPoint::new(after, 0), b)
            .as_deref(),
        Some(&Constant::U(1, 1))
    );
    // Index granularity: the bool pin also holds after the `Assert` (instruction 0) in its own
    // block, but not at the assert's own index.
    assert_eq!(
        cc.asserted_const(fid, ProgramPoint::new(entry_id, 1), b)
            .as_deref(),
        Some(&Constant::U(1, 1))
    );
    assert_eq!(
        cc.asserted_const(fid, ProgramPoint::new(entry_id, 0), b),
        None
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
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: x,
            rhs: y,
        },
        SourceLocation::test(),
    ));
    f.get_entry_mut()
        .set_terminator(Terminator::Jmp(after, vec![]));
    f.get_block_mut(after)
        .set_terminator(Terminator::Return(vec![]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    assert!(cc.asserted_equal(fid, ProgramPoint::new(after, 0), x, y));
    assert!(cc.asserted_equal(fid, ProgramPoint::new(after, 0), y, x)); // symmetric
    // Index granularity: also holds after the assert (index 0) in its own block, symmetrically ...
    assert!(cc.asserted_equal(fid, ProgramPoint::new(entry_id, 1), x, y));
    assert!(cc.asserted_equal(fid, ProgramPoint::new(entry_id, 1), y, x));
    // ... but not at the assert's own index.
    assert!(!cc.asserted_equal(fid, ProgramPoint::new(entry_id, 0), x, y));
    // Structural congruence (unconditional) does not see the assert.
    assert!(!cc.known_equal(fid, x, y));
}

/// The asserted-equal leader is *dominance-aware*, not smallest-id: with `def(a)` dominating
/// `def(b)`, the leader of both is the dominating `a` even though `b` has the smaller value id.
#[test]
fn asserted_leader_picks_dominating_member() {
    let mut ssa = HLSSA::with_main("main".to_string());
    // `b` is allocated first so `b.0 < a.0`; a smallest-id leader would (wrongly) pick `b`.
    let b = ssa.fresh_value();
    let a = ssa.fresh_value();

    let f = ssa.get_unique_entrypoint_mut();
    let mid = f.add_block();
    f.get_entry_mut().push_parameter(a, Type::u(32)); // def(a): entry
    f.get_entry_mut()
        .set_terminator(Terminator::Jmp(mid, vec![]));
    let mid_block = f.get_block_mut(mid);
    mid_block.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            // index 0: def(b) in mid, dominated by entry
            kind: BinaryArithOpKind::Add,
            result: b,
            lhs: a,
            rhs: a,
        },
        SourceLocation::test(),
    ));
    mid_block.push_instruction(Located::with(
        OpCode::AssertCmp {
            // index 1: a == b (neither constant ⇒ an equality pair)
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    mid_block.set_terminator(Terminator::Return(vec![])); // index 2

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    assert!(b.0 < a.0, "test premise: b has the smaller value id");
    // After the assert (terminator index 2) the leader is the dominating definition `a`, for both
    // members — not the lower-id `b`.
    assert_eq!(
        cc.asserted_leader(fid, ProgramPoint::new(mid, 2), a),
        Some(a)
    );
    assert_eq!(
        cc.asserted_leader(fid, ProgramPoint::new(mid, 2), b),
        Some(a)
    );
    // Nothing at the assert's own index (the equality does not yet hold).
    assert_eq!(cc.asserted_leader(fid, ProgramPoint::new(mid, 1), a), None);
}

/// The leader is the representative of the *transitive* class: `a == b` and `b == c` make all three
/// share one leader, which is strictly stronger than the direct-pair `asserted_equal` check.
#[test]
fn asserted_leader_is_transitive() {
    let mut ssa = HLSSA::with_main("main".to_string());
    // Allocated in order, so `a.0 < b.0 < c.0`; all three are entry params (same def site), so the
    // dominance-root-most is the smallest-id `a`.
    let (a, b, c) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    let entry_id = f.get_entry_id();
    f.get_entry_mut().push_parameter(a, Type::u(32));
    f.get_entry_mut().push_parameter(b, Type::u(32));
    f.get_entry_mut().push_parameter(c, Type::u(32));
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::AssertCmp {
            // index 0: a == b
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::AssertCmp {
            // index 1: b == c
            kind: CmpKind::Eq,
            lhs: b,
            rhs: c,
        },
        SourceLocation::test(),
    ));
    f.get_entry_mut().set_terminator(Terminator::Return(vec![])); // index 2

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    // After both asserts the whole {a, b, c} class shares the leader `a`.
    let p2 = ProgramPoint::new(entry_id, 2);
    assert_eq!(cc.asserted_leader(fid, p2, a), Some(a));
    assert_eq!(cc.asserted_leader(fid, p2, b), Some(a));
    assert_eq!(cc.asserted_leader(fid, p2, c), Some(a));
    // The leader is strictly stronger than the direct-pair check: `a == c` is never asserted
    // directly, so `asserted_equal` does not see it, but the transitive leader does.
    assert!(!cc.asserted_equal(fid, p2, a, c));
    assert!(cc.asserted_equal(fid, p2, a, b));
    assert!(cc.asserted_equal(fid, p2, b, c));
    // Index granularity: at index 1 only `a == b` is in effect, so `c` is still on its own.
    let p1 = ProgramPoint::new(entry_id, 1);
    assert_eq!(cc.asserted_leader(fid, p1, a), Some(a));
    assert_eq!(cc.asserted_leader(fid, p1, c), None);
}

/// The leader holds at every index of a dominated block (cross-block `assert_eq`) and, within the
/// asserting block, only at use-indices strictly after the establishing assert.
#[test]
fn asserted_leader_index_granular() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (x, y) = (ssa.fresh_value(), ssa.fresh_value()); // x.0 < y.0 ⇒ leader x

    let f = ssa.get_unique_entrypoint_mut();
    let entry_id = f.get_entry_id();
    let after = f.add_block();
    f.get_entry_mut().push_parameter(x, Type::u(32));
    f.get_entry_mut().push_parameter(y, Type::u(32));
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::AssertCmp {
            // index 0
            kind: CmpKind::Eq,
            lhs: x,
            rhs: y,
        },
        SourceLocation::test(),
    ));
    f.get_entry_mut()
        .set_terminator(Terminator::Jmp(after, vec![])); // index 1
    f.get_block_mut(after)
        .set_terminator(Terminator::Return(vec![]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    // Cross-block: holds at every index of the dominated `after`.
    assert_eq!(
        cc.asserted_leader(fid, ProgramPoint::new(after, 0), x),
        Some(x)
    );
    assert_eq!(
        cc.asserted_leader(fid, ProgramPoint::new(after, 0), y),
        Some(x)
    );
    // Same block: only after the assert's index (the terminator at index 1) ...
    assert_eq!(
        cc.asserted_leader(fid, ProgramPoint::new(entry_id, 1), x),
        Some(x)
    );
    assert_eq!(
        cc.asserted_leader(fid, ProgramPoint::new(entry_id, 1), y),
        Some(x)
    );
    // ... never at the assert's own index.
    assert_eq!(
        cc.asserted_leader(fid, ProgramPoint::new(entry_id, 0), x),
        None
    );
    assert_eq!(
        cc.asserted_leader(fid, ProgramPoint::new(entry_id, 0), y),
        None
    );
}

/// An equality asserted on only one branch yields a leader within that branch (after the assert) but
/// nowhere it fails to dominate — not the merge, the other branch, or before the branch.
#[test]
fn asserted_leader_respects_dominance_fanout() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let cond = ssa.fresh_value();
    let (x, y) = (ssa.fresh_value(), ssa.fresh_value()); // x.0 < y.0 ⇒ leader x

    let f = ssa.get_unique_entrypoint_mut();
    let entry_id = f.get_entry_id();
    let then_b = f.add_block();
    let else_b = f.add_block();
    let merge = f.add_block();
    f.get_entry_mut().push_parameter(cond, Type::u(1));
    f.get_entry_mut().push_parameter(x, Type::u(32));
    f.get_entry_mut().push_parameter(y, Type::u(32));
    f.get_entry_mut()
        .set_terminator(Terminator::JmpIf(cond, then_b, else_b));
    f.get_block_mut(then_b).push_instruction(Located::with(
        OpCode::AssertCmp {
            // index 0: x == y, only on the `then` branch
            kind: CmpKind::Eq,
            lhs: x,
            rhs: y,
        },
        SourceLocation::test(),
    ));
    f.get_block_mut(then_b)
        .set_terminator(Terminator::Jmp(merge, vec![]));
    f.get_block_mut(else_b)
        .set_terminator(Terminator::Jmp(merge, vec![]));
    f.get_block_mut(merge)
        .set_terminator(Terminator::Return(vec![]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    // Within the asserting branch, after the assert: the leader holds.
    assert_eq!(
        cc.asserted_leader(fid, ProgramPoint::new(then_b, 1), x),
        Some(x)
    );
    assert_eq!(
        cc.asserted_leader(fid, ProgramPoint::new(then_b, 1), y),
        Some(x)
    );
    // The `then` branch dominates neither the merge nor the `else` branch, and the assert does not
    // hold at its own index or before the branch.
    assert_eq!(
        cc.asserted_leader(fid, ProgramPoint::new(merge, 0), x),
        None
    );
    assert_eq!(
        cc.asserted_leader(fid, ProgramPoint::new(else_b, 0), x),
        None
    );
    assert_eq!(
        cc.asserted_leader(fid, ProgramPoint::new(entry_id, 0), x),
        None
    );
    assert_eq!(
        cc.asserted_leader(fid, ProgramPoint::new(then_b, 0), x),
        None
    );
}

/// A value pinned by an `asserted_const` (one side constant) is in no equality *pair*, and a value
/// touched by no assert at all is in no class — both have no leader.
#[test]
fn asserted_leader_none_without_equality() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c5 = ssa.add_const(Constant::U(32, 5));
    let (x, z) = (ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    let entry_id = f.get_entry_id();
    f.get_entry_mut().push_parameter(x, Type::u(32));
    f.get_entry_mut().push_parameter(z, Type::u(32));
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::AssertCmp {
            // index 0: x == 5 — a constant pin, not an equality pair
            kind: CmpKind::Eq,
            lhs: x,
            rhs: c5,
        },
        SourceLocation::test(),
    ));
    f.get_entry_mut().set_terminator(Terminator::Return(vec![])); // index 1

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    let p1 = ProgramPoint::new(entry_id, 1);
    // Sanity: the constant pin is recorded as an `asserted_const`, not an equality.
    assert_eq!(
        cc.asserted_const(fid, p1, x).as_deref(),
        Some(&Constant::U(32, 5))
    );
    // So `x` has no asserted-equal leader, and the untouched `z` has none anywhere.
    assert_eq!(cc.asserted_leader(fid, p1, x), None);
    assert_eq!(cc.asserted_leader(fid, p1, z), None);
}

/// Even in a function that *does* have an equality class (`a == b`), a value in no pair (`z`) has
/// no leader — it gets no entry in the precomputed per-block table, so the lookup short-circuits to
/// `None`, while a participant `a` still resolves to its leader.
#[test]
fn asserted_leader_none_for_non_pair_value_amid_classes() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let (a, b, z) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    let entry_id = f.get_entry_id();
    f.get_entry_mut().push_parameter(a, Type::u(32));
    f.get_entry_mut().push_parameter(b, Type::u(32));
    f.get_entry_mut().push_parameter(z, Type::u(32)); // never in any assert
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::AssertCmp {
            // index 0: a == b — a real equality pair, so the function has a non-empty `def_key`
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    f.get_entry_mut().set_terminator(Terminator::Return(vec![])); // index 1

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    let p1 = ProgramPoint::new(entry_id, 1);
    // The participant resolves to its leader (the dominance-earliest, here the smaller-id `a`)...
    assert_eq!(cc.asserted_leader(fid, p1, a), Some(a));
    assert_eq!(cc.asserted_leader(fid, p1, b), Some(a));
    // ...while `z`, in no pair, short-circuits to `None`.
    assert_eq!(cc.asserted_leader(fid, p1, z), None);
}

/// Two same-block equality asserts at different indices grow one class in steps: a use between them
/// sees the smaller class's leader; a use after both sees the merged class's (lower) leader. This
/// exercises a value carrying more than one `(threshold, leader)` step and the index binary search.
#[test]
fn asserted_leader_multi_threshold_same_block() {
    let mut ssa = HLSSA::with_main("main".to_string());
    // Allocated in order ⇒ a.0 < c.0 < d.0; all entry params (same def site), so leaders are by id.
    let (a, c, d) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    let entry_id = f.get_entry_id();
    f.get_entry_mut().push_parameter(a, Type::u(32));
    f.get_entry_mut().push_parameter(c, Type::u(32));
    f.get_entry_mut().push_parameter(d, Type::u(32));
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::AssertCmp {
            // index 0: c == d ⇒ class {c, d}, leader c
            kind: CmpKind::Eq,
            lhs: c,
            rhs: d,
        },
        SourceLocation::test(),
    ));
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::AssertCmp {
            // index 1: a == c ⇒ merges into {a, c, d}, leader a
            kind: CmpKind::Eq,
            lhs: a,
            rhs: c,
        },
        SourceLocation::test(),
    ));
    f.get_entry_mut().set_terminator(Terminator::Return(vec![])); // index 2

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);

    // Between the asserts (index 1): only `c == d` is in effect, so `d`'s leader is `c`, and `a` —
    // which only the not-yet-effective second assert mentions — has no class yet.
    let p1 = ProgramPoint::new(entry_id, 1);
    assert_eq!(cc.asserted_leader(fid, p1, d), Some(c));
    assert_eq!(cc.asserted_leader(fid, p1, c), Some(c));
    assert_eq!(cc.asserted_leader(fid, p1, a), None);

    // After both (index 2): the merged class has the lower leader `a` — `d`'s second step.
    let p2 = ProgramPoint::new(entry_id, 2);
    assert_eq!(cc.asserted_leader(fid, p2, d), Some(a));
    assert_eq!(cc.asserted_leader(fid, p2, c), Some(a));
    assert_eq!(cc.asserted_leader(fid, p2, a), Some(a));
}

/// A same-block assert can merge two *distinct dominating* (cross-block) classes. The merged leader
/// is returned only at indices after that local assert; before it, each cross class keeps its own
/// leader.
#[test]
fn asserted_leader_local_merges_two_cross_classes() {
    let mut ssa = HLSSA::with_main("main".to_string());
    // a.0 < b.0 < c.0 < d.0; all entry params, so leaders are by id.
    let (a, b, c, d) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    let mid = f.add_block();
    f.get_entry_mut().push_parameter(a, Type::u(32));
    f.get_entry_mut().push_parameter(b, Type::u(32));
    f.get_entry_mut().push_parameter(c, Type::u(32));
    f.get_entry_mut().push_parameter(d, Type::u(32));
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::AssertCmp {
            // index 0: a == b ⇒ dominating class {a, b}
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::AssertCmp {
            // index 1: c == d ⇒ dominating class {c, d}
            kind: CmpKind::Eq,
            lhs: c,
            rhs: d,
        },
        SourceLocation::test(),
    ));
    f.get_entry_mut()
        .set_terminator(Terminator::Jmp(mid, vec![])); // index 2
    f.get_block_mut(mid).push_instruction(Located::with(
        OpCode::AssertCmp {
            // index 0 (in `mid`): b == c ⇒ merges the two cross classes into {a, b, c, d}
            kind: CmpKind::Eq,
            lhs: b,
            rhs: c,
        },
        SourceLocation::test(),
    ));
    f.get_block_mut(mid)
        .set_terminator(Terminator::Return(vec![])); // index 1

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);

    // At `mid` index 0 (before the local merge): the two cross classes are separate — `d`'s leader is
    // `c`, `b`'s is `a`.
    let mid0 = ProgramPoint::new(mid, 0);
    assert_eq!(cc.asserted_leader(fid, mid0, d), Some(c));
    assert_eq!(cc.asserted_leader(fid, mid0, b), Some(a));

    // At `mid` index 1 (after the local `b == c`): one class {a, b, c, d}, leader `a`.
    let mid1 = ProgramPoint::new(mid, 1);
    assert_eq!(cc.asserted_leader(fid, mid1, d), Some(a));
    assert_eq!(cc.asserted_leader(fid, mid1, c), Some(a));
    assert_eq!(cc.asserted_leader(fid, mid1, b), Some(a));
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
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Eq,
            result: eq,
            lhs: x,
            rhs: y,
        },
        SourceLocation::test(),
    ));
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
    entry.push_instruction(Located::with(
        OpCode::WriteWitness {
            result: Some(r1),
            value: v1,
            pinned: false,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::WriteWitness {
            result: Some(r2),
            value: v2,
            pinned: false,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::WriteWitness {
            result: Some(rp),
            value: v1,
            pinned: true,
        },
        SourceLocation::test(),
    ));
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
    entry.push_instruction(Located::with(
        OpCode::WriteWitness {
            result: Some(r),
            value: v,
            pinned: false,
        },
        SourceLocation::test(),
    ));
    // `value_of(r)` strips r's witness wrapper: w1, w2 are honestly equal to r. Two separate
    // reads stay distinct (ValueOf is excluded from value-numbering), so both join r's set.
    entry.push_instruction(Located::with(
        OpCode::Cast {
            result: w1,
            value: r,
            target: CastTarget::ValueOf,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::Cast {
            result: w2,
            value: r,
            target: CastTarget::ValueOf,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::WriteWitness {
            result: Some(r2),
            value: v2,
            pinned: false,
        },
        SourceLocation::test(),
    ));
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

/// An assert placed in the *last* block proves nothing by dominance (nothing follows it). The
/// post-dominance direction — which would carry its fact up to the earlier blocks it post-dominates
/// — was removed because it is unsound for loop-carried values (see `mod.rs`'s "Deferred
/// Improvements" and `post_dominating_assert_unsound_for_loop_carried_value`), so the
/// fact is *not* propagated to `entry`/`mid`. Index granularity still recovers it within the
/// asserting block.
#[test]
fn post_dominance_not_propagated_at_block_granularity() {
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
    f.get_block_mut(tail).push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: x,
            rhs: c5,
        },
        SourceLocation::test(),
    ));
    f.get_block_mut(tail)
        .set_terminator(Terminator::Return(vec![]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    // `tail` only post-dominates `mid`/`entry` (it does not dominate them), so its assert's fact is
    // withheld there now that the post-dominance direction is gone.
    assert_eq!(cc.asserted_const(fid, ProgramPoint::new(mid, 0), x), None);
    assert_eq!(
        cc.asserted_const(fid, ProgramPoint::new(entry_id, 0), x),
        None
    );
    // In the asserting block, index granularity still recovers it *after* the assert (index 0), but
    // not at the assert's own index.
    assert_eq!(
        cc.asserted_const(fid, ProgramPoint::new(tail, 1), x)
            .as_deref(),
        Some(&Constant::U(32, 5))
    );
    assert_eq!(cc.asserted_const(fid, ProgramPoint::new(tail, 0), x), None);
}

/// Regression for the soundness bug that motivated removing the post-dominance direction: a
/// loop-carried value asserted *after* the loop. `after` post-dominates `header`/`body` and
/// `def(v) = header` dominates them, yet `v` is `0..9` at the header on every iteration but the
/// last — so pinning `v = 10` there is false, and a constraint-preserving consumer could fold the
/// loop away and turn an accepting run rejecting. With post-dominance gone the fact is withheld at
/// `header` and `body` (the old code wrongly produced `Some(10)`).
#[test]
fn post_dominating_assert_unsound_for_loop_carried_value() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c0 = ssa.add_const(Constant::U(32, 0));
    let c1 = ssa.add_const(Constant::U(32, 1));
    let c10 = ssa.add_const(Constant::U(32, 10));
    let (v, lt, v1) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    let header = f.add_block();
    let body = f.add_block();
    let after = f.add_block();

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
    let after_block = f.get_block_mut(after);
    after_block.push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: v,
            rhs: c10,
        },
        SourceLocation::test(),
    ));
    after_block.set_terminator(Terminator::Return(vec![]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    // The (removed) post-dominance attribution would pin `v = 10` at the loop header/body; it must
    // not, regardless of `def(v)` being in scope there.
    assert_eq!(
        cc.asserted_const(fid, ProgramPoint::new(header, 0), v),
        None
    );
    assert_eq!(cc.asserted_const(fid, ProgramPoint::new(body, 0), v), None);
    // Sanity: within `after`, after the assert (index 0), index granularity still recovers it.
    assert_eq!(
        cc.asserted_const(fid, ProgramPoint::new(after, 1), v)
            .as_deref(),
        Some(&Constant::U(32, 10))
    );
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
    f.get_block_mut(then_b).push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: x,
            rhs: c5,
        },
        SourceLocation::test(),
    ));
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
    assert_eq!(
        cc.asserted_const(fid, ProgramPoint::new(entry_id, 0), x),
        None
    );
    assert_eq!(cc.asserted_const(fid, ProgramPoint::new(merge, 0), x), None);
    // In the asserting block itself the fact is still recovered after the assert (index
    // granularity), but never at the assert's own index.
    assert_eq!(
        cc.asserted_const(fid, ProgramPoint::new(then_b, 1), x)
            .as_deref(),
        Some(&Constant::U(32, 5))
    );
    assert_eq!(
        cc.asserted_const(fid, ProgramPoint::new(then_b, 0), x),
        None
    );
}

/// A use in the terminator sits at `index == instruction count`, so the asserting block's own pin
/// reaches a value the terminator forwards (here a return), proving the index convention.
#[test]
fn local_assert_reaches_terminator_use() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c5 = ssa.add_const(Constant::U(32, 5));
    let (x, doubled) = (ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    let entry_id = f.get_entry_id();
    f.get_entry_mut().push_parameter(x, Type::u(32));
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::AssertCmp {
            // index 0
            kind: CmpKind::Eq,
            lhs: x,
            rhs: c5,
        },
        SourceLocation::test(),
    ));
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::BinaryArithOp {
            // index 1
            kind: BinaryArithOpKind::Add,
            result: doubled,
            lhs: x,
            rhs: x,
        },
        SourceLocation::test(),
    ));
    f.get_entry_mut()
        .set_terminator(Terminator::Return(vec![doubled])); // terminator: index 2

    let fid = ssa.get_unique_entrypoint_id();
    let term_index = ssa
        .get_unique_entrypoint()
        .get_entry()
        .get_instructions()
        .count();
    assert_eq!(term_index, 2);
    let cc = run_in_test(&ssa);
    // The pin holds at the terminator's index (== instruction count) and at the add after it ...
    assert_eq!(
        cc.asserted_const(fid, ProgramPoint::new(entry_id, term_index), x)
            .as_deref(),
        Some(&Constant::U(32, 5))
    );
    assert_eq!(
        cc.asserted_const(fid, ProgramPoint::new(entry_id, 1), x)
            .as_deref(),
        Some(&Constant::U(32, 5))
    );
    // ... but not at the assert's own index.
    assert_eq!(
        cc.asserted_const(fid, ProgramPoint::new(entry_id, 0), x),
        None
    );
}

/// Two asserts on different values in one block are each recovered only after their own index.
#[test]
fn multiple_local_asserts_each_recovered_after_its_index() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c5 = ssa.add_const(Constant::U(32, 5));
    let c6 = ssa.add_const(Constant::U(32, 6));
    let (x, y) = (ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    let entry_id = f.get_entry_id();
    f.get_entry_mut().push_parameter(x, Type::u(32));
    f.get_entry_mut().push_parameter(y, Type::u(32));
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::AssertCmp {
            // index 0: x == 5
            kind: CmpKind::Eq,
            lhs: x,
            rhs: c5,
        },
        SourceLocation::test(),
    ));
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::AssertCmp {
            // index 1: y == 6
            kind: CmpKind::Eq,
            lhs: y,
            rhs: c6,
        },
        SourceLocation::test(),
    ));
    f.get_entry_mut().set_terminator(Terminator::Return(vec![]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    // At index 1 only `x == 5` is in effect (the `y` assert has not been passed yet).
    assert_eq!(
        cc.asserted_const(fid, ProgramPoint::new(entry_id, 1), x)
            .as_deref(),
        Some(&Constant::U(32, 5))
    );
    assert_eq!(
        cc.asserted_const(fid, ProgramPoint::new(entry_id, 1), y),
        None
    );
    // At index 2 both hold.
    assert_eq!(
        cc.asserted_const(fid, ProgramPoint::new(entry_id, 2), x)
            .as_deref(),
        Some(&Constant::U(32, 5))
    );
    assert_eq!(
        cc.asserted_const(fid, ProgramPoint::new(entry_id, 2), y)
            .as_deref(),
        Some(&Constant::U(32, 6))
    );
}

/// Two contradictory asserts in one block: a use after both sees the *earliest* (first-writer-wins,
/// the instruction order is the deterministic tie-break), and the first assert's own operand is
/// never informed by the later one.
#[test]
fn contradictory_local_asserts_first_writer_wins() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c5 = ssa.add_const(Constant::U(32, 5));
    let c7 = ssa.add_const(Constant::U(32, 7));
    let x = ssa.fresh_value();

    let f = ssa.get_unique_entrypoint_mut();
    let entry_id = f.get_entry_id();
    f.get_entry_mut().push_parameter(x, Type::u(32));
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::AssertCmp {
            // index 0: x == 5
            kind: CmpKind::Eq,
            lhs: x,
            rhs: c5,
        },
        SourceLocation::test(),
    ));
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::AssertCmp {
            // index 1: x == 7
            kind: CmpKind::Eq,
            lhs: x,
            rhs: c7,
        },
        SourceLocation::test(),
    ));
    f.get_entry_mut().set_terminator(Terminator::Return(vec![]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    // The first assert's own operand (index 0) is never informed.
    assert_eq!(
        cc.asserted_const(fid, ProgramPoint::new(entry_id, 0), x),
        None
    );
    // Between the two asserts (index 1) only the first is in effect.
    assert_eq!(
        cc.asserted_const(fid, ProgramPoint::new(entry_id, 1), x)
            .as_deref(),
        Some(&Constant::U(32, 5))
    );
    // After both, first-writer-wins keeps the earliest establisher.
    assert_eq!(
        cc.asserted_const(fid, ProgramPoint::new(entry_id, 2), x)
            .as_deref(),
        Some(&Constant::U(32, 5))
    );
}

/// A block that is both dominated by an outer assert *and* makes its own asserts: the cross-block
/// fact holds at *every* index (including before the local re-assert — the documented
/// before-the-assert behavior), while a local fact holds only after its own index; a cross-block
/// fact also takes precedence over a contradictory local one.
#[test]
fn entry_and_local_assert_facts_layer() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c5 = ssa.add_const(Constant::U(32, 5));
    let c6 = ssa.add_const(Constant::U(32, 6));
    let c7 = ssa.add_const(Constant::U(32, 7));
    let (a, b) = (ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    let inner = f.add_block();
    f.get_entry_mut().push_parameter(a, Type::u(32));
    f.get_entry_mut().push_parameter(b, Type::u(32));
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::AssertCmp {
            // outer: a == 5
            kind: CmpKind::Eq,
            lhs: a,
            rhs: c5,
        },
        SourceLocation::test(),
    ));
    f.get_entry_mut()
        .set_terminator(Terminator::Jmp(inner, vec![]));
    let inner_block = f.get_block_mut(inner);
    inner_block.push_instruction(Located::with(
        OpCode::AssertCmp {
            // index 0: b == 6
            kind: CmpKind::Eq,
            lhs: b,
            rhs: c6,
        },
        SourceLocation::test(),
    ));
    inner_block.push_instruction(Located::with(
        OpCode::AssertCmp {
            // index 1: a == 7 (contradicts the dominating outer assert)
            kind: CmpKind::Eq,
            lhs: a,
            rhs: c7,
        },
        SourceLocation::test(),
    ));
    inner_block.set_terminator(Terminator::Return(vec![]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    // The outer (cross-block) pin on `a` holds at every index of `inner` — before the local
    // re-assert and after it — and wins over the contradictory local `a == 7`.
    assert_eq!(
        cc.asserted_const(fid, ProgramPoint::new(inner, 0), a)
            .as_deref(),
        Some(&Constant::U(32, 5))
    );
    assert_eq!(
        cc.asserted_const(fid, ProgramPoint::new(inner, 2), a)
            .as_deref(),
        Some(&Constant::U(32, 5))
    );
    // The local pin on `b` is the before-the-assert gap: absent at index 0, present after index 0.
    assert_eq!(cc.asserted_const(fid, ProgramPoint::new(inner, 0), b), None);
    assert_eq!(
        cc.asserted_const(fid, ProgramPoint::new(inner, 1), b)
            .as_deref(),
        Some(&Constant::U(32, 6))
    );
}

/// The asserted operand is defined earlier in the same block: local facts need no `in_scope` check
/// (the operand is necessarily defined before the assert), and the pin is recovered after it.
#[test]
fn local_assert_on_value_defined_earlier_in_block() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c10 = ssa.add_const(Constant::U(32, 10));
    let (p, y) = (ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    let entry_id = f.get_entry_id();
    f.get_entry_mut().push_parameter(p, Type::u(32));
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::BinaryArithOp {
            // index 0: y = p + p
            kind: BinaryArithOpKind::Add,
            result: y,
            lhs: p,
            rhs: p,
        },
        SourceLocation::test(),
    ));
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::AssertCmp {
            // index 1: y == 10
            kind: CmpKind::Eq,
            lhs: y,
            rhs: c10,
        },
        SourceLocation::test(),
    ));
    f.get_entry_mut()
        .set_terminator(Terminator::Return(vec![y])); // index 2

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    // At/before the assert (index 1): no fact. After it (the return at index 2 forwards `y`): 10.
    assert_eq!(
        cc.asserted_const(fid, ProgramPoint::new(entry_id, 1), y),
        None
    );
    assert_eq!(
        cc.asserted_const(fid, ProgramPoint::new(entry_id, 2), y)
            .as_deref(),
        Some(&Constant::U(32, 10))
    );
}

/// An assert inside a loop body is recovered after it within the body, and is not mis-attributed to
/// the loop header (which the body neither dominates nor post-dominates).
#[test]
fn local_assert_in_loop_body() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let c0 = ssa.add_const(Constant::U(32, 0));
    let c1 = ssa.add_const(Constant::U(32, 1));
    let (n, i, lt, next) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    let f = ssa.get_unique_entrypoint_mut();
    let header = f.add_block();
    let body = f.add_block();
    let exit = f.add_block();
    f.get_entry_mut().push_parameter(n, Type::u(32));
    f.get_entry_mut()
        .set_terminator(Terminator::Jmp(header, vec![c0]));
    let header_block = f.get_block_mut(header);
    header_block.push_parameter(i, Type::u(32));
    header_block.push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Lt,
            result: lt,
            lhs: i,
            rhs: n,
        },
        SourceLocation::test(),
    ));
    header_block.set_terminator(Terminator::JmpIf(lt, body, exit));
    let body_block = f.get_block_mut(body);
    // `next = i + 1` keeps `i` loop-variant (else it would fold to the constant 0 and the assert
    // would degrade to a constant pin rather than the pure equality this test exercises).
    body_block.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            // index 0
            kind: BinaryArithOpKind::Add,
            result: next,
            lhs: i,
            rhs: c1,
        },
        SourceLocation::test(),
    ));
    body_block.push_instruction(Located::with(
        OpCode::AssertCmp {
            // index 1: i == n
            kind: CmpKind::Eq,
            lhs: i,
            rhs: n,
        },
        SourceLocation::test(),
    ));
    body_block.set_terminator(Terminator::Jmp(header, vec![next]));
    f.get_block_mut(exit)
        .set_terminator(Terminator::Return(vec![]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);
    // Recovered after the assert (index 1) in the body, not at or before its own index.
    assert!(cc.asserted_equal(fid, ProgramPoint::new(body, 2), i, n));
    assert!(!cc.asserted_equal(fid, ProgramPoint::new(body, 1), i, n));
    assert!(!cc.asserted_equal(fid, ProgramPoint::new(body, 0), i, n));
    // Not mis-attributed to the header (body neither dominates nor post-dominates it).
    assert!(!cc.asserted_equal(fid, ProgramPoint::new(header, 0), i, n));
    assert!(!cc.asserted_equal(fid, ProgramPoint::new(header, 2), i, n));
}

/// A callee that returns a constant: the call result folds interprocedurally in the caller's
/// context, while the SCS-visible intraprocedural view leaves it `Bottom`.
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
        entry.push_instruction(Located::with(
            OpCode::Call {
                results: vec![r],
                function: CallTarget::Static(helper),
                args: vec![],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
        entry.set_terminator(Terminator::Return(vec![r]));
    }

    let cc = run_in_test(&ssa);
    assert_eq!(
        cc.const_of_in(main_id, &Context::empty(), r).as_deref(),
        Some(&Constant::Field(Field::from(5u64)))
    );
    // The intraprocedural view (what SCS reads) never folds a call result.
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
        entry.push_instruction(Located::with(
            OpCode::Call {
                results: vec![r],
                function: CallTarget::Static(id),
                args: vec![c7],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
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
        entry.set_terminator(Terminator::Return(vec![eq]));
    }
    {
        let mf = ssa.get_function_mut(main_id);
        mf.add_return_type(Type::u(1));
        mf.get_entry_mut().push_parameter(x0, Type::u(32));
        let entry = mf.get_entry_mut();
        entry.push_instruction(Located::with(
            OpCode::Call {
                results: vec![r],
                function: CallTarget::Static(g),
                args: vec![x0],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
        entry.set_terminator(Terminator::Return(vec![r]));
    }

    let cc = run_in_test(&ssa);
    // The summary return-jump is `Const(true)`, so the call result folds in the caller context.
    assert_eq!(
        cc.const_of_in(main_id, &Context::empty(), r).as_deref(),
        Some(&Constant::U(1, 1))
    );
    // The intraprocedural view (what SCS reads) never folds a call result.
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
        entry.set_terminator(Terminator::Return(vec![x]));
    }
    {
        let mf = ssa.get_function_mut(main_id);
        mf.add_return_type(Type::u(32));
        mf.get_entry_mut().push_parameter(x0, Type::u(32));
        let entry = mf.get_entry_mut();
        entry.push_instruction(Located::with(
            OpCode::Call {
                results: vec![r],
                function: CallTarget::Static(g),
                args: vec![x0],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
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
        entry.push_instruction(Located::with(
            OpCode::Call {
                results: vec![t],
                function: CallTarget::Static(g),
                args: vec![x],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
        entry.set_terminator(Terminator::Return(vec![eq]));
    }
    {
        let mf = ssa.get_function_mut(main_id);
        mf.add_return_type(Type::u(1));
        mf.get_entry_mut().push_parameter(x0, Type::u(32));
        let entry = mf.get_entry_mut();
        entry.push_instruction(Located::with(
            OpCode::Call {
                results: vec![r],
                function: CallTarget::Static(g),
                args: vec![x0],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
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

/// The context-parameterized conditional queries: the branch-fact family (`const_in_block_in`)
/// is context-*precise* — it sees the per-context parameter constant the intraprocedural query
/// cannot. (The assert/disequality/witness family is likewise rebuilt per context; see the
/// `*_in` conditional tests below.)
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
        entry.push_instruction(Located::with(
            OpCode::Call {
                results: vec![r],
                function: CallTarget::Static(id),
                args: vec![c7],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
        entry.set_terminator(Terminator::Return(vec![r]));
    }

    let entry_id = ssa.get_function(id).get_entry_id();
    let cc = run_in_test(&ssa);
    let ctx = sole_context(&cc, id);

    // Intraprocedurally `p` is unconstrained; under the single call context it is the constant 7.
    assert_eq!(cc.const_in_block(id, entry_id, p), None);
    assert_eq!(
        cc.const_in_block_in(id, &ctx, entry_id, p).as_deref(),
        Some(&Constant::Field(Field::from(7u64)))
    );
}

/// Per-context conditional facts can be more precise: `assert(x == p)` is only an asserted
/// *equality* intraprocedurally (`p` opaque), but pins `x` to an asserted *constant* in a context
/// where `p` is a lattice constant. The pair then *migrates* channels — it leaves the per-context
/// eq channel, so `asserted_equal_in` is deliberately not a pointwise superset of `asserted_equal`.
#[test]
fn asserted_const_refines_per_context() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let g = ssa.add_function("g".to_string());
    let (x, p) = (ssa.fresh_value(), ssa.fresh_value());
    let c7 = ssa.add_const(Constant::Field(Field::from(7u64)));
    let x0 = ssa.fresh_value();

    let after = {
        let hf = ssa.get_function_mut(g);
        let after = hf.add_block();
        hf.get_entry_mut().push_parameter(x, Type::field());
        hf.get_entry_mut().push_parameter(p, Type::field());
        hf.get_entry_mut().push_instruction(Located::with(
            OpCode::AssertCmp {
                kind: CmpKind::Eq,
                lhs: x,
                rhs: p,
            },
            SourceLocation::test(),
        ));
        hf.get_entry_mut()
            .set_terminator(Terminator::Jmp(after, vec![]));
        hf.get_block_mut(after)
            .set_terminator(Terminator::Return(vec![]));
        after
    };
    {
        let mf = ssa.get_function_mut(main_id);
        let entry = mf.get_entry_mut();
        entry.push_parameter(x0, Type::field());
        entry.push_instruction(Located::with(
            OpCode::Call {
                results: vec![],
                function: CallTarget::Static(g),
                args: vec![x0, c7],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
        entry.set_terminator(Terminator::Return(vec![]));
    }

    let cc = run_in_test(&ssa);
    let ctx = sole_context(&cc, g);
    let pp = ProgramPoint::new(after, 0);

    // Intraprocedurally `p` is opaque: the assert is an equality, not a constant pin.
    assert_eq!(cc.asserted_const(g, pp, x), None);
    assert!(cc.asserted_equal(g, pp, x, p));
    // In the context `p` is the constant 7, so the same assert pins `x` to it...
    assert_eq!(
        cc.asserted_const_in(g, &ctx, pp, x).as_deref(),
        Some(&Constant::Field(Field::from(7u64)))
    );
    // ...and the pair has migrated out of the per-context eq channel.
    assert!(!cc.asserted_equal_in(g, &ctx, pp, x, p));
}

/// `witness_forward_in` drops a forward established in a context-dead block: the witness scan is
/// reachability-gated, so a forward the intraprocedural view reports can be absent per context —
/// the hazard that forbids `witness_forward_in` from delegating to the intraprocedural map. The
/// other direction exists too; see `context_constant_can_split_congruence`
#[test]
fn witness_forward_in_prunes_context_dead_forward() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let g = ssa.add_function("g".to_string());
    let (p, w, r) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let c_true = ssa.add_const(Constant::U(1, 1));
    let w0 = ssa.fresh_value();

    {
        let hf = ssa.get_function_mut(g);
        let then_b = hf.add_block();
        let else_b = hf.add_block();
        let after = hf.add_block();
        hf.get_entry_mut().push_parameter(p, Type::u(1));
        hf.get_entry_mut().push_parameter(w, Type::u(32));
        hf.get_entry_mut()
            .set_terminator(Terminator::JmpIf(p, then_b, else_b));
        hf.get_block_mut(then_b)
            .set_terminator(Terminator::Jmp(after, vec![]));
        // The false arm establishes the forward `r → w`; it is dead once a context pins `p`.
        hf.get_block_mut(else_b).push_instruction(Located::with(
            OpCode::WriteWitness {
                result: Some(r),
                value: w,
                pinned: false,
            },
            SourceLocation::test(),
        ));
        hf.get_block_mut(else_b)
            .set_terminator(Terminator::Jmp(after, vec![]));
        hf.get_block_mut(after)
            .set_terminator(Terminator::Return(vec![]));
    }
    {
        let mf = ssa.get_function_mut(main_id);
        let entry = mf.get_entry_mut();
        entry.push_parameter(w0, Type::u(32));
        entry.push_instruction(Located::with(
            OpCode::Call {
                results: vec![],
                function: CallTarget::Static(g),
                args: vec![c_true, w0],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
        entry.set_terminator(Terminator::Return(vec![]));
    }

    let cc = run_in_test(&ssa);
    let ctx = sole_context(&cc, g);

    // Intraprocedurally `p` is opaque, both arms live: the forward is gathered.
    assert_eq!(cc.witness_forward(g, r), [w].as_slice());
    // In the context `p = true` kills the false arm — and its forward with it.
    assert!(cc.witness_forward_in(g, &ctx, r).is_empty());
}

/// Parity when nothing refines: under `main`'s empty context the seeds are Bottom, and `main` has
/// no static calls (the per-context solve enables the eval-call summary channel the
/// intraprocedural one keeps off, so a folding call would break the "nothing refines" premise) —
/// every per-context conditional channel coincides with its intraprocedural counterpart.
#[test]
fn conditional_queries_parity_under_empty_context() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let (x, a, b) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let (d, e) = (ssa.fresh_value(), ssa.fresh_value());
    let (eq, r) = (ssa.fresh_value(), ssa.fresh_value());
    let c5 = ssa.add_const(Constant::U(32, 5));

    let f = ssa.get_unique_entrypoint_mut();
    let then_b = f.add_block();
    let else_b = f.add_block();
    let after = f.add_block();
    for v in [x, a, b, d, e] {
        f.get_entry_mut().push_parameter(v, Type::u(32));
    }
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: x,
            rhs: c5,
        },
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
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::WriteWitness {
            result: Some(r),
            value: x,
            pinned: false,
        },
        SourceLocation::test(),
    ));
    f.get_entry_mut().push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Eq,
            result: eq,
            lhs: d,
            rhs: e,
        },
        SourceLocation::test(),
    ));
    f.get_entry_mut()
        .set_terminator(Terminator::JmpIf(eq, then_b, else_b));
    f.get_block_mut(then_b)
        .set_terminator(Terminator::Jmp(after, vec![]));
    f.get_block_mut(else_b)
        .set_terminator(Terminator::Jmp(after, vec![]));
    f.get_block_mut(after)
        .set_terminator(Terminator::Return(vec![]));

    let cc = run_in_test(&ssa);
    let ctx = Context::empty();
    let pp = ProgramPoint::new(after, 0);

    // Assert-const channel.
    assert_eq!(
        cc.asserted_const(main_id, pp, x).as_deref(),
        Some(&Constant::U(32, 5))
    );
    assert_eq!(
        cc.asserted_const_in(main_id, &ctx, pp, x),
        cc.asserted_const(main_id, pp, x)
    );
    // Assert-eq channel and its leaders.
    assert!(cc.asserted_equal(main_id, pp, a, b));
    assert!(cc.asserted_equal_in(main_id, &ctx, pp, a, b));
    assert_eq!(cc.asserted_leader(main_id, pp, b), Some(a));
    assert_eq!(
        cc.asserted_leader_in(main_id, &ctx, pp, b),
        cc.asserted_leader(main_id, pp, b)
    );
    // Disequality channel (the false edge is the sole in-edge of `else_b` in both views).
    assert!(cc.known_unequal(main_id, else_b, d, e));
    assert!(cc.known_unequal_in(main_id, &ctx, else_b, d, e));
    // Witness channel.
    assert_eq!(cc.witness_forward(main_id, r), [x].as_slice());
    assert_eq!(
        cc.witness_forward_in(main_id, &ctx, r),
        cc.witness_forward(main_id, r)
    );
}

/// A context can *create* a disequality: pinning the branch predicate prunes one predecessor of
/// the false-edge target, making the false edge its sole executable in-edge — so `known_unequal_in`
/// holds where the intraprocedural `known_unequal` (two live in-edges) cannot.
#[test]
fn known_unequal_in_gains_from_pruned_predecessor() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let g = ssa.add_function("g".to_string());
    let (p, d, e, eq) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );
    let c_false = ssa.add_const(Constant::U(1, 0));
    let (d0, e0) = (ssa.fresh_value(), ssa.fresh_value());

    let join = {
        let hf = ssa.get_function_mut(g);
        let side = hf.add_block();
        let main_b = hf.add_block();
        let then_b = hf.add_block();
        let join = hf.add_block();
        hf.get_entry_mut().push_parameter(p, Type::u(1));
        hf.get_entry_mut().push_parameter(d, Type::u(32));
        hf.get_entry_mut().push_parameter(e, Type::u(32));
        hf.get_entry_mut().push_instruction(Located::with(
            OpCode::Cmp {
                kind: CmpKind::Eq,
                result: eq,
                lhs: d,
                rhs: e,
            },
            SourceLocation::test(),
        ));
        hf.get_entry_mut()
            .set_terminator(Terminator::JmpIf(p, side, main_b));
        // `side` is `join`'s second predecessor; it dies once a context pins `p = false`.
        hf.get_block_mut(side)
            .set_terminator(Terminator::Jmp(join, vec![]));
        hf.get_block_mut(main_b)
            .set_terminator(Terminator::JmpIf(eq, then_b, join));
        hf.get_block_mut(then_b)
            .set_terminator(Terminator::Return(vec![]));
        hf.get_block_mut(join)
            .set_terminator(Terminator::Return(vec![]));
        join
    };
    {
        let mf = ssa.get_function_mut(main_id);
        let entry = mf.get_entry_mut();
        entry.push_parameter(d0, Type::u(32));
        entry.push_parameter(e0, Type::u(32));
        entry.push_instruction(Located::with(
            OpCode::Call {
                results: vec![],
                function: CallTarget::Static(g),
                args: vec![c_false, d0, e0],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
        entry.set_terminator(Terminator::Return(vec![]));
    }

    let cc = run_in_test(&ssa);
    let ctx = sole_context(&cc, g);

    // Intraprocedurally `join` has two live in-edges (`side` and the false edge): no diseq.
    assert!(!cc.known_unequal(g, join, d, e));
    // The context pins `p = false`, killing `side`: the false edge becomes `join`'s sole
    // executable in-edge, and the disequality appears.
    assert!(cc.known_unequal_in(g, &ctx, join, d, e));
}

/// An assert pinning a value to an *aggregate* constant contributes no conditional fact: `Blob`
/// constants are never surfaced (the module "Aggregate Folding" contract), so the pin enters
/// neither the assert-const channel — whose consumers materialise the constant — nor the eq
/// channel, whose pairs must be constant-free.
#[test]
fn asserted_const_never_aggregate() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let c0 = ssa.add_const(Constant::U(32, 10));
    let c1 = ssa.add_const(Constant::U(32, 20));
    let (x, s) = (ssa.fresh_value(), ssa.fresh_value());

    let f = ssa.get_unique_entrypoint_mut();
    let after = f.add_block();
    f.get_entry_mut().push_parameter(x, Type::u(32).array_of(2));
    let entry = f.get_entry_mut();
    // `s` aggregate-folds to a lattice `Const(Blob)`, so the assert pins `x` to a Blob.
    entry.push_instruction(Located::with(
        OpCode::MkSeq {
            result: s,
            elems: vec![c0, c1],
            seq_type: SequenceTargetType::Array(2),
            elem_type: Type::u(32),
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: x,
            rhs: s,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Jmp(after, vec![]));
    f.get_block_mut(after)
        .set_terminator(Terminator::Return(vec![]));

    let cc = run_in_test(&ssa);
    let pp = ProgramPoint::new(after, 0);

    // The Blob must not surface through the conditional channel...
    assert_eq!(cc.asserted_const(main_id, pp, x), None);
    // ...nor reroute into the eq channel as a constant-sided pair.
    assert!(!cc.asserted_equal(main_id, pp, x, s));
}

/// The per-context analog: a callee parameter seeded with the caller's constant-aggregate argument
/// pins the asserted side to a `Blob` in that context. The pin is dropped there too — per context
/// the assert contributes *no* fact at all (the intraprocedural eq view keeps the pair), one of the
/// fact-losing directions the impl docs call out.
#[test]
fn asserted_const_in_never_aggregate() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let g = ssa.add_function("g".to_string());
    let c0 = ssa.add_const(Constant::U(32, 10));
    let c1 = ssa.add_const(Constant::U(32, 20));
    let (a, x) = (ssa.fresh_value(), ssa.fresh_value());
    let (s, y) = (ssa.fresh_value(), ssa.fresh_value());

    let after = {
        let hf = ssa.get_function_mut(g);
        let after = hf.add_block();
        hf.get_entry_mut()
            .push_parameter(a, Type::u(32).array_of(2));
        hf.get_entry_mut()
            .push_parameter(x, Type::u(32).array_of(2));
        hf.get_entry_mut().push_instruction(Located::with(
            OpCode::AssertCmp {
                kind: CmpKind::Eq,
                lhs: x,
                rhs: a,
            },
            SourceLocation::test(),
        ));
        hf.get_entry_mut()
            .set_terminator(Terminator::Jmp(after, vec![]));
        hf.get_block_mut(after)
            .set_terminator(Terminator::Return(vec![]));
        after
    };
    {
        let mf = ssa.get_function_mut(main_id);
        let entry = mf.get_entry_mut();
        entry.push_parameter(y, Type::u(32).array_of(2));
        entry.push_instruction(Located::with(
            OpCode::MkSeq {
                result: s,
                elems: vec![c0, c1],
                seq_type: SequenceTargetType::Array(2),
                elem_type: Type::u(32),
            },
            SourceLocation::test(),
        ));
        entry.push_instruction(Located::with(
            OpCode::Call {
                results: vec![],
                function: CallTarget::Static(g),
                args: vec![s, y],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
        entry.set_terminator(Terminator::Return(vec![]));
    }

    let cc = run_in_test(&ssa);
    let ctx = sole_context(&cc, g);
    let pp = ProgramPoint::new(after, 0);

    // Intraprocedurally both sides are opaque: an ordinary asserted equality.
    assert!(cc.asserted_equal(g, pp, x, a));
    assert_eq!(cc.asserted_const(g, pp, x), None);
    // In the context `a` is a Blob: the pin surfaces through no per-context channel.
    assert_eq!(cc.asserted_const_in(g, &ctx, pp, x), None);
    assert!(!cc.asserted_equal_in(g, &ctx, pp, x, a));
}

/// When the *asserted* side itself becomes a per-context constant, the fact leaves
/// `asserted_const_in` — but only because it migrated to the strictly stronger *unconditional*
/// per-context channel, where `const_of_in` answers it without any constraint-preservation proviso.
/// Nothing is lost; the documented "no pointwise inclusion" is this migration.
#[test]
fn asserted_const_migrates_to_unconditional_channel() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let g = ssa.add_function("g".to_string());
    let x = ssa.fresh_value();
    let c7 = ssa.add_const(Constant::Field(Field::from(7u64)));

    let after = {
        let hf = ssa.get_function_mut(g);
        let after = hf.add_block();
        hf.get_entry_mut().push_parameter(x, Type::field());
        hf.get_entry_mut().push_instruction(Located::with(
            OpCode::AssertCmp {
                kind: CmpKind::Eq,
                lhs: x,
                rhs: c7,
            },
            SourceLocation::test(),
        ));
        hf.get_entry_mut()
            .set_terminator(Terminator::Jmp(after, vec![]));
        hf.get_block_mut(after)
            .set_terminator(Terminator::Return(vec![]));
        after
    };
    {
        let mf = ssa.get_function_mut(main_id);
        let entry = mf.get_entry_mut();
        entry.push_instruction(Located::with(
            OpCode::Call {
                results: vec![],
                function: CallTarget::Static(g),
                args: vec![c7],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
        entry.set_terminator(Terminator::Return(vec![]));
    }

    let cc = run_in_test(&ssa);
    let ctx = sole_context(&cc, g);
    let pp = ProgramPoint::new(after, 0);

    // Intraprocedurally `x` is opaque, so the assert pins it conditionally.
    assert_eq!(
        cc.asserted_const(g, pp, x).as_deref(),
        Some(&Constant::Field(Field::from(7u64)))
    );
    // In the context the seed already proves `x = 7` unconditionally: the conditional entry is
    // gone...
    assert_eq!(cc.asserted_const_in(g, &ctx, pp, x), None);
    // ...because the unconditional channel now carries it.
    assert_eq!(
        cc.const_of_in(g, &ctx, x).as_deref(),
        Some(&Constant::Field(Field::from(7u64)))
    );
}

/// A per-context constant can *split* an intraprocedurally-proven congruence. Intraprocedurally
/// `x = call g(a)` is value-numbered equal to `y = a*3` via `g`'s grafted symbolic return, the
/// writeback folds the `x == y` branch, and the else arm is dead — its forward never gathered. Per
/// context `a = 5` folds `y` to a constant, whose `Const` label replaces the structural one, while
/// the call result `x` (lattice-opaque) keeps its `Op` class: the classes split, the branch stays
/// live, and the forward appears in `witness_forward_in` only. Reachability-derived facts can thus
/// flow in the *growing* direction per context — the reason the impl docs promise no inclusion
/// between the two views in either direction.
#[test]
fn context_constant_can_split_congruence() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let f = ssa.add_function("f".to_string());
    let g = ssa.add_function("g".to_string());
    let c3 = ssa.add_const(Constant::Field(Field::from(3u64)));
    let c5 = ssa.add_const(Constant::Field(Field::from(5u64)));
    let (p, m) = (ssa.fresh_value(), ssa.fresh_value());
    let (a, w, x, y, eq, r) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );
    let w0 = ssa.fresh_value();

    // g(p) { return p * 3 } — pure, so its symbolic return `Mul(Param0, 3)` grafts.
    {
        let hf = ssa.get_function_mut(g);
        hf.add_return_type(Type::field());
        hf.get_entry_mut().push_parameter(p, Type::field());
        hf.get_entry_mut().push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Mul,
                result: m,
                lhs: p,
                rhs: c3,
            },
            SourceLocation::test(),
        ));
        hf.get_entry_mut()
            .set_terminator(Terminator::Return(vec![m]));
    }
    // f(a, w) { x = g(a); y = a*3; if x == y { } else { r = write_witness(w) } }
    {
        let hf = ssa.get_function_mut(f);
        let then_b = hf.add_block();
        let else_b = hf.add_block();
        let after = hf.add_block();
        hf.get_entry_mut().push_parameter(a, Type::field());
        hf.get_entry_mut().push_parameter(w, Type::u(32));
        hf.get_entry_mut().push_instruction(Located::with(
            OpCode::Call {
                results: vec![x],
                function: CallTarget::Static(g),
                args: vec![a],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
        hf.get_entry_mut().push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Mul,
                result: y,
                lhs: a,
                rhs: c3,
            },
            SourceLocation::test(),
        ));
        hf.get_entry_mut().push_instruction(Located::with(
            OpCode::Cmp {
                kind: CmpKind::Eq,
                result: eq,
                lhs: x,
                rhs: y,
            },
            SourceLocation::test(),
        ));
        hf.get_entry_mut()
            .set_terminator(Terminator::JmpIf(eq, then_b, else_b));
        hf.get_block_mut(then_b)
            .set_terminator(Terminator::Jmp(after, vec![]));
        hf.get_block_mut(else_b).push_instruction(Located::with(
            OpCode::WriteWitness {
                result: Some(r),
                value: w,
                pinned: false,
            },
            SourceLocation::test(),
        ));
        hf.get_block_mut(else_b)
            .set_terminator(Terminator::Jmp(after, vec![]));
        hf.get_block_mut(after)
            .set_terminator(Terminator::Return(vec![]));
    }
    {
        let mf = ssa.get_function_mut(main_id);
        let entry = mf.get_entry_mut();
        entry.push_parameter(w0, Type::u(32));
        entry.push_instruction(Located::with(
            OpCode::Call {
                results: vec![],
                function: CallTarget::Static(f),
                args: vec![c5, w0],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
        entry.set_terminator(Terminator::Return(vec![]));
    }

    let cc = run_in_test(&ssa);
    let ctx = sole_context(&cc, f);

    // Intraprocedurally the sym graft proves `x ≡ y`, the branch folds, the else arm dies: no
    // forward.
    assert!(cc.known_equal(f, x, y));
    assert!(cc.witness_forward(f, r).is_empty());
    // Per context `y` is `Const(15)` while `x` stays opaque: the congruence splits, the else arm
    // is live, and the forward exists only in the per-context view.
    assert!(!cc.known_equal_in(f, &ctx, x, y));
    assert_eq!(cc.witness_forward_in(f, &ctx, r), [w].as_slice());
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
        entry.push_instruction(Located::with(
            OpCode::Call {
                results: vec![r5],
                function: CallTarget::Static(id),
                args: vec![c5],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
        entry.push_instruction(Located::with(
            OpCode::Call {
                results: vec![r7],
                function: CallTarget::Static(id),
                args: vec![c7],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
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
        entry.push_instruction(Located::with(
            OpCode::Call {
                results: vec![r1],
                function: CallTarget::Static(helper),
                args: vec![],
                unconstrained: true,
            },
            SourceLocation::test(),
        ));
        entry.push_instruction(Located::with(
            OpCode::Call {
                results: vec![r2],
                function: CallTarget::Static(helper),
                args: vec![],
                unconstrained: true,
            },
            SourceLocation::test(),
        ));
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
        hf.get_entry_mut().push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: psum,
                lhs: p,
                rhs: p,
            },
            SourceLocation::test(),
        ));
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
            OpCode::Call {
                results: vec![r1],
                function: CallTarget::Static(dbl),
                args: vec![a],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
        entry.push_instruction(Located::with(
            OpCode::Call {
                results: vec![r2],
                function: CallTarget::Static(dbl),
                args: vec![b],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
        entry.push_instruction(Located::with(
            OpCode::Call {
                results: vec![r3],
                function: CallTarget::Static(dbl),
                args: vec![y],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
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
        hf.get_entry_mut().push_instruction(Located::with(
            OpCode::FreshWitness {
                result: w,
                result_type: Type::field(),
            },
            SourceLocation::test(),
        ));
        hf.get_entry_mut()
            .set_terminator(Terminator::Return(vec![w]));
    }
    let (r1, r2) = (ssa.fresh_value(), ssa.fresh_value());
    {
        let mf = ssa.get_function_mut(main_id);
        mf.add_return_type(Type::field());
        mf.add_return_type(Type::field());
        let entry = mf.get_entry_mut();
        entry.push_instruction(Located::with(
            OpCode::Call {
                results: vec![r1],
                function: CallTarget::Static(rnd),
                args: vec![],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
        entry.push_instruction(Located::with(
            OpCode::Call {
                results: vec![r2],
                function: CallTarget::Static(rnd),
                args: vec![],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
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
        hf.get_entry_mut().push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: isum,
                lhs: ip,
                rhs: ip,
            },
            SourceLocation::test(),
        ));
        hf.get_entry_mut()
            .set_terminator(Terminator::Return(vec![isum]));
    }
    let op = ssa.fresh_value();
    let ores = ssa.fresh_value();
    {
        let hf = ssa.get_function_mut(outer);
        hf.add_return_type(Type::field());
        hf.get_entry_mut().push_parameter(op, Type::field());
        hf.get_entry_mut().push_instruction(Located::with(
            OpCode::Call {
                results: vec![ores],
                function: CallTarget::Static(inner),
                args: vec![op],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
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
            OpCode::Call {
                results: vec![r1],
                function: CallTarget::Static(outer),
                args: vec![a],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
        entry.push_instruction(Located::with(
            OpCode::Call {
                results: vec![r2],
                function: CallTarget::Static(outer),
                args: vec![b],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
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
        hf.get_entry_mut().push_instruction(Located::with(
            OpCode::ArrayGet {
                result: elem,
                array: p,
                index: c0,
            },
            SourceLocation::test(),
        ));
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
        entry.push_instruction(Located::with(
            OpCode::MkSeq {
                result: a,
                elems: vec![x],
                seq_type: SequenceTargetType::Array(1),
                elem_type: Type::field(),
            },
            SourceLocation::test(),
        ));
        entry.push_instruction(Located::with(
            OpCode::MkSeq {
                result: b,
                elems: vec![x],
                seq_type: SequenceTargetType::Array(1),
                elem_type: Type::field(),
            },
            SourceLocation::test(),
        ));
        entry.push_instruction(Located::with(
            OpCode::MkSeq {
                result: d,
                elems: vec![y],
                seq_type: SequenceTargetType::Array(1),
                elem_type: Type::field(),
            },
            SourceLocation::test(),
        ));
        entry.push_instruction(Located::with(
            OpCode::Call {
                results: vec![r1],
                function: CallTarget::Static(pick),
                args: vec![a],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
        entry.push_instruction(Located::with(
            OpCode::Call {
                results: vec![r2],
                function: CallTarget::Static(pick),
                args: vec![b],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
        entry.push_instruction(Located::with(
            OpCode::Call {
                results: vec![r3],
                function: CallTarget::Static(pick),
                args: vec![d],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
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
    entry.push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Eq,
            result: eq,
            lhs: x,
            rhs: y,
        },
        SourceLocation::test(),
    ));
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
    entry.push_instruction(Located::with(
        OpCode::FreshWitness {
            result: w1,
            result_type: Type::field(),
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::FreshWitness {
            result: w2,
            result_type: Type::field(),
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Eq,
            result: eq,
            lhs: w1,
            rhs: w2,
        },
        SourceLocation::test(),
    ));
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
    entry.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: a,
            lhs: x,
            rhs: c1,
        },
        SourceLocation::test(),
    ));
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
            kind: CmpKind::Lt,
            result: lt,
            lhs: a,
            rhs: b,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![lt]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);

    assert!(cc.known_equal(fid, a, b));
    assert_eq!(cc.const_of(fid, lt).as_deref(), Some(&Constant::U(1, 0)));
}

/// A comparison of congruent operands folds to a constant `true` even when it is *witnessed*
/// (its result is `WitnessOf`-typed, as emitted by witness spilling / AD lowering) — the
/// must-equal fact holds either way. Keeping the substituted constant witness-typed is the
/// consumer's job; see the SCS test `witnessed_constant_is_cast_to_witness_of`.
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
    entry.push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Eq,
            result: ww,
            lhs: w,
            rhs: w,
        },
        SourceLocation::test(),
    ));
    // Plain comparison: result is `u1`.
    entry.push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Eq,
            result: nn,
            lhs: n,
            rhs: n,
        },
        SourceLocation::test(),
    ));
    entry.set_terminator(Terminator::Return(vec![ww, nn]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);

    // Both compare congruent (identical) operands, so both fold to `true` — the witnessed one
    // too. (SCS then keeps `ww` witness-typed via a cast; the analysis fact is the same.)
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
    header_block.push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Eq,
            result: eq,
            lhs: i,
            rhs: j,
        },
        SourceLocation::test(),
    ));
    header_block.push_instruction(Located::with(
        OpCode::Cmp {
            kind: CmpKind::Lt,
            result: lt,
            lhs: i,
            rhs: c10,
        },
        SourceLocation::test(),
    ));
    header_block.set_terminator(Terminator::JmpIf(lt, body, exit));
    let body_block = f.get_block_mut(body);
    body_block.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: i2,
            lhs: i,
            rhs: c1,
        },
        SourceLocation::test(),
    ));
    body_block.push_instruction(Located::with(
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: j2,
            lhs: j,
            rhs: c1,
        },
        SourceLocation::test(),
    ));
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
    entry.push_instruction(Located::with(
        OpCode::MkSeq {
            result: seq,
            elems: vec![c0, c1, c2],
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
    entry.push_instruction(Located::with(
        OpCode::MkRepeated {
            result: seq,
            element: elem,
            seq_type: SequenceTargetType::Array(4),
            count: 4,
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
    entry.push_instruction(Located::with(
        OpCode::MkSeq {
            result: seq,
            elems: vec![c0, c1, c2],
            seq_type: SequenceTargetType::Array(3),
            elem_type: Type::u(32),
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::ArraySet {
            result: seq2,
            array: seq,
            index: idx1,
            value: new_val,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::ArrayGet {
            result: at_set,
            array: seq2,
            index: idx1,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::ArrayGet {
            result: at_orig,
            array: seq2,
            index: idx0,
        },
        SourceLocation::test(),
    ));
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
    entry.push_instruction(Located::with(
        OpCode::MkSeq {
            result: base,
            elems: vec![mid],
            seq_type: SequenceTargetType::Slice,
            elem_type: Type::u(32),
        },
        SourceLocation::test(),
    ));
    // Front: [front, mid]; element 0 is the pushed value.
    entry.push_instruction(Located::with(
        OpCode::SlicePush {
            dir: SliceOpDir::Front,
            result: pushed_front,
            slice: base,
            values: vec![front],
        },
        SourceLocation::test(),
    ));
    // Back: [mid, back]; element 1 is the pushed value.
    entry.push_instruction(Located::with(
        OpCode::SlicePush {
            dir: SliceOpDir::Back,
            result: pushed_back,
            slice: base,
            values: vec![back],
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::ArrayGet {
            result: head,
            array: pushed_front,
            index: idx0,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::ArrayGet {
            result: tail,
            array: pushed_back,
            index: idx1,
        },
        SourceLocation::test(),
    ));
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
    entry.push_instruction(Located::with(
        OpCode::MkSeqOfBlob {
            result: seq,
            element_type: Type::u(32),
            blob,
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
    entry.push_instruction(Located::with(
        OpCode::MkSeq {
            result: seq,
            elems: vec![c0, c1],
            seq_type: SequenceTargetType::Array(2),
            elem_type: Type::u(32),
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::ArrayGet {
            result: g_oob,
            array: seq,
            index: idx_oob,
        },
        SourceLocation::test(),
    ));
    // Non-constant element keeps the whole aggregate non-constant.
    entry.push_instruction(Located::with(
        OpCode::MkSeq {
            result: seq_nc,
            elems: vec![c0, p],
            seq_type: SequenceTargetType::Array(2),
            elem_type: Type::u(32),
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::ArrayGet {
            result: g_nc,
            array: seq_nc,
            index: idx0,
        },
        SourceLocation::test(),
    ));
    // Over-cap repeat is refused before materialising the aggregate.
    entry.push_instruction(Located::with(
        OpCode::MkRepeated {
            result: big,
            element: c0,
            seq_type: SequenceTargetType::Array((1 << 16) + 1),
            count: (1 << 16) + 1,
            elem_type: Type::u(32),
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::ArrayGet {
            result: g_big,
            array: big,
            index: idx0,
        },
        SourceLocation::test(),
    ));
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
    entry.push_instruction(Located::with(
        OpCode::MkSeq {
            result: base,
            elems: vec![c0, c1],
            seq_type: SequenceTargetType::Array(2),
            elem_type: Type::u(32),
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::ArraySet {
            result: set_oob,
            array: base,
            index: idx_oob,
            value: c0,
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::ArrayGet {
            result: at_set_oob,
            array: set_oob,
            index: idx0,
        },
        SourceLocation::test(),
    ));

    // `SlicePush` whose result would exceed the cap (1 + CAP values): refused.
    entry.push_instruction(Located::with(
        OpCode::MkSeq {
            result: one,
            elems: vec![c0],
            seq_type: SequenceTargetType::Slice,
            elem_type: Type::u(32),
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::SlicePush {
            dir: SliceOpDir::Back,
            result: pushed,
            slice: one,
            values: vec![c0; 1 << 12],
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::ArrayGet {
            result: at_pushed,
            array: pushed,
            index: idx0,
        },
        SourceLocation::test(),
    ));

    // Over-cap `MkSeq`: refused before materialising the aggregate.
    entry.push_instruction(Located::with(
        OpCode::MkSeq {
            result: big_seq,
            elems: vec![c0; over_cap],
            seq_type: SequenceTargetType::Array(over_cap),
            elem_type: Type::u(32),
        },
        SourceLocation::test(),
    ));
    entry.push_instruction(Located::with(
        OpCode::ArrayGet {
            result: at_big,
            array: big_seq,
            index: idx0,
        },
        SourceLocation::test(),
    ));

    entry.set_terminator(Terminator::Return(vec![at_set_oob, at_pushed, at_big]));

    let fid = ssa.get_unique_entrypoint_id();
    let cc = run_in_test(&ssa);

    assert_eq!(cc.const_of(fid, at_set_oob), None);
    assert_eq!(cc.const_of(fid, at_pushed), None);
    assert_eq!(cc.const_of(fid, at_big), None);
}

// SYMBOLIC INTERPROCEDURAL CONGRUENCE
// ================================================================================================

/// A symbolic return jump ignores an argument the return does not use: `f(x, y) = x + 1` makes
/// `f(a, c)` and `f(a, d)` congruent even when `c ≢ d` (which the whole-argument `CallDet`
/// numbering cannot see), while a call differing in the *used* argument stays distinct.
#[test]
fn symbolic_jump_ignores_dead_argument() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let f = ssa.add_function("f".to_string());
    let c1 = ssa.add_const(Constant::Field(Field::from(1u64)));
    let (x, y, fa) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    // f(x, y) = x + 1  (y is dead).
    {
        let hf = ssa.get_function_mut(f);
        hf.add_return_type(Type::field());
        hf.get_entry_mut().push_parameter(x, Type::field());
        hf.get_entry_mut().push_parameter(y, Type::field());
        hf.get_entry_mut().push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: fa,
                lhs: x,
                rhs: c1,
            },
            SourceLocation::test(),
        ));
        hf.get_entry_mut()
            .set_terminator(Terminator::Return(vec![fa]));
    }

    let (a, b, c, d) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );
    let (r1, r2, r3) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    {
        let mf = ssa.get_function_mut(main_id);
        mf.add_return_type(Type::field());
        mf.add_return_type(Type::field());
        mf.add_return_type(Type::field());
        mf.get_entry_mut().push_parameter(a, Type::field());
        mf.get_entry_mut().push_parameter(b, Type::field());
        mf.get_entry_mut().push_parameter(c, Type::field());
        mf.get_entry_mut().push_parameter(d, Type::field());
        let entry = mf.get_entry_mut();
        for (res, args) in [(r1, vec![a, c]), (r2, vec![a, d]), (r3, vec![b, c])] {
            entry.push_instruction(Located::with(
                OpCode::Call {
                    results: vec![res],
                    function: CallTarget::Static(f),
                    args,
                    unconstrained: false,
                },
                SourceLocation::test(),
            ));
        }
        entry.set_terminator(Terminator::Return(vec![r1, r2, r3]));
    }

    let cc = run_in_test(&ssa);
    assert!(cc.known_equal(main_id, r1, r2)); // dead 2nd arg ⇒ congruent
    assert!(!cc.known_equal(main_id, r1, r3)); // distinct 1st arg ⇒ distinct
}

/// A symbolic return jump relates a call result to an *open expression* over the caller's values:
/// `f(x) = x + 5` makes `f(a)` congruent to a caller-local `a + 5` (which the opaque per-callee
/// `CallDet` node cannot), and not to `a + 6`. The call result still stays `Bottom` in the lattice,
/// so the SCS contract (the `eval_call` constant channel untouched) holds.
#[test]
fn symbolic_jump_relates_call_to_open_expression() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let f = ssa.add_function("f".to_string());
    let c5 = ssa.add_const(Constant::Field(Field::from(5u64)));
    let c6 = ssa.add_const(Constant::Field(Field::from(6u64)));
    let (x, fa) = (ssa.fresh_value(), ssa.fresh_value());

    // f(x) = x + 5
    {
        let hf = ssa.get_function_mut(f);
        hf.add_return_type(Type::field());
        hf.get_entry_mut().push_parameter(x, Type::field());
        hf.get_entry_mut().push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: fa,
                lhs: x,
                rhs: c5,
            },
            SourceLocation::test(),
        ));
        hf.get_entry_mut()
            .set_terminator(Terminator::Return(vec![fa]));
    }

    let a = ssa.fresh_value();
    let (r, w, w2) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    {
        let mf = ssa.get_function_mut(main_id);
        mf.add_return_type(Type::field());
        mf.get_entry_mut().push_parameter(a, Type::field());
        let entry = mf.get_entry_mut();
        entry.push_instruction(Located::with(
            OpCode::Call {
                results: vec![r],
                function: CallTarget::Static(f),
                args: vec![a],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
        entry.push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: w,
                lhs: a,
                rhs: c5,
            },
            SourceLocation::test(),
        ));
        entry.push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: w2,
                lhs: a,
                rhs: c6,
            },
            SourceLocation::test(),
        ));
        entry.set_terminator(Terminator::Return(vec![r]));
    }

    let cc = run_in_test(&ssa);
    assert!(cc.known_equal(main_id, r, w)); // r ≡ a + 5
    assert!(!cc.known_equal(main_id, r, w2)); // r ≢ a + 6
    // The call result is never folded into the constant lattice (Bottom contract).
    assert_eq!(cc.const_of(main_id, r), None);
    assert!(cc.new_const_values(main_id).iter().all(|(v, _)| *v != r));
}

/// A two-level return expression grafts through a synthetic intermediate node: `f(x) = (x + 5) * 2`
/// makes `f(a)` congruent to a caller-local `(a + 5) * 2`. The synthetic scaffolding is stripped,
/// so it never reaches a congruence class or `compute_leaders` (which would otherwise treat a
/// def-less id as an illegal leader).
#[test]
fn symbolic_jump_depth_two_grafts_via_synthetic() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let f = ssa.add_function("f".to_string());
    let c5 = ssa.add_const(Constant::Field(Field::from(5u64)));
    let c2 = ssa.add_const(Constant::Field(Field::from(2u64)));
    let (x, ft, fa) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    // f(x) = (x + 5) * 2
    {
        let hf = ssa.get_function_mut(f);
        hf.add_return_type(Type::field());
        hf.get_entry_mut().push_parameter(x, Type::field());
        hf.get_entry_mut().push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: ft,
                lhs: x,
                rhs: c5,
            },
            SourceLocation::test(),
        ));
        hf.get_entry_mut().push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Mul,
                result: fa,
                lhs: ft,
                rhs: c2,
            },
            SourceLocation::test(),
        ));
        hf.get_entry_mut()
            .set_terminator(Terminator::Return(vec![fa]));
    }

    let a = ssa.fresh_value();
    let (r, wt, w) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    {
        let mf = ssa.get_function_mut(main_id);
        mf.add_return_type(Type::field());
        mf.get_entry_mut().push_parameter(a, Type::field());
        let entry = mf.get_entry_mut();
        entry.push_instruction(Located::with(
            OpCode::Call {
                results: vec![r],
                function: CallTarget::Static(f),
                args: vec![a],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
        entry.push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: wt,
                lhs: a,
                rhs: c5,
            },
            SourceLocation::test(),
        ));
        entry.push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Mul,
                result: w,
                lhs: wt,
                rhs: c2,
            },
            SourceLocation::test(),
        ));
        entry.set_terminator(Terminator::Return(vec![r]));
    }

    let cc = run_in_test(&ssa);
    assert!(cc.known_equal(main_id, r, w)); // r ≡ (a + 5) * 2

    // `r`'s class is exactly `{r, w}` — no synthetic id leaked in.
    let mut r_class = cc.congruence_class(main_id, r);
    r_class.sort();
    let mut expected = vec![r, w];
    expected.sort();
    assert_eq!(r_class, expected);

    // The grafted inner `Add(a, 5)` is congruent to the real `wt`, but the synthetic that carried
    // it was stripped, so `wt`'s class is `{wt}` alone and its leader is a real, def-dominating
    // value.
    assert_eq!(cc.congruence_class(main_id, wt), vec![wt]);
    assert_eq!(cc.leader(main_id, wt), Some(wt));
    let leader = cc.leader(main_id, r).expect("r is in a class");
    assert!(leader == r || leader == w);
}

/// The dead-argument win survives a two-level return: `f(x, y) = (x + 5) * 2` (y dead) makes
/// `f(a, c) ≡ f(a, d)` — the case a depth-1-only graft would lose by falling back to
/// `CallDet[x, y]`.
#[test]
fn symbolic_jump_dead_argument_behind_depth_two() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let f = ssa.add_function("f".to_string());
    let c5 = ssa.add_const(Constant::Field(Field::from(5u64)));
    let c2 = ssa.add_const(Constant::Field(Field::from(2u64)));
    let (x, y, ft, fa) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    // f(x, y) = (x + 5) * 2  (y is dead).
    {
        let hf = ssa.get_function_mut(f);
        hf.add_return_type(Type::field());
        hf.get_entry_mut().push_parameter(x, Type::field());
        hf.get_entry_mut().push_parameter(y, Type::field());
        hf.get_entry_mut().push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: ft,
                lhs: x,
                rhs: c5,
            },
            SourceLocation::test(),
        ));
        hf.get_entry_mut().push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Mul,
                result: fa,
                lhs: ft,
                rhs: c2,
            },
            SourceLocation::test(),
        ));
        hf.get_entry_mut()
            .set_terminator(Terminator::Return(vec![fa]));
    }

    let (a, b, c, d) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );
    let (r1, r2, r3) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    {
        let mf = ssa.get_function_mut(main_id);
        mf.add_return_type(Type::field());
        mf.add_return_type(Type::field());
        mf.add_return_type(Type::field());
        mf.get_entry_mut().push_parameter(a, Type::field());
        mf.get_entry_mut().push_parameter(b, Type::field());
        mf.get_entry_mut().push_parameter(c, Type::field());
        mf.get_entry_mut().push_parameter(d, Type::field());
        let entry = mf.get_entry_mut();
        for (res, args) in [(r1, vec![a, c]), (r2, vec![a, d]), (r3, vec![b, c])] {
            entry.push_instruction(Located::with(
                OpCode::Call {
                    results: vec![res],
                    function: CallTarget::Static(f),
                    args,
                    unconstrained: false,
                },
                SourceLocation::test(),
            ));
        }
        entry.set_terminator(Terminator::Return(vec![r1, r2, r3]));
    }

    let cc = run_in_test(&ssa);
    assert!(cc.known_equal(main_id, r1, r2)); // dead 2nd arg, two levels deep ⇒ still congruent
    assert!(!cc.known_equal(main_id, r1, r3)); // distinct 1st arg ⇒ distinct
}

/// A return deeper than the depth cap is not symbolically expressed (`Sym = None`), so the call
/// falls back to `CallDet`: two identical calls stay congruent and a distinct argument does not,
/// but the result is *not* related to the caller-local full expression.
#[test]
fn symbolic_jump_depth_over_cap_falls_back_to_calldet() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let f = ssa.add_function("f".to_string());
    let c1 = ssa.add_const(Constant::Field(Field::from(1u64)));
    let c2 = ssa.add_const(Constant::Field(Field::from(2u64)));
    let c3 = ssa.add_const(Constant::Field(Field::from(3u64)));
    let (x, ft, fu, fa) = (
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
        ssa.fresh_value(),
    );

    // f(x) = ((x + 1) * 2) + 3  (depth 3, over the cap).
    {
        let hf = ssa.get_function_mut(f);
        hf.add_return_type(Type::field());
        hf.get_entry_mut().push_parameter(x, Type::field());
        hf.get_entry_mut().push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: ft,
                lhs: x,
                rhs: c1,
            },
            SourceLocation::test(),
        ));
        hf.get_entry_mut().push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Mul,
                result: fu,
                lhs: ft,
                rhs: c2,
            },
            SourceLocation::test(),
        ));
        hf.get_entry_mut().push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: fa,
                lhs: fu,
                rhs: c3,
            },
            SourceLocation::test(),
        ));
        hf.get_entry_mut()
            .set_terminator(Terminator::Return(vec![fa]));
    }

    let (a, b) = (ssa.fresh_value(), ssa.fresh_value());
    let (r1, r2, r3) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let (wt, wu, w) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    {
        let mf = ssa.get_function_mut(main_id);
        mf.add_return_type(Type::field());
        mf.add_return_type(Type::field());
        mf.add_return_type(Type::field());
        mf.get_entry_mut().push_parameter(a, Type::field());
        mf.get_entry_mut().push_parameter(b, Type::field());
        let entry = mf.get_entry_mut();
        for (res, arg) in [(r1, a), (r2, a), (r3, b)] {
            entry.push_instruction(Located::with(
                OpCode::Call {
                    results: vec![res],
                    function: CallTarget::Static(f),
                    args: vec![arg],
                    unconstrained: false,
                },
                SourceLocation::test(),
            ));
        }
        // The caller-local `((a + 1) * 2) + 3`.
        entry.push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: wt,
                lhs: a,
                rhs: c1,
            },
            SourceLocation::test(),
        ));
        entry.push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Mul,
                result: wu,
                lhs: wt,
                rhs: c2,
            },
            SourceLocation::test(),
        ));
        entry.push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: w,
                lhs: wu,
                rhs: c3,
            },
            SourceLocation::test(),
        ));
        entry.set_terminator(Terminator::Return(vec![r1, r2, r3]));
    }

    let cc = run_in_test(&ssa);
    assert!(cc.known_equal(main_id, r1, r2)); // CallDet still fires for identical args
    assert!(!cc.known_equal(main_id, r1, r3)); // distinct arg ⇒ distinct
    assert!(!cc.known_equal(main_id, r1, w)); // no symbolic graft past the depth cap
}

/// A return that flows through a nested call is opaque at a single interprocedural level
/// (`Sym = None`), falling back to `CallDet`: `g(x) = h(x) + 1` numbers identical calls congruent
/// and a distinct argument distinct, but does not graft the open expression.
#[test]
fn symbolic_jump_nested_call_falls_back_to_calldet() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let h = ssa.add_function("h".to_string());
    let g = ssa.add_function("g".to_string());
    let c1 = ssa.add_const(Constant::Field(Field::from(1u64)));

    let (hx, hr) = (ssa.fresh_value(), ssa.fresh_value());
    // h(x) = x + 1
    {
        let hf = ssa.get_function_mut(h);
        hf.add_return_type(Type::field());
        hf.get_entry_mut().push_parameter(hx, Type::field());
        hf.get_entry_mut().push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: hr,
                lhs: hx,
                rhs: c1,
            },
            SourceLocation::test(),
        ));
        hf.get_entry_mut()
            .set_terminator(Terminator::Return(vec![hr]));
    }

    let (gx, gc, ga) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    // g(x) = h(x) + 1
    {
        let gf = ssa.get_function_mut(g);
        gf.add_return_type(Type::field());
        gf.get_entry_mut().push_parameter(gx, Type::field());
        gf.get_entry_mut().push_instruction(Located::with(
            OpCode::Call {
                results: vec![gc],
                function: CallTarget::Static(h),
                args: vec![gx],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
        gf.get_entry_mut().push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: ga,
                lhs: gc,
                rhs: c1,
            },
            SourceLocation::test(),
        ));
        gf.get_entry_mut()
            .set_terminator(Terminator::Return(vec![ga]));
    }

    let (a, b) = (ssa.fresh_value(), ssa.fresh_value());
    let (r1, r2, r3) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    {
        let mf = ssa.get_function_mut(main_id);
        mf.add_return_type(Type::field());
        mf.add_return_type(Type::field());
        mf.add_return_type(Type::field());
        mf.get_entry_mut().push_parameter(a, Type::field());
        mf.get_entry_mut().push_parameter(b, Type::field());
        let entry = mf.get_entry_mut();
        for (res, arg) in [(r1, a), (r2, a), (r3, b)] {
            entry.push_instruction(Located::with(
                OpCode::Call {
                    results: vec![res],
                    function: CallTarget::Static(g),
                    args: vec![arg],
                    unconstrained: false,
                },
                SourceLocation::test(),
            ));
        }
        entry.set_terminator(Terminator::Return(vec![r1, r2, r3]));
    }

    let cc = run_in_test(&ssa);
    assert!(cc.known_equal(main_id, r1, r2)); // CallDet fallback: identical args congruent
    assert!(!cc.known_equal(main_id, r1, r3)); // distinct arg ⇒ distinct
}

/// Commutativity is carried verbatim from `op_signature`: a non-commutative `f(x, y) = x - y` does
/// not equate `f(a, b)` with `f(b, a)` (which an unsound commute would), while identical calls
/// match.
#[test]
fn symbolic_jump_preserves_noncommutativity() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let f = ssa.add_function("f".to_string());
    let (x, y, fa) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    // f(x, y) = x - y
    {
        let hf = ssa.get_function_mut(f);
        hf.add_return_type(Type::field());
        hf.get_entry_mut().push_parameter(x, Type::field());
        hf.get_entry_mut().push_parameter(y, Type::field());
        hf.get_entry_mut().push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Sub,
                result: fa,
                lhs: x,
                rhs: y,
            },
            SourceLocation::test(),
        ));
        hf.get_entry_mut()
            .set_terminator(Terminator::Return(vec![fa]));
    }

    let (a, b) = (ssa.fresh_value(), ssa.fresh_value());
    let (r1, r2, r3) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    {
        let mf = ssa.get_function_mut(main_id);
        mf.add_return_type(Type::field());
        mf.add_return_type(Type::field());
        mf.add_return_type(Type::field());
        mf.get_entry_mut().push_parameter(a, Type::field());
        mf.get_entry_mut().push_parameter(b, Type::field());
        let entry = mf.get_entry_mut();
        for (res, args) in [(r1, vec![a, b]), (r2, vec![b, a]), (r3, vec![a, b])] {
            entry.push_instruction(Located::with(
                OpCode::Call {
                    results: vec![res],
                    function: CallTarget::Static(f),
                    args,
                    unconstrained: false,
                },
                SourceLocation::test(),
            ));
        }
        entry.set_terminator(Terminator::Return(vec![r1, r2, r3]));
    }

    let cc = run_in_test(&ssa);
    assert!(!cc.known_equal(main_id, r1, r2)); // x - y ≢ y - x
    assert!(cc.known_equal(main_id, r1, r3)); // same operand order ⇒ congruent
}

/// A return mixing a formal with a fresh witness is not a pure function of the formals: `build_sym`
/// bails on the witness operand (`Sym = None`) and, being non-deterministic, the return is not
/// `CallDet`-numbered either, so two such calls stay distinct.
#[test]
fn symbolic_jump_witness_operand_is_not_grafted() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let f = ssa.add_function("f".to_string());
    let (x, wit, fa) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

    // f(x) = x + witness
    {
        let hf = ssa.get_function_mut(f);
        hf.add_return_type(Type::field());
        hf.get_entry_mut().push_parameter(x, Type::field());
        hf.get_entry_mut().push_instruction(Located::with(
            OpCode::FreshWitness {
                result: wit,
                result_type: Type::field(),
            },
            SourceLocation::test(),
        ));
        hf.get_entry_mut().push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: fa,
                lhs: x,
                rhs: wit,
            },
            SourceLocation::test(),
        ));
        hf.get_entry_mut()
            .set_terminator(Terminator::Return(vec![fa]));
    }

    let a = ssa.fresh_value();
    let (r1, r2) = (ssa.fresh_value(), ssa.fresh_value());
    {
        let mf = ssa.get_function_mut(main_id);
        mf.add_return_type(Type::field());
        mf.add_return_type(Type::field());
        mf.get_entry_mut().push_parameter(a, Type::field());
        let entry = mf.get_entry_mut();
        for res in [r1, r2] {
            entry.push_instruction(Located::with(
                OpCode::Call {
                    results: vec![res],
                    function: CallTarget::Static(f),
                    args: vec![a],
                    unconstrained: false,
                },
                SourceLocation::test(),
            ));
        }
        entry.set_terminator(Terminator::Return(vec![r1, r2]));
    }

    let cc = run_in_test(&ssa);
    assert!(!cc.known_equal(main_id, r1, r2)); // witness in return ⇒ neither grafted nor CallDet
}

/// A grafted return that names a *callee-only* constant must not produce a false congruence by
/// minting its synthetic scaffolding onto a real value's id.
#[test]
fn symbolic_jump_synthetic_does_not_collide_with_callee_only_constant() {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let f = ssa.add_function("f".to_string());

    // Caller-side constants used to *synthesize* Field(5) without naming the callee-only constant,
    // so the caller has a value in `C`'s constant class without referencing `C`'s id.
    let two = ssa.add_const(Constant::Field(Field::from(2u64)));
    let three = ssa.add_const(Constant::Field(Field::from(3u64)));

    // `f`'s formal, then every caller value — all minted before the callee-only constant so its id
    // is exactly one above the caller's local maximum (the single-synthetic window).
    let x = ssa.fresh_value();
    let a = ssa.fresh_value();
    let v = ssa.fresh_value();
    let w = ssa.fresh_value();
    let r = ssa.fresh_value();

    // The callee-only constant, interned last. Field(5) so the caller's `2 + 3` folds into its
    // class.
    let c = ssa.add_const(Constant::Field(Field::from(5u64)));

    // `f`'s internal op results.
    let (ft, fa) = (ssa.fresh_value(), ssa.fresh_value());

    // Precondition: `c` sits exactly where a local-max-upward synthetic counter would mint its
    // first synthetic. The downward allocator avoids it, but this positioning keeps the test a live
    // regression guard against reverting to the buggy scheme. If id allocation ever shifts, it fails
    // loudly rather than silently no longer exercising the collision.
    assert_eq!(c.0, r.0 + 1);

    // f(x) = x + (x * C)
    {
        let hf = ssa.get_function_mut(f);
        hf.add_return_type(Type::field());
        hf.get_entry_mut().push_parameter(x, Type::field());
        hf.get_entry_mut().push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Mul,
                result: ft,
                lhs: x,
                rhs: c,
            },
            SourceLocation::test(),
        ));
        hf.get_entry_mut().push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: fa,
                lhs: x,
                rhs: ft,
            },
            SourceLocation::test(),
        ));
        hf.get_entry_mut()
            .set_terminator(Terminator::Return(vec![fa]));
    }

    {
        let mf = ssa.get_function_mut(main_id);
        mf.add_return_type(Type::field());
        mf.add_return_type(Type::field());
        mf.get_entry_mut().push_parameter(a, Type::field());
        let entry = mf.get_entry_mut();
        // v = 2 + 3, which folds to Field(5) — the same constant class as `c`.
        entry.push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: v,
                lhs: two,
                rhs: three,
            },
            SourceLocation::test(),
        ));
        // r = f(a)
        entry.push_instruction(Located::with(
            OpCode::Call {
                results: vec![r],
                function: CallTarget::Static(f),
                args: vec![a],
                unconstrained: false,
            },
            SourceLocation::test(),
        ));
        // w = a + v, a genuine `a + 5`.
        entry.push_instruction(Located::with(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: w,
                lhs: a,
                rhs: v,
            },
            SourceLocation::test(),
        ));
        entry.set_terminator(Terminator::Return(vec![r, w]));
    }

    let cc = run_in_test(&ssa);
    // f(a) = a + a*5, never a + 5, so `r` must not be congruent to `w`.
    assert!(!cc.known_equal(main_id, r, w));
}
