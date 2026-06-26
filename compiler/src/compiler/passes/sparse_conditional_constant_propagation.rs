//! Sparse conditional constant propagation (Wegman–Zadeck) over the HLSSA.
//!
//! This pass relies on the combined [`ClickCooper`] analysis for the decisions that it makes. The
//! analysis currently exposes some additional functionality to ensure that SCCP can remain
//! byte-for-byte identical to the previous implementation. This pass **will be removed** in
//! subsequent work, and those additional pieces of functionality will be removed with them.
//!
//! # Folding Semantics
//!
//! Folding is driven by Click-Cooper's `new_const_values`, which reports every value proven
//! unconditionally constant — including congruence-derived must-equal facts, i.e. a comparison of
//! congruent operands (`x == x → true`, `x < x → false`) even when neither operand is itself
//! constant. A pure scalar fold whose result is constant is dropped and its uses aliased to the
//! bare interned constant.
//!
//! The same applies to a pure sequence *projection* (`ArrayGet` / `SliceLen`) that Click-Cooper
//! folded over a constant aggregate at a constant in-bounds index: its scalar result is the only
//! thing surfaced, so it is dropped and aliased identically.
//!
//! The **exception** is a `WitnessOf`-typed folded constant (a witnessed comparison of congruent
//! operands): aliasing it to a bare constant would drop the wrapper and mistype the IR, so it is
//! instead redefined in place as `cast <const> to WitnessOf`, keeping the value witness-typed.

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        analysis::{
            click_cooper::ClickCooper,
            flow_analysis::FlowAnalysis,
            types::{TypeInfo, Types},
        },
        pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
        passes::fix_double_jumps::{ReplaceScope, ValueReplacements},
        ssa::{
            BlockId, FunctionId, Instruction, Terminator, ValueId,
            hlssa::{CastTarget, HLFunction, HLSSA, OpCode},
        },
    },
};

// SCCP
// ================================================================================================

/// A standard implementation of (Wegman-Zadeck) sparse conditional constant propagation.
pub struct SCCP {}

impl Pass for SCCP {
    fn name(&self) -> &'static str {
        "sccp"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![ClickCooper::id(), TypeInfo::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        rewrite_all(ssa, store.get::<ClickCooper>(), store.get::<TypeInfo>());
    }
}

impl SCCP {
    pub fn new() -> Self {
        Self {}
    }

    /// Standalone entry (tests / callers without an `AnalysisStore`): recomputes `TypeInfo` from
    /// the current SSA, then rewrites. The pass-manager `run` path uses the cached `TypeInfo`
    /// instead.
    pub fn do_run(&self, ssa: &mut HLSSA, cc: &ClickCooper) {
        let flow = FlowAnalysis::run(ssa);
        let type_info = Types::new().run(ssa, &flow);
        rewrite_all(ssa, cc, &type_info);
    }
}

fn rewrite_all(ssa: &mut HLSSA, cc: &ClickCooper, type_info: &TypeInfo) {
    let fids: Vec<_> = ssa.get_function_ids().collect();
    for fid in fids {
        let mut function = ssa.take_function(fid);
        rewrite(&mut function, ssa, cc, type_info, fid);
        ssa.put_function(fid, function);
    }
}

// REWRITING FUNCTIONALITY
// ================================================================================================

fn rewrite(
    function: &mut HLFunction,
    ssa: &HLSSA,
    cc: &ClickCooper,
    type_info: &TypeInfo,
    fid: FunctionId,
) {
    let fn_type_info = type_info.get_function(fid);
    // Drop blocks the analysis never reached. Every kept terminator only targets reachable blocks:
    // a JmpIf with a constant condition is rewritten to a Jmp to its (reachable) live successor
    // below, and all other terminators had all successor edges marked executable.
    let all_blocks: Vec<BlockId> = function.get_blocks().map(|(id, _)| *id).collect();
    for bid in &all_blocks {
        if !cc.is_reachable(fid, *bid) {
            let _ = function.take_block(*bid);
        }
    }

    // Alias every constant-valued value (instruction results and block parameters) to the interned
    // constant. `const_values` comes from a deterministic `FxHashMap` iteration (the hasher has no
    // per-run seed), so interning a missing constant (which allocates a fresh value id) is
    // deterministic across runs.
    //
    // A `WitnessOf`-typed constant is the exception: it cannot be aliased to a bare interned
    // constant (that would drop the wrapper and mistype the IR — e.g. a witnessed comparison of
    // congruent operands folds to a provably-true witness). Those are collected here and redefined
    // in place as a `cast <const> to WitnessOf` in the instruction loop below, preserving the
    // witness type.
    let mut replacements = ValueReplacements::new();
    let const_values = cc.new_const_values(fid);
    let const_set: HashSet<ValueId> = const_values.iter().map(|(v, _)| *v).collect();

    // In practice `witness_consts` only ever holds `Cmp` instruction *results* — the sole witnessed
    // constants the analysis produces (a vacuous witnessed comparison).
    let mut witness_consts = HashMap::default();
    for (v, c) in &const_values {
        if fn_type_info.get_value_type(*v).is_witness_of() {
            witness_consts.insert(*v, c.clone());
        } else {
            replacements.insert(*v, ssa.add_const((**c).clone()));
        }
    }

    let kept_blocks: Vec<BlockId> = function.get_blocks().map(|(id, _)| *id).collect();
    for bid in kept_blocks {
        let local_replacements = bool_fact_replacements(ssa, cc, fid, bid);
        let block = function.get_block_mut(bid);

        let instructions = block.take_instructions();
        let mut kept = Vec::with_capacity(instructions.len());
        for instr in instructions {
            // Purity gate: a single-result instruction whose result the analysis proved constant is
            // dropped only when it is a pure scalar fold; its uses are aliased above either way.
            //
            // A `WitnessOf`-typed constant keeps its value id, redefined as a cast of the interned
            // constant to `WitnessOf`, so the value's witness type is preserved for its (possibly
            // witnessed) uses and return slots, rather than being aliased to a bare constant.
            {
                let mut results = instr.get_results();
                if let (Some(r), None) = (results.next(), results.next()) {
                    let is_const = const_set.contains(r);

                    // A surfaced scalar constant is produced either by a pure scalar fold or by a
                    // pure sequence *projection* — an `ArrayGet`/`SliceLen` that ClickCooper folded
                    // over a constant aggregate at a proven in-bounds constant index. Both are pure
                    // single-result reads, so the definition is dropped and its uses are aliased to
                    // the interned constant; any other constant-producing op is an analysis bug.
                    let foldable = instr.is_pure_scalar_fold()
                        || matches!(instr, OpCode::ArrayGet { .. } | OpCode::SliceLen { .. });
                    assert!(
                        !is_const || foldable,
                        "ICE: Result {r:?} of non-foldable instruction {instr:?} is in the \
                         constant set; ClickCooper must only fold pure scalar ops and pure sequence \
                         projections to a constant"
                    );

                    if is_const && foldable {
                        if let Some(c) = witness_consts.get(r) {
                            let bare = ssa.add_const((**c).clone());
                            kept.push(OpCode::Cast {
                                result: *r,
                                value: bare,
                                target: CastTarget::WitnessOf,
                            });
                        }
                        continue;
                    }
                }
            }

            // A select with a constant condition aliases to the chosen arm even when the arms are
            // not constants.
            if let OpCode::Select {
                result,
                cond,
                if_t,
                if_f,
            } = &instr
            {
                if let Some(b) = cc.const_bool_in_block(fid, bid, *cond) {
                    replacements.insert(*result, if b { *if_t } else { *if_f });
                    continue;
                }
            }
            let mut instr = instr;
            local_replacements.replace_inputs(&mut instr);
            kept.push(instr);
        }

        block.put_instructions(kept);

        if let Some(Terminator::JmpIf(cond, t, f)) = block.get_terminator() {
            let (cond, t, f) = (*cond, *t, *f);
            if let Some(b) = cc.const_bool_in_block(fid, bid, cond) {
                let target = if b { t } else { f };
                block.set_terminator(Terminator::Jmp(target, vec![]));
            }
        }
        local_replacements.replace_terminator(block.get_terminator_mut());
    }

    replacements.apply_to_function(function, ReplaceScope::Inputs);
}

/// Within `bid`, replace every value the analysis knows is a constant boolean (a branch predicate
/// fact) with that constant.
///
/// Local and structure-preserving as the establishing branch is kept.
fn bool_fact_replacements(
    ssa: &HLSSA,
    cc: &ClickCooper,
    fid: FunctionId,
    bid: BlockId,
) -> ValueReplacements {
    let mut replacements = ValueReplacements::new();
    for (value, c) in cc.block_bool_facts(fid, bid) {
        replacements.insert(value, ssa.add_const((*c).clone()));
    }
    replacements
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{
        Field,
        analysis::click_cooper::tests::run_in_test,
        ssa::hlssa::{BinaryArithOpKind, CastTarget, CmpKind, Constant, SequenceTargetType, Type},
    };

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
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: sum,
            lhs: c2,
            rhs: c3,
        });
        entry.push_instruction(OpCode::Cmp {
            kind: CmpKind::Eq,
            result: is_five,
            lhs: sum,
            rhs: c5,
        });
        entry.set_terminator(Terminator::JmpIf(is_five, then_b, else_b));
        f.get_block_mut(then_b)
            .set_terminator(Terminator::Return(vec![sum]));
        f.get_block_mut(else_b)
            .set_terminator(Terminator::Return(vec![c2]));

        let cc = run_in_test(&ssa);
        SCCP::new().do_run(&mut ssa, &cc);

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

        let cc = run_in_test(&ssa);
        SCCP::new().do_run(&mut ssa, &cc);

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
        header_block.push_instruction(OpCode::Cmp {
            kind: CmpKind::Lt,
            result: lt,
            lhs: i_param,
            rhs: c10,
        });
        header_block.set_terminator(Terminator::JmpIf(lt, body, exit));
        let body_block = f.get_block_mut(body);
        body_block.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: next,
            lhs: i_param,
            rhs: c1,
        });
        body_block.set_terminator(Terminator::Jmp(header, vec![next]));
        f.get_block_mut(exit)
            .set_terminator(Terminator::Return(vec![i_param]));

        let cc = run_in_test(&ssa);
        SCCP::new().do_run(&mut ssa, &cc);

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
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: sum,
            lhs: c200,
            rhs: c100,
        });
        entry.set_terminator(Terminator::Return(vec![sum]));

        let cc = run_in_test(&ssa);
        SCCP::new().do_run(&mut ssa, &cc);

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
        entry.push_instruction(OpCode::Cast {
            result: wit,
            value: c5,
            target: CastTarget::WitnessOf,
        });
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: doubled,
            lhs: wit,
            rhs: wit,
        });
        entry.set_terminator(Terminator::Return(vec![doubled]));

        let cc = run_in_test(&ssa);
        SCCP::new().do_run(&mut ssa, &cc);

        let f = ssa.get_unique_entrypoint();
        assert_eq!(f.get_entry().get_instructions().count(), 2);
    }

    /// A degenerate loop: the branch condition is a parameter of a block that is itself only
    /// reachable through that branch, so no executable jump ever supplies the parameter and the
    /// condition converges stuck at ⊤. Such a condition has a use its definition does not
    /// dominate — malformed SSA — and must ICE rather than let the rewrite delete blocks the kept
    /// `JmpIf` still targets.
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

        let cc = run_in_test(&ssa);
        SCCP::new().do_run(&mut ssa, &cc);
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
        entry.push_instruction(OpCode::Select {
            result: sel,
            cond: c_true,
            if_t: arm_t,
            if_f: arm_f,
        });
        entry.set_terminator(Terminator::Return(vec![sel]));

        let cc = run_in_test(&ssa);
        SCCP::new().do_run(&mut ssa, &cc);

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
        then_block.push_instruction(OpCode::Select {
            result: selected,
            cond,
            if_t: arm_t,
            if_f: arm_f,
        });
        then_block.push_instruction(OpCode::Not {
            result: not_cond,
            value: cond,
        });
        then_block.set_terminator(Terminator::Return(vec![selected, not_cond]));
        f.get_block_mut(else_b)
            .set_terminator(Terminator::Return(vec![arm_f, c_false]));

        let cc = run_in_test(&ssa);
        SCCP::new().do_run(&mut ssa, &cc);

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
        merge_block.push_instruction(OpCode::Not {
            result: not_cond,
            value: cond,
        });
        merge_block.set_terminator(Terminator::Return(vec![not_cond]));

        let cc = run_in_test(&ssa);
        SCCP::new().do_run(&mut ssa, &cc);

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

        let cc = run_in_test(&ssa);
        SCCP::new().do_run(&mut ssa, &cc);

        let f = ssa.get_unique_entrypoint();
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
        join_block.push_instruction(OpCode::Not {
            result: not_cond,
            value: cond,
        });
        join_block.set_terminator(Terminator::Return(vec![not_cond]));

        let cc = run_in_test(&ssa);
        SCCP::new().do_run(&mut ssa, &cc);

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

        let cc = run_in_test(&ssa);
        SCCP::new().do_run(&mut ssa, &cc);

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

    /// A branch decided by *congruence* — `CmpEq(a, b)` with `a` and `b` structurally equal but not
    /// constant — folds away through the combined-fixpoint writeback: the comparison is dropped,
    /// the `JmpIf` becomes a `Jmp` to the then-block, and the dead else-block is deleted. The two
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

        let cc = run_in_test(&ssa);
        SCCP::new().do_run(&mut ssa, &cc);

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

    /// A *witnessed* comparison of congruent operands folds to a constant, but — unlike a plain
    /// scalar fold, which is dropped and aliased to the bare interned constant — the value keeps
    /// its `WitnessOf` type: it is redefined in place as a `cast <const> to WitnessOf`, so the
    /// (witnessed) return slot stays correctly typed.
    #[test]
    fn witnessed_constant_is_cast_to_witness_of() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (w, ww) = (ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        let entry = f.get_entry_mut();
        // An operand is `WitnessOf`, so the comparison result is `WitnessOf(u1)`.
        entry.push_parameter(w, Type::witness_of(Type::u(32)));
        entry.push_instruction(OpCode::Cmp {
            kind: CmpKind::Eq,
            result: ww,
            lhs: w,
            rhs: w,
        });
        entry.set_terminator(Terminator::Return(vec![ww]));

        let cc = run_in_test(&ssa);
        SCCP::new().do_run(&mut ssa, &cc);

        let f = ssa.get_unique_entrypoint();
        // The witnessed comparison is gone, replaced by a cast of the folded constant to `WitnessOf`
        // (keeping `ww` witness-typed) rather than aliased to a bare constant.
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
        // The return still references `ww`, now witness-typed, matching its WitnessOf return slot.
        assert!(matches!(
            f.get_entry().get_terminator(),
            Some(Terminator::Return(vals)) if vals.as_slice() == [ww]
        ));
    }

    /// A constant lookup-table projection folds at the SCCP level: `ArrayGet`/`SliceLen` over a
    /// constant `MkSeq` are dropped and their uses aliased to the interned scalar constant, while
    /// the aggregate `MkSeq` itself (never surfaced) is left in place. Exercises the widened
    /// `foldable` gate — the purity assert must accept these pure sequence projections.
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
        entry.push_instruction(OpCode::MkSeq {
            result: seq,
            elems: vec![c10, c20, c30],
            seq_type: SequenceTargetType::Array(3),
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

        let cc = run_in_test(&ssa);
        SCCP::new().do_run(&mut ssa, &cc);

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
}
