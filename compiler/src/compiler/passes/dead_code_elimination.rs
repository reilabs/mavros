//! Performs dead-code elimination using a standard mark-and-sweep liveness algorithm with
//! cross-function propagation.
//!
//! It is capable of dropping dead blocks, dead instructions, and dead block parameters. It also
//! prunes arguments and results in call instructions to live parameters, and rewrites terminators.
//!
//! TODO Refactor to use the existing liveness analysis (#157).

use mavros_artifacts::FieldConfig;

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        analysis::{
            flow_analysis::{CFG, FlowAnalysis},
            types::{TypeInfo, Types},
            value_range_analysis::{ValueRangeAnalysis, ValueRanges},
        },
        pass_manager::{AnalysisId, AnalysisStore, Pass},
        passes::shared::divmod_guard::{
            divmod_can_fail, divmod_provably_defined, emit_divmod_is_defined_assert,
        },
        passes::shared::seq_bounds::{SeqBoundsCheck, emit_bounds_assert, failable_bounds},
        ssa::{
            BlockId, FunctionId, Instruction, SourceLocation, Terminator, ValueId,
            hlssa::{
                BinaryArithOpKind, CallTarget, Constant, HLFunction, HLSSA, LocatedOpCode, OpCode,
                builder::HLEmitter,
            },
        },
        util::ice_non_elided_tuple,
    },
};

/// An [`HLEmitter`] that appends into a plain instruction vector.
///
/// DCE's sweep builds each block's new instruction list by hand rather than using `HLBlockEmitter`,
/// but a partial op's failure check has to be built exactly the way every other pass builds it (see
/// [`crate::compiler::passes::shared`]'s `divmod_guard` and `seq_bounds`). This adapter bridges the
/// two so there is only ever one definition of each check.
struct VecEmitter<'a, 'b> {
    ssa: &'a HLSSA,
    out: &'b mut Vec<LocatedOpCode>,
    location: SourceLocation,
}

impl HLEmitter for VecEmitter<'_, '_> {
    fn fresh_value(&mut self) -> ValueId {
        self.ssa.fresh_value()
    }

    fn emit(&mut self, instruction: OpCode) {
        let located = instruction.locate(self.location.clone());
        self.out.push(located);
    }

    fn emit_located(&mut self, instruction: LocatedOpCode) {
        self.out.push(instruction);
    }

    fn emit_constant(&mut self, value: Constant) -> ValueId {
        self.ssa.add_const(value)
    }

    fn field(&self) -> FieldConfig {
        self.ssa.field()
    }
}

/// The operands of an unguarded `Div`/`Mod`, which with the failable sequence ops are the
/// instructions DCE may not simply delete when their results go dead.
///
/// Deliberately matches only at the top level: a `Guard`-wrapped division must keep today's
/// behavior, because inside an inactive branch it is required *not* to fail and
/// `lower_divmod_guard` already encodes that.
fn unguarded_divmod_operands(instruction: &OpCode) -> Option<(ValueId, ValueId)> {
    match instruction {
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Div | BinaryArithOpKind::Mod,
            lhs,
            rhs,
            ..
        } => Some((*lhs, *rhs)),
        _ => None,
    }
}

pub struct DCE {
    config: Config,
}

#[derive(Debug)]
enum WorkItem {
    LiveBlock(FunctionId, BlockId),
    LiveValue(FunctionId, ValueId),
    LiveInstruction(FunctionId, BlockId, usize),
    LiveReturnSlot(FunctionId, usize),
}

enum ValueDefinition {
    Param(BlockId, usize),
    Instruction(BlockId, usize),
}

#[derive(Clone, Copy)]
pub struct Config {
    pub witness_shape_frozen: bool,

    /// When true, all blocks are marked as live, preventing removal of empty intermediate blocks.
    ///
    /// This is a workaround for untaint_control_flow not handling multiple merge predecessors.
    /// Remove this option once untaint_control_flow properly handles multiple jumps into merge
    /// blocks.
    pub preserve_all_blocks: bool,

    /// Whether an unguarded partial op — a failable `Div`/`Mod`, or a
    /// `SlicePop`/`SliceInsert`/`SliceRemove` — whose results are dead is replaced by its failure
    /// check instead of being deleted outright.
    ///
    /// **Only ever true before witness lowering**. From `spill_witness` onward the IR also contains
    /// partial ops the compiler generated itself — `lower_unsigned_divmod` computes its quotient
    /// and remainder *hints* with ordinary `Div`/`Mod` on `value_of(..)` operands. Those are not
    /// user ops and carry no Noir-level failure semantics; the guarded path even substitutes a
    /// divisor of `1` on purpose so an inactive branch's hint stays safe.
    ///
    /// Restricting it this way loses nothing. Every *user* partial op is present from
    /// `initial_ssa` onward, so the early runs see them all; and any that survives to
    /// `spill_witness` gets its check from `LowerPureGuards` or its instruction lowering, which
    /// run before anything there can kill it.
    pub rewrite_dead_partial_ops: bool,

    /// Whether an `ArraySet` on a fixed-length array whose result is dead is replaced by its bounds
    /// check instead of being deleted outright. Reads are excluded on cost grounds, see
    /// [`SeqBoundsCheck::SeqAccess`].
    ///
    /// A witness-indexed array access does not get a bounds check until `LowerWitnessArrayOps`,
    /// which runs at `driver.rs:484`. Deleting the access before that takes the only thing that
    /// could ever fail with it, and nothing downstream puts it back; a program whose out-of-range
    /// write happens to be unread would verify. Rewriting the dead access into the check keeps it.
    ///
    /// **Only ever true before witness lowering**. From `spill_witness` onward the IR also holds
    /// array writes the compiler generated itself (the per-slot `array_set` of every rebuild scan)
    /// which carry no Noir-level failure semantics, so a check on one is wasted work. Its index is
    /// a loop counter and its bound the array's static length, so the assert folds away having
    /// bought nothing. Doing that at every slot of every scan is the cost this avoids.
    ///
    /// Restricting it that way keeps the guarantee that matters. A *user* array access is present
    /// from `initial_ssa` onward and is dead from the start if nothing ever reads it, so the early
    /// runs see it and leave the check behind as an ordinary `AssertCmp` — which is
    /// `is_initially_live`, so every later run preserves it.
    ///
    /// What is **not** covered is an access that is live pre-untaint and only becomes dead
    /// afterwards; that is the same residual the divmod rewrite carries, and closing it needs a
    /// check emitted before `driver.rs:480` rather than a different DCE configuration.
    ///
    /// Cheap where it does not matter: a check on an in-range constant index folds away downstream,
    /// and with a pure index and no guard `LowerWitnessAssertOps` leaves the `AssertCmp` alone, so
    /// it stays a witness-generation-time check and costs no R1CS row.
    ///
    /// This mirrors Noir, which re-inserts the bounds check when DIE drops an unused array access
    /// rather than losing it (`ssa/opt/die.rs`, `insert_out_of_bounds_checks`).
    ///
    /// A witness-*conditional* dead write is covered, and needs nothing extra from
    /// `failable_bounds` matching only at the top level. Every run that rewrites is pre-untaint —
    /// each `Config::preserve_blocks()` site is in `make_struct_access_static` or `pre_wti` — while
    /// `Guard` is introduced by `untaint_control_flow`, whose own type-application step panics on
    /// any it meets on input. So a rewriting run never sees a guarded access in the first place: at
    /// that point the condition is still ordinary control flow, the `AssertCmp` lands in the branch
    /// block, and untaint predicates it along with everything else there. That is what keeps the
    /// guard-off world satisfiable rather than rejecting on an index it never uses; both worlds are
    /// pinned by `array_witness_set_dead_guarded_{true,false}`.
    ///
    /// The top-level match would only need revisiting to close the residual above, since *there*
    /// the dead access is `Guard`-wrapped and the assert would have to be re-wrapped in the same
    /// condition to keep an inactive branch from failing.
    pub rewrite_dead_seq_access: bool,
}

impl Config {
    pub fn pre_r1c() -> Self {
        Self {
            witness_shape_frozen: false,
            preserve_all_blocks: false,
            rewrite_dead_partial_ops: false,
            rewrite_dead_seq_access: false,
        }
    }

    pub fn post_r1c() -> Self {
        Self {
            witness_shape_frozen: true,
            preserve_all_blocks: false,
            rewrite_dead_partial_ops: false,
            rewrite_dead_seq_access: false,
        }
    }

    /// The pre-untaint configuration, used by the phases that still hold purely user-level IR.
    /// This is the only one that rewrites dead partial ops — see
    /// [`Config::rewrite_dead_partial_ops`].
    pub fn preserve_blocks() -> Self {
        Self {
            witness_shape_frozen: false,
            preserve_all_blocks: true,
            rewrite_dead_partial_ops: true,
            rewrite_dead_seq_access: true,
        }
    }
}

impl Pass for DCE {
    fn name(&self) -> &'static str {
        "dce"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        self.do_run(ssa, store.get::<FlowAnalysis>());
    }
}

impl DCE {
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    /// Whether this run may replace a dead unguarded partial op with its failure check.
    fn rewrites_dead_partial_ops(&self) -> bool {
        debug_assert!(
            !(self.config.rewrite_dead_partial_ops && self.config.witness_shape_frozen),
            "rewriting dead partial ops would add constraints after the witness shape is frozen"
        );
        self.config.rewrite_dead_partial_ops
    }

    /// Whether a dead unguarded `Div`/`Mod` still has to leave its failure check behind.
    ///
    /// `None` means this run does not rewrite at all, so the question does not arise. Otherwise the
    /// range domain gets to discharge it: a division it proves defined has nothing left to fail, so
    /// the instruction can simply be deleted — which is the ordinary DCE outcome and the reason the
    /// mark phase consults this too. Keeping the check would otherwise resurrect the entire chain
    /// that computes the operands, purely to assert something already known.
    ///
    /// The type this asks about is the *stripped* operand type, matching `LowerPureGuards`.
    ///
    /// `None` means [`Config::rewrite_dead_partial_ops`] is off: the ranges are only calculated
    /// when the flag is set. This avoids being able to turn it back on by accident.
    fn divmod_check_survives(
        &self,
        analyses: Option<(&TypeInfo, &ValueRanges)>,
        function_id: FunctionId,
        lhs: ValueId,
        rhs: ValueId,
    ) -> bool {
        let Some((types, ranges)) = analyses else {
            return false;
        };

        let lhs_type = types.get_function(function_id).get_value_type(lhs);
        if !divmod_can_fail(lhs_type) {
            return false;
        }

        let function_ranges = ranges.get_function(function_id);

        !divmod_provably_defined(
            &function_ranges.get(lhs),
            &function_ranges.get(rhs),
            lhs_type.peel_witness(),
        )
    }

    /// Whether this run may replace a dead array access with its bounds check.
    fn rewrites_dead_seq_access(&self) -> bool {
        debug_assert!(
            !(self.config.rewrite_dead_seq_access && self.config.witness_shape_frozen),
            "rewriting dead array accesses would add constraints after the witness shape is frozen"
        );
        debug_assert!(
            !(self.config.rewrite_dead_seq_access && !self.config.preserve_all_blocks),
            "rewriting dead array accesses is only sound before witness lowering, where every array access is still a user one — see `Config::rewrite_dead_seq_access`"
        );
        self.config.rewrite_dead_seq_access
    }

    /// Whether either dead-op rewrite is enabled, and so whether the sweep needs type information.
    fn needs_rewrite_types(&self) -> bool {
        self.rewrites_dead_partial_ops() || self.rewrites_dead_seq_access()
    }

    /// Whether this run rewrites this particular dead bounds-checked op. The two families are
    /// enabled by different flags because they get their checks back at different points in the
    /// pipeline — see [`Config::rewrite_dead_seq_access`].
    fn rewrites_bounds_of(&self, check: &SeqBoundsCheck) -> bool {
        match check {
            SeqBoundsCheck::Pop { .. }
            | SeqBoundsCheck::Insert { .. }
            | SeqBoundsCheck::Remove { .. } => self.rewrites_dead_partial_ops(),
            SeqBoundsCheck::SeqAccess { .. } => self.rewrites_dead_seq_access(),
        }
    }

    fn is_initially_live(&self, instruction: &OpCode) -> bool {
        match instruction {
            OpCode::Call {
                unconstrained: true,
                ..
            } => false,
            OpCode::Call { .. } | OpCode::Store { .. } => true,
            OpCode::Assert { .. } | OpCode::AssertConstant { .. } | OpCode::AssertCmp { .. } => {
                true
            }
            OpCode::AssertR1C { .. }
            | OpCode::Constrain { .. }
            | OpCode::Lookup { .. }
            | OpCode::DLookup { .. }
            | OpCode::NextDCoeff { .. }
            | OpCode::BumpD { .. }
            | OpCode::MemOp { .. }
            | OpCode::Rangecheck { .. }
            | OpCode::Todo { .. }
            | OpCode::InitGlobal { .. }
            | OpCode::DropGlobal { .. } => true,
            OpCode::WriteWitness { pinned, .. } => self.config.witness_shape_frozen || *pinned,
            OpCode::FreshWitness { .. } => self.config.witness_shape_frozen,
            OpCode::ToBits { .. } | OpCode::ToRadix { .. } => !self.config.witness_shape_frozen,
            OpCode::Load { .. }
            | OpCode::BinaryArithOp { .. }
            | OpCode::Cmp { .. }
            | OpCode::Alloc { .. }
            | OpCode::Select { .. }
            | OpCode::ArrayGet { .. }
            | OpCode::ArraySet { .. }
            | OpCode::SlicePush { .. }
            | OpCode::SlicePop { .. }
            | OpCode::SliceInsert { .. }
            | OpCode::SliceRemove { .. }
            | OpCode::SliceLen { .. }
            | OpCode::MkSeq { .. }
            | OpCode::MkSeqOfBlob { .. }
            | OpCode::MkRepeated { .. }
            | OpCode::Cast { .. }
            | OpCode::SExt { .. }
            | OpCode::BitRange { .. }
            | OpCode::Not { .. }
            | OpCode::MulConst { .. }
            | OpCode::ReadGlobal { .. }
            | OpCode::Spread { .. }
            | OpCode::Unspread { .. } => false,
            OpCode::TupleProj { .. } | OpCode::TupleRefProj { .. } | OpCode::MkTuple { .. } => {
                ice_non_elided_tuple()
            }
            OpCode::Guard { inner, .. } => self.is_initially_live(inner.as_ref()),
        }
    }

    pub fn do_run(&self, ssa: &mut HLSSA, cfg: &FlowAnalysis) {
        let function_ids: Vec<FunctionId> = ssa.get_function_ids().collect();

        // Typed and ranged for the whole module whenever this run can rewrite, without first
        // checking whether there is anything to rewrite: finding out would cost a full instruction
        // walk of its own, which is the same order as the analyses it would be guarding.
        //
        // Both must be computed *here*, before the mark phase, because the discharge below decides
        // whether a dead division's operands are seeded live at all — and before anything mutates
        // the module. `Types` walks every instruction in every block, including the dead ones still
        // waiting to be removed, so it has to see the module whole: typing after `retain_constants`
        // panics with "Error running opcode Cast { .. }" the moment a dead instruction's only-use
        // constant has just been pruned out from under it. Nothing between here and the sweep
        // touches the SSA, so running them at the top is strictly safer than running them later.
        //
        // The types serve every rewrite path; only the divmod path consults the ranges, as the
        // bounds paths have nothing for them to discharge (see the sweep below).
        let rewrite_types: Option<TypeInfo> = self
            .needs_rewrite_types()
            .then(|| Types::new().run(ssa, cfg));
        let divmod_ranges: Option<ValueRanges> = rewrite_types
            .as_ref()
            .filter(|_| self.rewrites_dead_partial_ops())
            .map(|types| ValueRangeAnalysis::new().run(ssa, cfg, types));
        let divmod_analyses = rewrite_types.as_ref().zip(divmod_ranges.as_ref());

        debug_assert_eq!(
            divmod_analyses.is_some(),
            self.rewrites_dead_partial_ops(),
            "`divmod_check_survives` reads `None` as 'the divmod rewrite is off'; the two must \
             agree exactly or it silently re-enables itself"
        );

        let mut definitions_by_function: HashMap<FunctionId, HashMap<ValueId, ValueDefinition>> =
            HashMap::default();
        let mut static_calls_by_callee: HashMap<FunctionId, Vec<(FunctionId, BlockId, usize)>> =
            HashMap::default();

        for function_id in &function_ids {
            let function = ssa.get_function(*function_id);
            definitions_by_function.insert(*function_id, self.generate_definitions(function));

            for (block_id, block) in function.get_blocks() {
                for (i, instruction) in block.get_instructions().enumerate() {
                    if let OpCode::Call {
                        function: CallTarget::Static(callee),
                        ..
                    } = instruction
                    {
                        static_calls_by_callee.entry(*callee).or_default().push((
                            *function_id,
                            *block_id,
                            i,
                        ));
                    }
                }
            }
        }

        let mut live_values: HashMap<FunctionId, HashSet<ValueId>> = HashMap::default();
        let mut live_blocks: HashMap<FunctionId, HashSet<BlockId>> = HashMap::default();
        let mut live_instructions: HashMap<FunctionId, HashMap<BlockId, HashSet<usize>>> =
            HashMap::default();
        let mut live_params: HashMap<FunctionId, HashMap<BlockId, HashSet<usize>>> =
            HashMap::default();
        let mut live_entry_params: HashMap<FunctionId, HashSet<usize>> = HashMap::default();
        let mut live_branches: HashMap<FunctionId, HashSet<BlockId>> = HashMap::default();
        let mut live_return_slots: HashMap<FunctionId, HashSet<usize>> = HashMap::default();

        let mut worklist: Vec<WorkItem> = vec![];

        for function_id in &function_ids {
            let function = ssa.get_function(*function_id);
            worklist.push(WorkItem::LiveBlock(*function_id, function.get_entry_id()));

            if self.config.preserve_all_blocks {
                for (block_id, _) in function.get_blocks() {
                    worklist.push(WorkItem::LiveBlock(*function_id, *block_id));
                }
            }

            for (block_id, block) in function.get_blocks() {
                for (i, instruction) in block.get_instructions().enumerate() {
                    if self.is_initially_live(instruction) {
                        worklist.push(WorkItem::LiveInstruction(*function_id, *block_id, i));
                    }

                    // An unguarded partial op keeps its operands alive even when its results are
                    // dead, because the sweep replaces such an op with a check *on those
                    // operands* rather than deleting it. Note this deliberately does not mark the
                    // op itself live: `live_instructions` staying false for it is precisely
                    // the signal the sweep uses to know it should rewrite.
                    //
                    // Nothing is pessimised for an op that stays: its operands were already
                    // live. Where the rewrite does fire, an operand the check turns out not to need
                    // — `lhs`, for everything but signed division — is held live one run longer
                    // than it has to be, because the seeding runs before there is any type
                    // information to tell the cases apart. It goes dead again immediately and the
                    // *next* DCE run drops it, which at the `pre_wti` site is a later pass in the
                    // same phase; at the `make_struct_access_static` site (the last pass of that
                    // phase) it instead carries into the next phase's input, where the first DCE
                    // reclaims it.
                    //
                    // A division the range domain proves defined is exempt: no check will be left
                    // behind for it, so holding its operands live would keep the chain that
                    // computes them alive for nothing. This is where most of the saving is — the
                    // sweep only ever *avoids emitting* a few instructions, while the mark phase
                    // decides whether an entire dependency chain survives. There is no matching
                    // exemption for the sequence bounds below: nothing here can prove such a check
                    // away, so their operands are always seeded.
                    if let Some((lhs, rhs)) = unguarded_divmod_operands(instruction)
                        && self.divmod_check_survives(divmod_analyses, *function_id, lhs, rhs)
                    {
                        worklist.push(WorkItem::LiveValue(*function_id, lhs));
                        worklist.push(WorkItem::LiveValue(*function_id, rhs));
                    }

                    if self.rewrites_dead_partial_ops()
                        && let Some(
                            check @ (SeqBoundsCheck::Pop { .. }
                            | SeqBoundsCheck::Insert { .. }
                            | SeqBoundsCheck::Remove { .. }),
                        ) = failable_bounds(instruction)
                    {
                        let (slice, index) = check.operands();
                        worklist.push(WorkItem::LiveValue(*function_id, slice));
                        if let Some(index) = index {
                            worklist.push(WorkItem::LiveValue(*function_id, index));
                        }
                    }

                    // A dead array write needs only its *index* held live: the bound it is checked
                    // against comes from the array's type, not from the array value, so the
                    // container itself and everything feeding it stay collectable. Noir's DIE keeps
                    // exactly the same operand for the same reason. A slice access needs nothing,
                    // since the sweep emits no check for one.
                    if self.rewrites_dead_seq_access()
                        && let Some(SeqBoundsCheck::SeqAccess { index, .. }) =
                            failable_bounds(instruction)
                    {
                        worklist.push(WorkItem::LiveValue(*function_id, index));
                    }
                }

                if matches!(block.get_terminator(), Some(Terminator::Return(_))) {
                    worklist.push(WorkItem::LiveBlock(*function_id, *block_id));
                }
            }
        }

        while let Some(item) = worklist.pop() {
            match item {
                WorkItem::LiveBlock(function_id, block_id) => {
                    if self.block_live(&live_blocks, function_id, block_id) {
                        continue;
                    }
                    live_blocks.entry(function_id).or_default().insert(block_id);

                    let function_cfg = cfg.get_function_cfg(function_id);
                    let function = ssa.get_function(function_id);

                    if let Some(Terminator::JmpIf(condition, _, _)) =
                        function.get_block(block_id).get_terminator()
                    {
                        worklist.push(WorkItem::LiveValue(function_id, *condition));
                    }

                    for pd in function_cfg.get_post_dominance_frontier(block_id) {
                        worklist.push(WorkItem::LiveBlock(function_id, pd));
                        live_branches.entry(function_id).or_default().insert(pd);

                        // This is an invariant enforced by Flow Analysis.
                        match function.get_block(pd).get_terminator() {
                            Some(Terminator::JmpIf(condition, _, _)) => {
                                worklist.push(WorkItem::LiveValue(function_id, *condition));
                            }
                            _ => panic!("ICE: It's a frontier, must end with a conditional"),
                        }
                    }

                    if function.get_block(block_id).has_parameters() {
                        for predecessor in function_cfg.get_jumps_into(block_id) {
                            worklist.push(WorkItem::LiveBlock(function_id, predecessor));
                        }
                    }
                }
                WorkItem::LiveValue(function_id, value_id) => {
                    if live_values
                        .entry(function_id)
                        .or_default()
                        .contains(&value_id)
                    {
                        continue;
                    }
                    live_values.entry(function_id).or_default().insert(value_id);

                    let definitions = definitions_by_function
                        .get(&function_id)
                        .expect("function definitions missing");
                    let Some(definition) = definitions.get(&value_id) else {
                        continue;
                    };

                    match definition {
                        ValueDefinition::Param(block_id, i) => {
                            if self.param_live(&live_params, function_id, *block_id, *i) {
                                continue;
                            }

                            live_params
                                .entry(function_id)
                                .or_default()
                                .entry(*block_id)
                                .or_default()
                                .insert(*i);

                            let function = ssa.get_function(function_id);
                            if *block_id == function.get_entry_id() {
                                if live_entry_params.entry(function_id).or_default().insert(*i) {
                                    if let Some(callsites) =
                                        static_calls_by_callee.get(&function_id)
                                    {
                                        for (caller_fn, caller_block, caller_i) in callsites {
                                            // Only propagate callee param liveness to callsite args
                                            // if the callsite instruction is already live. Constrained
                                            // calls are always initially live so this passes immediately.
                                            // Unconstrained calls may be dead; when they later become
                                            // live, LiveInstruction handling will propagate at that point.
                                            if !self.instruction_live(
                                                &live_instructions,
                                                *caller_fn,
                                                *caller_block,
                                                *caller_i,
                                            ) {
                                                continue;
                                            }
                                            let caller = ssa.get_function(*caller_fn);
                                            if let OpCode::Call { args, .. } = caller
                                                .get_block(*caller_block)
                                                .get_instruction(*caller_i)
                                                .as_ref()
                                            {
                                                assert!(
                                                    *i < args.len(),
                                                    "ICE: live callee entry param index out of bounds at callsite"
                                                );
                                                worklist.push(WorkItem::LiveValue(
                                                    *caller_fn, args[*i],
                                                ));
                                            }
                                        }
                                    }
                                }
                            }

                            worklist.push(WorkItem::LiveBlock(function_id, *block_id));

                            let function_cfg = cfg.get_function_cfg(function_id);
                            for pred in function_cfg.get_jumps_into(*block_id) {
                                let jumpin_block = function.get_block(pred);
                                match jumpin_block.get_terminator() {
                                    Some(Terminator::Jmp(_, params)) => {
                                        assert!(
                                            *i < params.len(),
                                            "ICE: phi param index out of bounds in predecessor jump"
                                        );
                                        worklist.push(WorkItem::LiveValue(function_id, params[*i]));
                                    }
                                    _ => panic!(
                                        "ICE: the block has phis, so jumps into it must be Jmps"
                                    ),
                                }
                            }
                        }
                        ValueDefinition::Instruction(block_id, i) => {
                            let function = ssa.get_function(function_id);
                            let instruction = function.get_block(*block_id).get_instruction(*i);

                            if let OpCode::Call {
                                results,
                                function: CallTarget::Static(callee),
                                ..
                            } = instruction.as_ref()
                            {
                                if let Some(result_idx) =
                                    results.iter().position(|result| *result == value_id)
                                {
                                    worklist.push(WorkItem::LiveReturnSlot(*callee, result_idx));
                                }
                            }

                            worklist.push(WorkItem::LiveInstruction(function_id, *block_id, *i));
                        }
                    }
                }
                WorkItem::LiveInstruction(function_id, block_id, i) => {
                    if self.instruction_live(&live_instructions, function_id, block_id, i) {
                        continue;
                    }

                    live_instructions
                        .entry(function_id)
                        .or_default()
                        .entry(block_id)
                        .or_default()
                        .insert(i);

                    worklist.push(WorkItem::LiveBlock(function_id, block_id));

                    let function = ssa.get_function(function_id);
                    let instruction = function.get_block(block_id).get_instruction(i);

                    if let OpCode::Call {
                        function: CallTarget::Static(callee),
                        args,
                        ..
                    } = instruction.as_ref()
                    {
                        // When a Call becomes live, propagate already-live callee entry
                        // params back to the corresponding callsite args. Other callee
                        // params may become live later; ValueDefinition::Param handles
                        // those by revisiting already-live callsites.
                        if let Some(live_params) = live_entry_params.get(callee) {
                            for param_idx in live_params.iter() {
                                if *param_idx < args.len() {
                                    worklist
                                        .push(WorkItem::LiveValue(function_id, args[*param_idx]));
                                }
                            }
                        }
                    } else {
                        for input in instruction.get_inputs() {
                            worklist.push(WorkItem::LiveValue(function_id, *input));
                        }
                    }
                }
                WorkItem::LiveReturnSlot(function_id, slot) => {
                    if !live_return_slots
                        .entry(function_id)
                        .or_default()
                        .insert(slot)
                    {
                        continue;
                    }

                    let function = ssa.get_function(function_id);
                    for (block_id, block) in function.get_blocks() {
                        if let Some(Terminator::Return(values)) = block.get_terminator() {
                            assert!(
                                slot < values.len(),
                                "ICE: return slot index out of bounds for return terminator"
                            );
                            worklist.push(WorkItem::LiveBlock(function_id, *block_id));
                            worklist.push(WorkItem::LiveValue(function_id, values[slot]));
                        }
                    }
                }
            }
        }

        // Sweep the module-level constant storage: a constant is live iff some live instruction
        // (in any function) references its `ValueId`.
        let all_live: HashSet<ValueId> = live_values
            .values()
            .flat_map(|set| set.iter().copied())
            .collect();
        ssa.retain_constants(|vid, _| all_live.contains(vid));

        for function_id in function_ids {
            let function_cfg = cfg.get_function_cfg(function_id);
            let mut function = ssa.take_function(function_id);
            let entry_id = function.get_entry_id();

            for block_id in function_cfg.get_domination_pre_order() {
                let mut block = function.take_block(block_id);
                if !self.block_live(&live_blocks, function_id, block_id) {
                    continue;
                }

                let instructions = block.take_instructions();
                let mut new_instructions = vec![];

                for (i, mut instruction) in instructions.into_iter().enumerate() {
                    if !self.instruction_live(&live_instructions, function_id, block_id, i) {
                        // The pipeline contract behind [`Config::rewrite_dead_partial_ops`], which
                        // nothing else enforces: a run that does *not* rewrite must never be the
                        // one that deletes a failable slice op, because the only other thing that
                        // ever emits its bounds check is `InstructionLowering::slice_ops`, which
                        // replaces the op rather than surviving alongside it. Dropping one here
                        // would silently take the check with it, and an out-of-range witness index
                        // in a program that never reads the result would verify.
                        //
                        // It holds today because every DCE that can still see one of these carries
                        // `preserve_blocks()` (the two direct runs, plus SCS's and
                        // `PRE::pre_untaint`'s), untaint runs none, and `slice_ops` is the first
                        // pass of `witness_spilling`. Moving that lowering later, or adding a
                        // `pre_r1c()` run before it, breaks the contract — and fails here rather
                        // than in whichever program happens to have the unread out-of-range op.
                        //
                        // Guarded ops are deliberately out of scope, matching `failable_bounds`:
                        // inside an inactive branch they are required not to fail, so deleting one
                        // loses nothing. So is `SeqAccess` — its check comes back from
                        // `LowerWitnessArrayOps` rather than from a lowering of its own, and
                        // [`Config::rewrite_dead_seq_access`] documents the live-then-dead residual
                        // that leaves, so a non-rewriting run dropping one is expected.
                        debug_assert!(
                            self.rewrites_dead_partial_ops()
                                || !matches!(
                                    failable_bounds(&instruction),
                                    Some(
                                        SeqBoundsCheck::Pop { .. }
                                            | SeqBoundsCheck::Insert { .. }
                                            | SeqBoundsCheck::Remove { .. }
                                    )
                                ),
                            "DCE deleted a failable slice op without leaving its bounds check \
                             behind: {instruction:?}"
                        );

                        // A partial op whose results are dead must not vanish. Noir treats a bad
                        // division and an out-of-bounds slice op as execution failures whether or
                        // not anything reads the result, and mavros never sees Noir's SSA-level
                        // check, so deleting the op here would delete the only thing that could
                        // ever fail. Replace it with the check alone: the arithmetic still goes,
                        // which is the whole point of eliminating it.
                        if let Some(types) = rewrite_types.as_ref() {
                            if let Some((lhs, rhs)) = unguarded_divmod_operands(&instruction) {
                                // `divmod_check_survives` subsumes the `divmod_can_fail` type gate
                                // and additionally lets the range domain discharge the check, in
                                // which case the division just goes.
                                if self.divmod_check_survives(
                                    divmod_analyses,
                                    function_id,
                                    lhs,
                                    rhs,
                                ) {
                                    let lhs_type = types
                                        .get_function(function_id)
                                        .get_value_type(lhs)
                                        .strip_witness()
                                        .clone();
                                    let mut emitter = VecEmitter {
                                        ssa,
                                        out: &mut new_instructions,
                                        location: instruction.location().clone(),
                                    };
                                    emit_divmod_is_defined_assert(
                                        &mut emitter,
                                        lhs,
                                        rhs,
                                        &lhs_type,
                                    );
                                }
                            } else if let Some(check) = failable_bounds(&instruction)
                                && self.rewrites_bounds_of(&check)
                            {
                                // No `can_fail` gate to mirror `divmod_can_fail`: whether a seq op
                                // is in bounds turns on the index, and for a slice on the *length*,
                                // neither of which is a type property, so nothing here can decide
                                // it. The check is always emitted and folds away downstream
                                // (Click-Cooper knows the constant lengths, `SimplifyAsserts` drops
                                // the tautologies) whenever the op was in fact total — which is
                                // also what keeps a constant in-range index free.
                                let (seq, index) = check.operands();
                                let function_types = types.get_function(function_id);
                                let seq_ty = function_types.get_value_type(seq).clone();
                                let index_ty =
                                    index.map(|index| function_types.get_value_type(index).clone());
                                let mut emitter = VecEmitter {
                                    ssa,
                                    out: &mut new_instructions,
                                    location: instruction.location().clone(),
                                };
                                emit_bounds_assert(
                                    &mut emitter,
                                    &check,
                                    Some(&seq_ty),
                                    index_ty.as_ref(),
                                );
                            }
                        }
                        continue;
                    }

                    if let OpCode::Call {
                        results,
                        function: CallTarget::Static(callee),
                        args,
                        unconstrained: _,
                    } = &mut *instruction
                    {
                        let mut new_args = vec![];
                        for (arg_i, arg) in args.iter().enumerate() {
                            if self.entry_param_live(&live_entry_params, *callee, arg_i) {
                                new_args.push(*arg);
                            }
                        }
                        *args = new_args;

                        let mut new_results = vec![];
                        for (ret_i, result) in results.iter().enumerate() {
                            if self.return_slot_live(&live_return_slots, *callee, ret_i) {
                                new_results.push(*result);
                            }
                        }
                        *results = new_results;
                    }

                    new_instructions.push(instruction);
                }

                block.put_instructions(new_instructions);

                let new_terminator = match block.take_terminator() {
                    Some(Terminator::Jmp(target, params)) => {
                        if self.block_live(&live_blocks, function_id, target) {
                            let mut new_params = vec![];
                            for (i, param) in params.into_iter().enumerate() {
                                if self.block_param_live(
                                    &live_params,
                                    &live_entry_params,
                                    function_id,
                                    entry_id,
                                    target,
                                    i,
                                ) {
                                    new_params.push(param);
                                }
                            }
                            Terminator::Jmp(target, new_params)
                        } else {
                            let new_target = self.closest_live_post_dominator(
                                function_cfg,
                                block_id,
                                live_blocks.get(&function_id).unwrap_or(&HashSet::default()),
                            );
                            Terminator::Jmp(new_target, vec![])
                        }
                    }
                    Some(Terminator::JmpIf(condition, then, otherwise)) => {
                        if live_branches
                            .get(&function_id)
                            .unwrap_or(&HashSet::default())
                            .contains(&block_id)
                        {
                            Terminator::JmpIf(
                                condition,
                                self.closest_live_block(
                                    function_cfg,
                                    then,
                                    live_blocks.get(&function_id).unwrap_or(&HashSet::default()),
                                ),
                                self.closest_live_block(
                                    function_cfg,
                                    otherwise,
                                    live_blocks.get(&function_id).unwrap_or(&HashSet::default()),
                                ),
                            )
                        } else {
                            Terminator::Jmp(
                                self.closest_live_post_dominator(
                                    function_cfg,
                                    block_id,
                                    live_blocks.get(&function_id).unwrap_or(&HashSet::default()),
                                ),
                                vec![],
                            )
                        }
                    }
                    Some(Terminator::Return(values)) => {
                        let mut new_values = vec![];
                        for (i, value) in values.into_iter().enumerate() {
                            if self.return_slot_live(&live_return_slots, function_id, i) {
                                new_values.push(value);
                            }
                        }
                        Terminator::Return(new_values)
                    }
                    None => panic!("ICE: block has no terminator"),
                };

                block.set_terminator(new_terminator);

                let params = block.take_parameters();
                let mut new_params = vec![];
                for (i, param) in params.into_iter().enumerate() {
                    if self.block_param_live(
                        &live_params,
                        &live_entry_params,
                        function_id,
                        entry_id,
                        block_id,
                        i,
                    ) {
                        new_params.push(param);
                    }
                }
                block.put_parameters(new_params);

                function.put_block(block_id, block);
            }

            let old_returns = function.take_returns();
            for (i, return_type) in old_returns.into_iter().enumerate() {
                if self.return_slot_live(&live_return_slots, function_id, i) {
                    function.add_return_type(return_type);
                }
            }

            ssa.put_function(function_id, function);
        }
    }

    fn generate_definitions(&self, function: &HLFunction) -> HashMap<ValueId, ValueDefinition> {
        let mut definitions = HashMap::default();

        for (block_id, block) in function.get_blocks() {
            for (i, (val, _)) in block.get_parameters().enumerate() {
                definitions.insert(*val, ValueDefinition::Param(*block_id, i));
            }

            for (i, instruction) in block.get_instructions().enumerate() {
                for val in instruction.get_results() {
                    definitions.insert(*val, ValueDefinition::Instruction(*block_id, i));
                }
            }
        }

        definitions
    }

    fn block_live(
        &self,
        live_blocks: &HashMap<FunctionId, HashSet<BlockId>>,
        function_id: FunctionId,
        block_id: BlockId,
    ) -> bool {
        live_blocks
            .get(&function_id)
            .unwrap_or(&HashSet::default())
            .contains(&block_id)
    }

    fn instruction_live(
        &self,
        live_instructions: &HashMap<FunctionId, HashMap<BlockId, HashSet<usize>>>,
        function_id: FunctionId,
        block_id: BlockId,
        i: usize,
    ) -> bool {
        live_instructions
            .get(&function_id)
            .and_then(|blocks| blocks.get(&block_id))
            .unwrap_or(&HashSet::default())
            .contains(&i)
    }

    fn param_live(
        &self,
        live_params: &HashMap<FunctionId, HashMap<BlockId, HashSet<usize>>>,
        function_id: FunctionId,
        block_id: BlockId,
        i: usize,
    ) -> bool {
        live_params
            .get(&function_id)
            .and_then(|blocks| blocks.get(&block_id))
            .unwrap_or(&HashSet::default())
            .contains(&i)
    }

    fn entry_param_live(
        &self,
        live_entry_params: &HashMap<FunctionId, HashSet<usize>>,
        function_id: FunctionId,
        i: usize,
    ) -> bool {
        live_entry_params
            .get(&function_id)
            .unwrap_or(&HashSet::default())
            .contains(&i)
    }

    fn return_slot_live(
        &self,
        live_return_slots: &HashMap<FunctionId, HashSet<usize>>,
        function_id: FunctionId,
        i: usize,
    ) -> bool {
        live_return_slots
            .get(&function_id)
            .unwrap_or(&HashSet::default())
            .contains(&i)
    }

    fn block_param_live(
        &self,
        live_params: &HashMap<FunctionId, HashMap<BlockId, HashSet<usize>>>,
        live_entry_params: &HashMap<FunctionId, HashSet<usize>>,
        function_id: FunctionId,
        entry_id: BlockId,
        block_id: BlockId,
        i: usize,
    ) -> bool {
        if block_id == entry_id {
            return self.entry_param_live(live_entry_params, function_id, i);
        }
        self.param_live(live_params, function_id, block_id, i)
    }

    fn closest_live_block(
        &self,
        cfg: &CFG,
        block_id: BlockId,
        live_blocks: &HashSet<BlockId>,
    ) -> BlockId {
        if live_blocks.contains(&block_id) {
            return block_id;
        }
        self.closest_live_post_dominator(cfg, block_id, live_blocks)
    }

    fn closest_live_post_dominator(
        &self,
        cfg: &CFG,
        block_id: BlockId,
        live_blocks: &HashSet<BlockId>,
    ) -> BlockId {
        let mut current_block = cfg.get_post_dominator(block_id);
        while !live_blocks.contains(&current_block) {
            current_block = cfg.get_post_dominator(current_block);
        }
        current_block
    }
}
