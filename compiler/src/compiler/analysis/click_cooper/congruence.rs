//! The congruence factor: optimistic global value numbering by partition refinement.
//!
//! This is the value-numbering half of Click–Cooper (Alpern–Wegman–Zadeck, POPL '88). Every
//! like-shaped value starts assumed congruent (one class per operator kind) and classes are only
//! ever **split**, never merged. Because the partition starts maximal, loop-carried congruences —
//! two parallel induction variables that are equal across the back-edge — survive refinement.
//!
//! # Coupling and Staging
//!
//! Two values are congruent when they are computed by the same operator from operandwise-congruent
//! inputs. The factor reads two products of the constants + reachability fixpoint.
//!
//! - **Constants** are used to seed the partition — a value the lattice proved `Const(c)` is
//!   labelled by `c`, so two values equal to the same constant are congruent (the const →
//!   congruence coupling).
//! - **Reachability** scopes φ-operands — a block parameter's operands are the incoming jump-args
//!   on each *executable* predecessor edge only, so a dead in-edge never forces two φ's apart.

use std::sync::Arc;

use crate::{
    collections::{HashMap, HashSet},
    compiler::ssa::{
        BlockId, Instruction, Terminator, ValueId,
        hlssa::{BinaryArithOpKind, CastTarget, CmpKind, Constant, HLFunction, OpCode},
    },
};

// TYPES
// ================================================================================================

/// The type of the identifier for a given congruence class.
type ClassId = usize;

// CONGRUENCE
// ================================================================================================

/// The converged congruence partition for one function.
#[derive(Debug, Default)]
pub(crate) struct Congruence {
    /// A mapping from values to congruence class IDs.
    class_of: HashMap<ValueId, ClassId>,

    /// Class → members (each inner list sorted by value id). The inverse of `class_of`.
    ///
    /// The outer `Vec` is keyed by `ClassId`, which is a dense, contiguous, self-assigned `usize`
    /// (allocated as `members.len()` and never deleted — classes only ever split, never merge), and
    /// the refinement loop sweeps it by index.
    members: Vec<Vec<ValueId>>,
}

impl Congruence {
    /// `true` if `a` and `b` are proven congruent (structurally equal in every run).
    pub(crate) fn known_equal(&self, a: ValueId, b: ValueId) -> bool {
        match (self.class_of.get(&a), self.class_of.get(&b)) {
            (Some(x), Some(y)) => x == y,
            (Some(_), None) | (None, Some(_)) | (None, None) => false,
        }
    }

    /// Every value congruent to `v` (including `v`), sorted by value id. Empty if `v` is unknown.
    pub(crate) fn class_members(&self, v: ValueId) -> Vec<ValueId> {
        match self.class_of.get(&v) {
            Some(&cid) => self.members[cid].clone(),
            None => Vec::new(),
        }
    }

    /// A deterministic representative of `v`'s class (the smallest member by value id).
    ///
    /// Note: this is *not* yet guaranteed to dominate the other members, so it is not a legal
    /// redirect target on its own. The dominance-aware leader requires threading the `FlowAnalysis`
    /// dominance into the congruence partition.
    pub(crate) fn leader(&self, v: ValueId) -> Option<ValueId> {
        let &cid = self.class_of.get(&v)?;
        self.members[cid].first().copied()
    }

    /// Solve the partition for `function` over the converged reachability state.
    ///
    /// `const_of` reports the unconditional constant of a value (the constants+reachability lattice),
    /// used for the const→congruence seeding.
    pub(crate) fn build(
        function: &HLFunction,
        reachable: &HashSet<BlockId>,
        exec_edges: &HashSet<(BlockId, BlockId)>,
        const_of: impl Fn(ValueId) -> Option<Arc<Constant>>,
    ) -> Congruence {
        let entry = function.get_entry_id();

        // 1. Classify every value defined in a reachable block, and gather the full value universe
        //    (definitions plus every referenced operand).
        let mut nodes: HashMap<ValueId, Node> = HashMap::default();
        let mut universe: HashSet<ValueId> = HashSet::default();

        for (bid, block) in function.get_blocks() {
            if !reachable.contains(bid) {
                continue;
            }

            for (index, p) in block.get_parameter_values().copied().enumerate() {
                universe.insert(p);

                // Entry parameters are the function's formals: opaque at the intraprocedural level.
                let node = if *bid == entry {
                    Node::Opaque
                } else {
                    Node::Phi { block: *bid, index }
                };
                nodes.insert(p, node);
            }

            for instr in block.get_instructions() {
                for input in instr.get_inputs() {
                    universe.insert(*input);
                }
                match op_signature(instr) {
                    Some((key, operands, commutative)) => {
                        let mut results = instr.get_results();
                        if let Some(r) = results.next() {
                            universe.insert(*r);
                            nodes.insert(
                                *r,
                                Node::Op {
                                    key,
                                    operands,
                                    commutative,
                                },
                            );
                        }
                        // Pure folds are single-result; any extra results are opaque (defensive).
                        for r in results {
                            universe.insert(*r);
                            nodes.insert(*r, Node::Opaque);
                        }
                    }
                    None => {
                        for r in instr.get_results() {
                            universe.insert(*r);
                            nodes.insert(*r, Node::Opaque);
                        }
                    }
                }
            }

            match block.get_terminator() {
                Some(Terminator::Jmp(_, args)) => universe.extend(args.iter().copied()),
                Some(Terminator::JmpIf(cond, _, _)) => {
                    universe.insert(*cond);
                }
                Some(Terminator::Return(vals)) => universe.extend(vals.iter().copied()),
                None => {}
            }
        }

        let mut universe: Vec<ValueId> = universe.into_iter().collect();
        universe.sort_unstable();

        // 2. Seed the optimistic partition: one class per operand-free label. Constants group by
        //    value, like-shaped ops/φs group by operator, opaque values are singletons.
        let mut class_of: HashMap<ValueId, ClassId> = HashMap::default();
        let mut members: Vec<Vec<ValueId>> = Vec::new();
        let mut splittable: Vec<bool> = Vec::new();
        let mut label_to_class: HashMap<Label, ClassId> = HashMap::default();

        for &v in &universe {
            let label = if let Some(c) = const_of(v) {
                Label::Const(c)
            } else {
                match nodes.get(&v) {
                    Some(Node::Phi { block, .. }) => Label::Phi(*block),
                    Some(Node::Op { key, .. }) => Label::Op(key.clone()),
                    Some(Node::Opaque) | None => Label::Opaque(v),
                }
            };

            // Only operator/φ classes can split; constant and opaque classes are already exact.
            let can_split = matches!(label, Label::Phi(_) | Label::Op(_));
            let cid = *label_to_class.entry(label).or_insert_with(|| {
                members.push(Vec::new());
                splittable.push(can_split);
                members.len() - 1
            });
            members[cid].push(v);
            class_of.insert(v, cid);
        }

        // Executable predecessors per block, sorted ascending, computed once. The φ refinement
        // signature below would otherwise rescan the whole edge set on every evaluation (once per
        // φ member, per splittable class, per round). `exec_edges` is a set, so no predecessor is
        // duplicated; sorting matches the old `BTreeSet` order so φ-operand classes stay
        // positionally aligned and deterministic.
        let mut exec_preds: HashMap<BlockId, Vec<BlockId>> = HashMap::default();
        for (p, t) in exec_edges {
            exec_preds.entry(*t).or_default().push(*p);
        }
        for preds in exec_preds.values_mut() {
            preds.sort_unstable();
        }

        // 3. Refine to a stable partition: split any class whose members disagree on the classes of
        //    their operands. Splitting is monotone.
        //
        // The signature map is hoisted out of the loop and cleared per class so refinement does not
        // re-allocate it every round.
        let mut groups: HashMap<Sig, Vec<ValueId>> = HashMap::default();
        loop {
            let mut changed = false;
            let class_count = members.len();
            for cid in 0..class_count {
                if !splittable[cid] || members[cid].len() < 2 {
                    continue;
                }

                groups.clear();
                for &v in &members[cid] {
                    let sig = signature(function, &exec_preds, &nodes, &class_of, v);
                    groups.entry(sig).or_default().push(v);
                }

                if groups.len() <= 1 {
                    continue;
                }

                // Split into one class per signature, draining `groups` in its deterministic
                // iteration order. The partition and leaders are order-independent, so numbering is
                // invisible to every consumer.
                let mut drained = groups.drain();
                let (_, first) = drained.next().expect("groups has > 1 entry, checked above");
                members[cid] = first;
                for (_, g) in drained {
                    let new_cid = members.len();
                    for &v in &g {
                        class_of.insert(v, new_cid);
                    }
                    members.push(g);
                    splittable.push(true);
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        Congruence { class_of, members }
    }
}

// CLASSIFICATION
// ================================================================================================

/// What a value *is*, for congruence purposes.
enum Node {
    /// A block parameter (φ) of a non-entry block.
    ///
    /// Its operands are the incoming jump-args, one per executable predecessor edge.
    Phi { block: BlockId, index: usize },

    /// A pure, deterministic operator whose result is a function of its operand *values*.
    Op { key: OpKey, operands: Vec<ValueId>, commutative: bool },

    /// Anything else (memory, witnesses, calls, witness/runtime casts, …): not value-numbered, so it
    /// gets its own singleton class and is never congruent to another value.
    Opaque,
}

/// The operand-free operator key seeding the optimistic partition: two values can be congruent only
/// if they share this key (same operator and the same immediate, non-value attributes).
#[derive(Clone, PartialEq, Eq, Hash)]
enum OpKey {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    CmpEq,
    CmpLt,
    MulConst,
    Cast(CastTarget),
    SExt(usize, usize),
    BitRange(usize, usize),
    Not,
    Select,
}

/// The operand-free label that seeds the initial classes.
#[derive(Clone, PartialEq, Eq, Hash)]
enum Label {
    /// Grouped by constant value — the const→congruence coupling.
    Const(Arc<Constant>),

    /// φ-functions of the same join block (refined by their per-edge operands).
    Phi(BlockId),

    /// Like-shaped pure ops (refined by their operand classes).
    Op(OpKey),

    /// A value of its own — always a singleton.
    Opaque(ValueId),
}

/// A value's refinement signature.
#[derive(Clone, PartialEq, Eq, Hash)]
enum Sig {
    /// Operands mapped to their _current_ classes.
    Operands(Vec<ClassId>),

    /// A φ unresolvable on some executable edge (a non-`Jmp` predecessor) is never merged.
    Opaque(ValueId),
}

/// The pure, deterministic ops eligible for value numbering, paired with their operands and
/// commutativity.
fn op_signature(instr: &OpCode) -> Option<(OpKey, Vec<ValueId>, bool)> {
    use BinaryArithOpKind::*;
    Some(match instr {
        OpCode::BinaryArithOp { kind, lhs, rhs, .. } => {
            let (key, commutative) = match kind {
                Add => (OpKey::Add, true),
                Sub => (OpKey::Sub, false),
                Mul => (OpKey::Mul, true),
                Div => (OpKey::Div, false),
                Mod => (OpKey::Mod, false),
                And => (OpKey::And, true),
                Or => (OpKey::Or, true),
                Xor => (OpKey::Xor, true),
                Shl => (OpKey::Shl, false),
                Shr => (OpKey::Shr, false),
            };
            (key, vec![*lhs, *rhs], commutative)
        }
        OpCode::Cmp { kind, lhs, rhs, .. } => match kind {
            CmpKind::Eq => (OpKey::CmpEq, vec![*lhs, *rhs], true),
            CmpKind::Lt => (OpKey::CmpLt, vec![*lhs, *rhs], false),
        },
        OpCode::MulConst { const_val, var, .. } => (OpKey::MulConst, vec![*const_val, *var], true),
        OpCode::Cast { value, target, .. } => match target {
            CastTarget::Nop | CastTarget::Field | CastTarget::U(_) | CastTarget::I(_) => {
                (OpKey::Cast(target.clone()), vec![*value], false)
            }
            CastTarget::WitnessOf
            | CastTarget::ValueOf
            | CastTarget::ArrayToSlice
            | CastTarget::Map(_) => return None,
        },
        OpCode::SExt {
            value,
            from_bits,
            to_bits,
            ..
        } => (OpKey::SExt(*from_bits, *to_bits), vec![*value], false),
        OpCode::BitRange {
            value,
            offset,
            width,
            ..
        } => (OpKey::BitRange(*offset, *width), vec![*value], false),
        OpCode::Not { value, .. } => (OpKey::Not, vec![*value], false),
        OpCode::Select {
            cond, if_t, if_f, ..
        } => (OpKey::Select, vec![*cond, *if_t, *if_f], false),

        // Not pure/deterministic value-numbering candidates.
        OpCode::MkSeq { .. }
        | OpCode::MkSeqOfBlob { .. }
        | OpCode::MkRepeated { .. }
        | OpCode::Alloc { .. }
        | OpCode::Store { .. }
        | OpCode::Load { .. }
        | OpCode::Assert { .. }
        | OpCode::AssertCmp { .. }
        | OpCode::AssertR1C { .. }
        | OpCode::Call { .. }
        | OpCode::ArrayGet { .. }
        | OpCode::ArraySet { .. }
        | OpCode::SlicePush { .. }
        | OpCode::SliceLen { .. }
        | OpCode::ToBits { .. }
        | OpCode::ToRadix { .. }
        | OpCode::MemOp { .. }
        | OpCode::WriteWitness { .. }
        | OpCode::FreshWitness { .. }
        | OpCode::NextDCoeff { .. }
        | OpCode::BumpD { .. }
        | OpCode::Constrain { .. }
        | OpCode::Lookup { .. }
        | OpCode::DLookup { .. }
        | OpCode::Rangecheck { .. }
        | OpCode::ReadGlobal { .. }
        | OpCode::TupleProj { .. }
        | OpCode::TupleRefProj { .. }
        | OpCode::MkTuple { .. }
        | OpCode::Todo { .. }
        | OpCode::InitGlobal { .. }
        | OpCode::DropGlobal { .. }
        | OpCode::Spread { .. }
        | OpCode::Unspread { .. }
        | OpCode::Guard { .. } => return None,
    })
}

/// `v`'s current refinement signature.
///
/// `exec_preds` maps each block to its executable predecessors (sorted ascending).
fn signature(
    function: &HLFunction,
    exec_preds: &HashMap<BlockId, Vec<BlockId>>,
    nodes: &HashMap<ValueId, Node>,
    class_of: &HashMap<ValueId, ClassId>,
    v: ValueId,
) -> Sig {
    let class = |operand: &ValueId| class_of.get(operand).copied().unwrap_or(usize::MAX);
    match nodes.get(&v) {
        Some(Node::Op {
            operands,
            commutative,
            ..
        }) => {
            let mut classes: Vec<ClassId> = operands.iter().map(class).collect();
            if *commutative {
                classes.sort_unstable();
            }
            Sig::Operands(classes)
        }
        Some(Node::Phi { block, index }) => {
            // Predecessors are pre-sorted (deterministic) so positionally-aligned operand classes
            // are comparable.
            let preds: &[BlockId] = exec_preds.get(block).map(Vec::as_slice).unwrap_or(&[]);
            let mut classes = Vec::with_capacity(preds.len());
            for &p in preds {
                match function.get_block(p).get_terminator() {
                    Some(Terminator::Jmp(t, args)) if t == block && *index < args.len() => {
                        classes.push(class(&args[*index]));
                    }

                    // A parameterized block reached by a non-`Jmp` edge (or a `Jmp` that does not
                    // target this φ's block / lacks the operand) cannot be resolved here; refuse to
                    // merge it with any other φ.
                    Some(Terminator::Jmp(..))
                    | Some(Terminator::JmpIf(..))
                    | Some(Terminator::Return(..))
                    | None => return Sig::Opaque(v),
                }
            }
            Sig::Operands(classes)
        }
        Some(Node::Opaque) | None => Sig::Opaque(v),
    }
}
