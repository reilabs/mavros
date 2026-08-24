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
//!   on each _executable_ predecessor edge only, so a dead in-edge never forces two φ's apart.

use super::summary::{Sym, SymSummaries};
use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        analysis::click_cooper::def_order::DefOrder,
        ssa::{
            BlockId, FunctionId, Instruction, Terminator, ValueId,
            hlssa::{
                ArithGroup, CallTarget, CastTarget, CmpKind, Constant, HLFunction, OpCode,
                ScalarFold, SequenceTargetType, SliceOpDir, Type,
            },
        },
    },
};
use std::sync::Arc;

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

    /// Each value's dominance-aware leader.
    ///
    /// Populated by [`compute_leaders`](Congruence::compute_leaders); empty until then, so
    /// [`leader`] is only meaningful on facts that have been finalized against their CFG.
    leader_of: HashMap<ValueId, ValueId>,
}

impl Congruence {
    /// `true` if `a` and `b` are proven congruent (structurally equal in every run).
    pub(crate) fn known_equal(&self, a: ValueId, b: ValueId) -> bool {
        match (self.class_of.get(&a), self.class_of.get(&b)) {
            (Some(x), Some(y)) => x == y,
            (Some(_), None) | (None, Some(_)) | (None, None) => false,
        }
    }

    /// Every value congruent to `v` (including `v`), sorted by value id.
    ///
    /// Empty if `v` is unknown.
    pub(crate) fn class_members(&self, v: ValueId) -> &[ValueId] {
        match self.class_of.get(&v) {
            Some(&cid) => &self.members[cid],
            None => &[],
        }
    }

    /// The leader of `v`'s class: the root-most member whose definition dominates `v`'s definition,
    /// or returns `None` if `v` belongs to no class.
    ///
    /// This leader is, by definition, available at every use of `v` and is hence a legal redirect
    /// target. A consumer may replace every use of `v` with it.
    ///
    /// Populated by [`Self::compute_leaders`]. Without that finalize step, every member is its own
    /// leader.
    pub(crate) fn leader(&self, v: ValueId) -> Option<ValueId> {
        match self.leader_of.get(&v) {
            Some(&l) => Some(l),
            None => self.class_of.contains_key(&v).then_some(v),
        }
    }

    /// Build the dominance-aware [`Self::leader`] of every member using the shared definition
    /// `order`: `order.key(v)` is a total order that is a linear extension of definition dominance
    /// (if `def(w)` dominates `def(v)` then `key(w) < key(v)`), and `order.dominates_def(w, v)` is
    /// the dominance test itself.
    ///
    /// Must be called before [`Self::leader`] is queried.
    pub(crate) fn compute_leaders(&mut self, order: &DefOrder) {
        // For each member, the root-most class member dominating its definition. Every member that
        // dominates `def(v)` lies on `v`'s dominator chain, so all such members are mutually
        // comparable and the root-most is well-defined and order-independent.
        let mut leader_of = HashMap::default();
        for class in &self.members {
            let mut sorted = class.clone();
            sorted.sort_by_key(|&v| order.key(v));

            let mut chain: Vec<ValueId> = Vec::new();
            for &v in &sorted {
                while let Some(&top) = chain.last() {
                    if order.dominates_def(top, v) {
                        break;
                    }
                    chain.pop();
                }
                let leader = chain.first().copied().unwrap_or(v);
                leader_of.insert(v, leader);
                chain.push(v);
            }
        }
        self.leader_of = leader_of;
    }

    /// Solve the partition for `function` over the converged reachability state.
    ///
    /// `const_of` reports the unconditional constant of a value (the constants+reachability
    /// lattice), used for the const→congruence seeding.
    ///
    /// `call_det(g, j)` reports whether return position `j` of a static callee `g` is a
    /// deterministic function of its arguments. A constrained static call whose result is thusly
    /// deterministic is value-numbered cross-call, everything else stays opaque.
    pub(crate) fn build(
        function: &HLFunction,
        reachable: &HashSet<BlockId>,
        exec_edges: &HashSet<(BlockId, BlockId)>,
        const_of: impl Fn(ValueId) -> Option<Arc<Constant>>,
        call_det: impl Fn(FunctionId, usize) -> bool,
        sym_summaries: Option<&SymSummaries>,
    ) -> Congruence {
        let entry = function.get_entry_id();

        // 1. Classify every value defined in a reachable block, and gather the full value universe
        //    (definitions plus every referenced operand).
        let mut nodes: HashMap<ValueId, Node> = HashMap::default();
        let mut universe: HashSet<ValueId> = HashSet::default();

        // Synthetic-graft state for symbolic call returns, shared across every call site so
        // identical grafted subexpressions collapse to one synthetic id. A constrained static call
        // whose callee has a symbolic return has its result node grafted inline (below) from that
        // return expression over the call's actual arguments.
        //
        // Synthetic ids stand in for internal graft nodes that have no value in this function. They
        // must be disjoint from every real id `const_of` can resolve — not merely this function's
        // `universe`, but every program-interned constant id, since `const_of` queries the
        // whole-program snapshot and the single global monotonic id counter can place a constant
        // used only by another function _above_ this function's local max.
        let mut synthetic: HashSet<ValueId> = HashSet::default();
        let mut hashcons: HashMap<(OpKey, Vec<ValueId>), ValueId> = HashMap::default();
        let mut next_synth = u64::MAX;

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

                // A constrained static call whose result is a deterministic function of its
                // arguments is numbered structurally by `(callee, return-index)` over the argument
                // value classes. Thus, two such calls with operandwise-congruent arguments yield
                // congruent results (cross-call value numbering). Any other call result stays an
                // opaque singleton.
                //
                // An _unconstrained_ call is excluded (it stays opaque) for soundness, not just
                // conservatism: its result is prover advice, not pinned by any constraint, so two
                // such calls with congruent arguments are equal only in honest witness generation,
                // never forced equal in-circuit. Congruence here means equal in _every_ admissible
                // witness, so numbering unconstrained results congruent would underconstrain the
                // circuit. Such results are already tainted non-deterministic by
                // `analyze_determinism`, so `call_det` returns `false` for them regardless; this
                // guard makes the reason explicit.
                if let OpCode::Call {
                    results,
                    function: CallTarget::Static(g),
                    args,
                    unconstrained: false,
                } = instr
                {
                    for (j, r) in results.iter().enumerate() {
                        universe.insert(*r);

                        // A symbolic jump grafts the callee's return _expression_ over the actual
                        // arguments; it strictly refines the whole-argument `CallDet` numbering (it
                        // can ignore an unused argument and relate the result to an open
                        // expression), so prefer it. Grafting is inline: synthetic ids for internal
                        // tree nodes are minted disjoint from all real ids (see above), so no
                        // post-pass over a finalized universe is needed.
                        let sym = sym_summaries
                            .and_then(|s| s.get(g))
                            .and_then(|rets| rets.get(j))
                            .and_then(|jump| jump.as_ref());
                        if let Some(sym) = sym {
                            let node = Graft {
                                nodes: &mut nodes,
                                universe: &mut universe,
                                synthetic: &mut synthetic,
                                hashcons: &mut hashcons,
                                next_synth: &mut next_synth,
                            }
                            .instantiate_root(sym, args);
                            nodes.insert(*r, node);
                            continue;
                        }

                        let node = if call_det(*g, j) {
                            Node::op(OpKey::CallDet(*g, j), args.clone(), false)
                        } else {
                            Node::Opaque
                        };
                        nodes.insert(*r, node);
                    }
                    continue;
                }

                // The two-result slice ops are numbered per result position over the same
                // operands.
                if let OpCode::SlicePop {
                    dir,
                    result_slice,
                    result_elem,
                    slice,
                } = instr
                {
                    for (j, r) in [result_slice, result_elem].into_iter().enumerate() {
                        universe.insert(*r);
                        nodes.insert(*r, Node::op(OpKey::SlicePop(*dir, j), vec![*slice], false));
                    }
                    continue;
                }
                if let OpCode::SliceRemove {
                    result_slice,
                    result_elem,
                    slice,
                    index,
                } = instr
                {
                    for (j, r) in [result_slice, result_elem].into_iter().enumerate() {
                        universe.insert(*r);
                        nodes.insert(
                            *r,
                            Node::op(OpKey::SliceRemove(j), vec![*slice, *index], false),
                        );
                    }
                    continue;
                }

                match op_signature(instr) {
                    Some((key, operands, commutative)) => {
                        let mut results = instr.get_results();
                        if let Some(r) = results.next() {
                            universe.insert(*r);
                            nodes.insert(*r, Node::op(key, operands, commutative));
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

        // Strip the synthetic graft scaffolding. Refinement has finalized every _real_ value's
        // class (two real call results unified only through shared synthetic subnodes keep the same
        // class id), so dropping the synthetics preserves all real-value congruences while keeping
        // a def-less id out of `compute_leaders` (where `DefOrder` would treat it as defined at
        // entry and could emit it as an illegal leader) and out of every consumer query.
        if !synthetic.is_empty() {
            for v in &synthetic {
                class_of.remove(v);
            }
            for class in &mut members {
                class.retain(|v| !synthetic.contains(v));
            }
        }

        // Leaders are finalized separately by `compute_leaders` once a CFG is available; the
        // transient summary-phase partition that never exposes `leader` skips it.
        Congruence {
            class_of,
            members,
            leader_of: HashMap::default(),
        }
    }
}

// GRAFT
// ================================================================================================

/// Mutable state for grafting callees' symbolic return expressions into this function's value
/// numbering: the growing `nodes`/`universe` maps plus the synthetic-id bookkeeping shared across
/// every grafted call site (so identical subexpressions collapse to one synthetic id).
struct Graft<'a> {
    nodes: &'a mut HashMap<ValueId, Node>,
    universe: &'a mut HashSet<ValueId>,
    synthetic: &'a mut HashSet<ValueId>,
    hashcons: &'a mut HashMap<(OpKey, Vec<ValueId>), ValueId>,
    next_synth: &'a mut u64,
}

impl Graft<'_> {
    /// Instantiate an `Op`-rooted return `sym` as the node for a call result.
    ///
    /// The root becomes a real [`Node::Op`] over `sym`'s grafted operands — no synthetic for the
    /// root, which _is_ the call result value. A non-`Op` root (already covered by the constant /
    /// pass-through [`ReturnJump`](super::summary::ReturnJump)) or any unresolvable leaf yields
    /// [`Node::Opaque`].
    fn instantiate_root(&mut self, sym: &Sym, args: &[ValueId]) -> Node {
        let Sym::Op(key, commutative, children) = sym else {
            return Node::Opaque;
        };
        let mut operands = Vec::with_capacity(children.len());
        for child in children {
            match self.instantiate(child, args) {
                Some(id) => operands.push(id),
                None => return Node::Opaque,
            }
        }
        Node::op(key.clone(), operands, *commutative)
    }

    /// Instantiate a symbolic sub-expression `sym` over a call's actual `args`, returning the value
    /// id that represents it for value numbering, or `None` if a leaf is unresolvable.
    ///
    /// `Param`/`Const` leaves resolve to real ids (the argument, or the program-interned constant
    /// id, which is added to `universe` so the labelling pass numbers it). An internal `Op` mints
    /// (or, via `hashcons`, reuses) a synthetic id whose node is registered in `nodes` and recorded
    /// in `synthetic` for stripping after refinement.
    fn instantiate(&mut self, sym: &Sym, args: &[ValueId]) -> Option<ValueId> {
        match sym {
            Sym::Param(i) => args.get(*i).copied(),
            Sym::Const(id) => {
                self.universe.insert(*id);
                Some(*id)
            }
            Sym::Op(key, commutative, children) => {
                let mut operands = Vec::with_capacity(children.len());
                for child in children {
                    operands.push(self.instantiate(child, args)?);
                }

                let cache_key = (key.clone(), operands.clone());
                if let Some(&existing) = self.hashcons.get(&cache_key) {
                    return Some(existing);
                }

                let s = ValueId(*self.next_synth);
                *self.next_synth -= 1;
                self.nodes
                    .insert(s, Node::op(key.clone(), operands, *commutative));
                self.universe.insert(s);
                self.synthetic.insert(s);
                self.hashcons.insert(cache_key, s);
                Some(s)
            }
        }
    }
}

// CLASSIFICATION
// ================================================================================================

/// What a value _is_, for congruence purposes.
enum Node {
    /// A block parameter (φ) of a non-entry block.
    ///
    /// Its operands are the incoming jump-args, one per executable predecessor edge.
    Phi { block: BlockId, index: usize },

    /// A pure, deterministic operator whose result is a function of its operand _values_.
    Op { key: OpKey, operands: Vec<ValueId>, commutative: bool },

    /// Anything else (memory, witnesses, calls, witness/runtime casts, …): not value-numbered, so it
    /// gets its own singleton class and is never congruent to another value.
    Opaque,
}

impl Node {
    /// A pure-operator node (the common [`Node::Op`] constructor, shared by the direct build and the
    /// symbolic graft).
    fn op(key: OpKey, operands: Vec<ValueId>, commutative: bool) -> Node {
        Node::Op {
            key,
            operands,
            commutative,
        }
    }
}

/// The operand-free operator key seeding the optimistic partition: two values can be congruent only
/// if they share this key (same operator and the same immediate, non-value attributes).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum OpKey {
    /// A binary arithmetic operation: which operation, plus the sign it reads its operands with.
    ///
    /// `None` is not "unsigned" — it says the operation has no signed form at all, which is true
    /// only of `And`, `Or` and `Xor`. Every other group keeps its two forms in separate classes.
    Bin(ArithGroup, Option<bool>),
    CmpEq,
    CmpLt(bool),
    MulConst,
    Cast(CastTarget),
    SExt(usize, usize),
    BitRange(usize, usize),
    Not,
    Select,

    // Pure, value-semantic sequence ops. These are _not_ scalar-foldable for now, so they are
    // value-numbered but never constant-folded. The non-value attributes (`seq_type`, `elem_type`,
    // repeat count, push direction) ride in the key so differently-shaped sequences never share a
    // class.
    ArrayGet,
    ArraySet,
    SliceLen,
    MkSeq(SequenceTargetType, Type),
    MkRepeated(SequenceTargetType, usize, Type),
    MkSeqOfBlob(Type),
    SlicePush(SliceOpDir),
    SliceInsert,
    SlicePop(SliceOpDir, usize),
    SliceRemove(usize),

    /// Return position `usize` of a constrained static call to `FunctionId` whose result is a
    /// deterministic function of the call's arguments (its operands). Two such calls to the same
    /// callee with operandwise-congruent arguments are congruent.
    CallDet(FunctionId, usize),
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

/// The value operands of a pure, deterministic value-numbering op, or `None` if `instr` is not such
/// an op (memory, witnesses, calls, …).
///
/// Shared with the interprocedural determinism analysis ([`super::summary`]) so the two agree on
/// exactly which opcodes are deterministic functions of their operands — a single source of truth.
pub(crate) fn pure_op_operands(instr: &OpCode) -> Option<Vec<ValueId>> {
    op_signature(instr).map(|(_, operands, _)| operands)
}

/// The pure, deterministic ops eligible for value numbering, paired with their operands and
/// commutativity.
///
/// Value numbering is a strict _superset_ of [`OpCode::scalar_fold`]: it covers every foldable
/// scalar op (minus witness casts, which are foldable in name but never value-numbered) **plus**
/// the pure, value-semantic sequence ops, which are value-numbered here yet are not scalar-foldable
/// (their aggregate results never enter the constant lattice). So it may disagree with
/// `is_pure_scalar_fold` / `solver::transfer` on the sequence ops by design, but never on a scalar
/// op.
pub(crate) fn op_signature(instr: &OpCode) -> Option<(OpKey, Vec<ValueId>, bool)> {
    // Sequence ops are pure and deterministic but not scalar-foldable, so they are matched directly
    // rather than via `scalar_fold`. Sequence order is significant, so none are commutative.
    match instr {
        OpCode::ArrayGet { array, index, .. } => {
            return Some((OpKey::ArrayGet, vec![*array, *index], false));
        }
        OpCode::ArraySet {
            array,
            index,
            value,
            ..
        } => {
            return Some((OpKey::ArraySet, vec![*array, *index, *value], false));
        }
        OpCode::SliceLen { slice, .. } => return Some((OpKey::SliceLen, vec![*slice], false)),
        OpCode::MkSeq {
            elems,
            seq_type,
            elem_type,
            ..
        } => {
            return Some((
                OpKey::MkSeq(*seq_type, elem_type.clone()),
                elems.clone(),
                false,
            ));
        }
        OpCode::MkRepeated {
            element,
            seq_type,
            count,
            elem_type,
            ..
        } => {
            return Some((
                OpKey::MkRepeated(*seq_type, *count, elem_type.clone()),
                vec![*element],
                false,
            ));
        }
        OpCode::MkSeqOfBlob {
            element_type, blob, ..
        } => {
            return Some((OpKey::MkSeqOfBlob(element_type.clone()), vec![*blob], false));
        }
        OpCode::SlicePush {
            dir, slice, values, ..
        } => {
            let mut operands = Vec::with_capacity(values.len() + 1);
            operands.push(*slice);
            operands.extend(values.iter().copied());
            return Some((OpKey::SlicePush(*dir), operands, false));
        }
        OpCode::SliceInsert {
            slice,
            index,
            value,
            ..
        } => {
            return Some((OpKey::SliceInsert, vec![*slice, *index, *value], false));
        }
        _ => {}
    }

    // The sign participates in the key wherever the operation has one, and the opcode is the
    // authority on it — never the operand type, which no longer carries a sign to read.
    //
    // A signed and an unsigned form never share a key, including for `Add`/`Sub`/`Mul`/`Shl` whose
    // results agree bit for bit: the merge picks a survivor, and the survivor's opcode is what
    // selects the rejecting constraints emitted later. See `BinaryArithOpKind::signedness`.
    use ArithGroup::*;
    Some(match instr.scalar_fold()? {
        ScalarFold::Bin { kind, lhs, rhs } => {
            let group = kind.group();
            let commutative = match group {
                Add | Mul | And | Or | Xor => true,
                Sub | Div | Rem | Shl | Shr => false,
            };
            (
                OpKey::Bin(group, kind.signedness()),
                vec![lhs, rhs],
                commutative,
            )
        }
        // The comparisons split the same way: the ordering arm carries its sign, `Eq` has none to
        // carry. Matched so that a new `CmpKind` is a compile error here rather than a silent
        // fall-through into one of these two keys.
        ScalarFold::Cmp { kind, lhs, rhs } => match kind {
            CmpKind::Eq => (OpKey::CmpEq, vec![lhs, rhs], true),
            CmpKind::ULt | CmpKind::SLt => (OpKey::CmpLt(kind.is_signed()), vec![lhs, rhs], false),
        },
        ScalarFold::MulConst { const_val, var } => (OpKey::MulConst, vec![const_val, var], true),
        ScalarFold::Cast { target, value } => match target {
            CastTarget::Nop | CastTarget::Field | CastTarget::Int(_) => {
                (OpKey::Cast(target.clone()), vec![value], false)
            }
            // Foldable scalar ops, but never value-numbered: a witness cast is opaque.
            CastTarget::WitnessOf
            | CastTarget::ValueOf
            | CastTarget::ArrayToSlice
            | CastTarget::Map(_) => return None,
        },
        ScalarFold::SExt {
            value,
            from_bits,
            to_bits,
        } => (OpKey::SExt(from_bits, to_bits), vec![value], false),
        ScalarFold::BitRange {
            value,
            offset,
            width,
        } => (OpKey::BitRange(offset, width), vec![value], false),
        ScalarFold::Not { value } => (OpKey::Not, vec![value], false),
        ScalarFold::Select { cond, if_t, if_f } => (OpKey::Select, vec![cond, if_t, if_f], false),
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

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ssa::hlssa::BinaryArithOpKind;

    fn key(instr: OpCode) -> OpKey {
        op_signature(&instr)
            .expect("a binary operation has a value-numbering key")
            .0
    }

    fn arith(kind: BinaryArithOpKind) -> OpKey {
        key(OpCode::BinaryArithOp {
            kind,
            result: ValueId(0),
            lhs: ValueId(1),
            rhs: ValueId(2),
        })
    }

    fn cmp(kind: CmpKind) -> OpKey {
        key(OpCode::Cmp {
            kind,
            result: ValueId(0),
            lhs: ValueId(1),
            rhs: ValueId(2),
        })
    }

    /// Every group that has two forms, paired unsigned-first.
    const SIGNED_PAIRS: [(BinaryArithOpKind, BinaryArithOpKind); 7] = [
        (BinaryArithOpKind::UAdd, BinaryArithOpKind::SAdd),
        (BinaryArithOpKind::USub, BinaryArithOpKind::SSub),
        (BinaryArithOpKind::UMul, BinaryArithOpKind::SMul),
        (BinaryArithOpKind::UDiv, BinaryArithOpKind::SDiv),
        (BinaryArithOpKind::URem, BinaryArithOpKind::SRem),
        (BinaryArithOpKind::UShl, BinaryArithOpKind::SShl),
        (BinaryArithOpKind::UShr, BinaryArithOpKind::SShr),
    ];

    #[test]
    fn a_signed_and_an_unsigned_form_never_share_a_key() {
        // Merging these does not lose an optimisation, it computes the wrong thing: the survivor of
        // the class is lowered for the sign it happens to carry, and for `Add`/`Sub`/`Mul` that
        // silently swaps one overflow check for the other. No exceptions -- in particular not for
        // the groups whose two forms agree bit for bit, which is the exception that used to be here.
        for (unsigned, signed) in SIGNED_PAIRS {
            assert_ne!(
                arith(unsigned),
                arith(signed),
                "{unsigned:?} and {signed:?} must not be congruent over the same operands"
            );
        }
        assert_ne!(cmp(CmpKind::ULt), cmp(CmpKind::SLt));
    }

    #[test]
    fn the_sign_free_operations_are_unaffected() {
        // `And`/`Or`/`Xor` have one form, but `Eq` reads no sign. None of them may acquire a
        // discriminator that would stop them merging with themselves.
        use BinaryArithOpKind::*;
        for kind in [And, Or, Xor] {
            assert_eq!(arith(kind), arith(kind));
        }
        assert_eq!(cmp(CmpKind::Eq), cmp(CmpKind::Eq));
        assert_ne!(cmp(CmpKind::Eq), cmp(CmpKind::ULt));
    }
}
