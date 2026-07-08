//! The elimination sweep: full-redundancy removal against the congruence partition.
//!
//! This subsumes a simple dominance-scoped CSE pass. Redundancy is detected through two cooperating
//! channels, both recording into one shared [`ValueReplacements`] map:
//!
//! - **Leader Redirects:** Every eligible instruction result is redirected to the dominance-aware
//!   leader of its congruence class ([`ClickCooper::leader`]). This inherits everything the
//!   optimistic partition can prove — loop-carried congruences, cross-call (`CallDet` / symbolic
//!   graft) numbering, const-seeded classes — none of which expression hashing can see. (Cross-call
//!   numbering is inherited only through expressions *derived from* congruent call results; call
//!   results themselves are never redirected — see the exclusions below.)
//! - **Canonical-Key Dedup:** A CSE-style interner keyed over *congruence classes* rather than raw
//!   values: commutative `Add`/`Mul`/`And`/`Or`/`Xor` chains are flattened and sorted (`And`/`Or`
//!   deduped, `MulConst` unified with `Mul`), and every terminal operand resolves to its class
//!   representative ([`ClickCooper::class_key`]). This reproduces the CSE interner's equivalences
//!   the binary value numbering cannot express (reassociation, `MulConst` ≡ `Mul`) and covers the
//!   ops congruence deliberately does not number (witness-boundary casts, `ToBits`/`ToRadix`,
//!   `ReadGlobal`, unpinned `WriteWitness` slot-sharing). Operands are chased through the _pending_
//!   replacements as keys are built, so a dedup enables its dependents' dedup within the same run
//!   (the interner-DAG effect CSE gets from expression hashing).
//!
//! Availability stays dominance-scoped exactly as in CSE (`can_replace`: earlier in the same block,
//! or in a dominating block) — no code moves in this sweep. Side-effecting assertions (`Rangecheck`,
//! and `Lookup` when configured) are deduplicated by _dropping_ a dominated duplicate, never
//! redirected. Redirected definitions are left in place for the pass's integrated DCE.
//!
//! Deliberately _not_ deduplicated in this stage, matching the CSE it replaces: aggregate
//! constructors (`MkSeq`/`MkRepeated`/`MkSeqOfBlob`/`ArraySet`/`SlicePush`/`SliceLen`), calls
//! (including congruent deterministic call results), pinned `WriteWitness`/`FreshWitness`,
//! dynamic `ToRadix`, and `Guard`-wrapped ops (opaque). Block parameters are never redirected (that
//! is `TrivialPhiElimination`/`DeduplicatePhis` territory), though they do serve as redirect
//! _targets_ through their classes.

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        analysis::{click_cooper::ClickCooper, flow_analysis::CFG, types::FunctionTypeInfo},
        passes::shared::{availability::can_replace, value_replacements::ValueReplacements},
        ssa::{
            FunctionId, ProgramPoint, ValueId,
            hlssa::{
                BinaryArithOpKind, CastTarget, Endianness, HLFunction, LookupTarget, OpCode, Radix,
                Type,
            },
        },
        util::ice_non_elided_tuple,
    },
};

// ELIMINATION
// ================================================================================================

/// Run the elimination sweep over one function.
///
/// Returns the canonical key of every keyed instruction result — the value-equivalence map the
/// motion stages plan against (two values with one key compute equal values in every run).
pub(crate) fn eliminate_function(
    function: &mut HLFunction,
    cc: &ClickCooper,
    fid: FunctionId,
    types: &FunctionTypeInfo,
    cfg: &CFG,
    deduplicate_lookups: bool,
) -> HashMap<ValueId, NodeId> {
    let mut eliminator = Eliminator {
        cc,
        fid,
        cfg,
        types,
        deduplicate_lookups,
        nodes: Vec::new(),
        ids: HashMap::default(),
        node_of: HashMap::default(),
        replacements: ValueReplacements::new(),
        value_groups: HashMap::default(),
        assert_groups: HashMap::default(),
        to_remove: HashSet::default(),
    };
    eliminator.run(function);
    eliminator.node_of
}

/// Per-function elimination state.
struct Eliminator<'a> {
    cc: &'a ClickCooper,
    fid: FunctionId,
    cfg: &'a CFG,
    types: &'a FunctionTypeInfo,
    deduplicate_lookups: bool,

    /// The canonical-key interner ([`NodeId`] -> node and node -> [`NodeId`]).
    nodes: Vec<CanonNode>,
    ids: HashMap<CanonNode, NodeId>,

    /// Each processed instruction result's canonical key, consulted for chain flattening and for
    /// structural keying of values the congruence partition did not number.
    node_of: HashMap<ValueId, NodeId>,

    /// The single shared redirect map both channels record into. Reads during key construction
    /// chase entries recorded earlier in the walk, composing nested dedups within one run.
    replacements: ValueReplacements,

    /// Group leaders per canonical key: the occurrences no earlier *same-typed* occurrence covers.
    ///
    /// A group may hold dominance-comparable leaders of different types (the witness-wrapper
    /// asymmetry); the scan in [`Self::record_value`] is type-guarded, so each type's leaders are
    /// mutually non-dominating among themselves.
    value_groups: HashMap<NodeId, Vec<(ProgramPoint, ValueId)>>,

    /// Group leaders per assertion key.
    assert_groups: HashMap<AssertKey, Vec<ProgramPoint>>,

    /// Dominated duplicate assertions, dropped at apply time (pristine indices).
    to_remove: HashSet<ProgramPoint>,
}

impl Eliminator<'_> {
    fn run(&mut self, function: &mut HLFunction) {
        // Channel 1: Congruence-leader redirects. Complete before the canonical-key walk so its
        // entries are visible to every key's operand chase.
        for block_id in self.cfg.get_domination_pre_order() {
            for instruction in function.get_block(block_id).get_instructions() {
                let Some(result) = leader_redirect_candidate(instruction) else {
                    continue;
                };
                let Some(leader) = self.cc.leader(self.fid, result) else {
                    continue;
                };
                if leader == result
                    || self.types.get_value_type(result) != self.types.get_value_type(leader)
                {
                    continue;
                }
                self.replacements.insert(result, leader);
            }
        }

        // Channel 2: Canonical-key dedup + assertion dedup, one incremental preorder walk. SSA
        // guarantees a definition precedes its uses in this order, so by the time an operand is
        // resolved every dedup it participates in has already been recorded.
        for block_id in self.cfg.get_domination_pre_order() {
            let block = function.get_block(block_id);
            for (index, instruction) in block.get_instructions().enumerate() {
                let point = ProgramPoint::new(block_id, index);
                self.visit(point, instruction);
            }
        }

        // Apply: Drop dominated duplicate assertions (by pristine index) and rewrite every inst's
        // input and terminator through the redirect map, over the same reachable blocks the facts
        // were computed on. (An unreachable block keeps its original operands: its defs are never
        // removed here, and DCE seeds liveness from every block, so nothing dangles.) Redirected
        // definitions stay for the integrated DCE to sweep.
        for block_id in self.cfg.get_domination_pre_order() {
            let block = function.get_block_mut(block_id);
            let old_instructions = block.take_instructions();
            let mut new_instructions = Vec::with_capacity(old_instructions.len());
            for (index, mut instruction) in old_instructions.into_iter().enumerate() {
                if self.to_remove.contains(&ProgramPoint::new(block_id, index)) {
                    continue;
                }
                self.replacements.replace_inputs(&mut instruction);
                new_instructions.push(instruction);
            }
            block.put_instructions(new_instructions);
            self.replacements
                .replace_terminator(block.get_terminator_mut());
        }
    }

    /// Key one instruction and record it into the value or assertion channel.
    fn visit(&mut self, point: ProgramPoint, instruction: &OpCode) {
        use BinaryArithOpKind::*;
        match instruction {
            OpCode::BinaryArithOp {
                kind,
                result,
                lhs,
                rhs,
            } => {
                let node = match kind {
                    Add => self.chain(ChainKind::Add, *result, &[*lhs, *rhs]),
                    Mul => self.chain(ChainKind::Mul, *result, &[*lhs, *rhs]),
                    And => self.chain(ChainKind::And, *result, &[*lhs, *rhs]),
                    Or => self.chain(ChainKind::Or, *result, &[*lhs, *rhs]),
                    Xor => self.chain(ChainKind::Xor, *result, &[*lhs, *rhs]),
                    Sub => self.binary(*lhs, *rhs, |lhs, rhs| CanonNode::Sub { lhs, rhs }),
                    Div => self.binary(*lhs, *rhs, |lhs, rhs| CanonNode::Div { lhs, rhs }),
                    Mod => self.binary(*lhs, *rhs, |lhs, rhs| CanonNode::Mod { lhs, rhs }),
                    Shl => self.binary(*lhs, *rhs, |lhs, rhs| CanonNode::Shl { lhs, rhs }),
                    Shr => self.binary(*lhs, *rhs, |lhs, rhs| CanonNode::Shr { lhs, rhs }),
                };
                self.record_value(point, *result, node);
            }
            // Unified with `Mul` chains so `MulConst` dedups against `BinaryArithOp::Mul`.
            OpCode::MulConst {
                result,
                const_val,
                var,
            } => {
                let node = self.chain(ChainKind::Mul, *result, &[*const_val, *var]);
                self.record_value(point, *result, node);
            }
            OpCode::Cmp {
                kind,
                result,
                lhs,
                rhs,
            } => {
                let node = match kind {
                    crate::compiler::ssa::hlssa::CmpKind::Eq => {
                        self.binary(*lhs, *rhs, |lhs, rhs| CanonNode::Eq { lhs, rhs })
                    }
                    crate::compiler::ssa::hlssa::CmpKind::Lt => {
                        self.binary(*lhs, *rhs, |lhs, rhs| CanonNode::Lt { lhs, rhs })
                    }
                };
                self.record_value(point, *result, node);
            }
            OpCode::Not { result, value } => {
                let node = CanonNode::Not(self.operand(*value));
                let node = self.intern(node);
                self.record_value(point, *result, node);
            }
            OpCode::Select {
                result,
                cond,
                if_t,
                if_f,
            } => {
                let node = CanonNode::Select {
                    cond: self.operand(*cond),
                    then: self.operand(*if_t),
                    otherwise: self.operand(*if_f),
                };
                let node = self.intern(node);
                self.record_value(point, *result, node);
            }
            OpCode::ArrayGet {
                result,
                array,
                index,
            } => {
                let node = CanonNode::ArrayGet {
                    array: self.operand(*array),
                    index: self.operand(*index),
                };
                let node = self.intern(node);
                self.record_value(point, *result, node);
            }
            OpCode::Cast {
                result,
                value,
                target,
            } => {
                let node = CanonNode::Cast {
                    value: self.operand(*value),
                    target: target.clone(),
                };
                let node = self.intern(node);
                self.record_value(point, *result, node);
            }
            OpCode::SExt {
                result,
                value,
                from_bits,
                to_bits,
            } => {
                let node = CanonNode::SExt {
                    value: self.operand(*value),
                    from_bits: *from_bits,
                    to_bits: *to_bits,
                };
                let node = self.intern(node);
                self.record_value(point, *result, node);
            }
            OpCode::BitRange {
                result,
                value,
                offset,
                width,
            } => {
                let node = CanonNode::BitRange {
                    value: self.operand(*value),
                    offset: *offset,
                    width: *width,
                };
                let node = self.intern(node);
                self.record_value(point, *result, node);
            }
            OpCode::ToBits {
                result,
                value,
                endianness,
                count,
            } => {
                let node = CanonNode::BitsOf {
                    value: self.operand(*value),
                    endianness: *endianness,
                    count: *count,
                };
                let node = self.intern(node);
                self.record_value(point, *result, node);
            }
            OpCode::ToRadix {
                result,
                value,
                radix,
                endianness,
                count,
            } => match radix {
                // `Dyn(_)` carries a runtime bound; only the static `Bytes` form is keyed.
                Radix::Bytes => {
                    let node = CanonNode::BytesOf {
                        value: self.operand(*value),
                        endianness: *endianness,
                        count: *count,
                    };
                    let node = self.intern(node);
                    self.record_value(point, *result, node);
                }
                Radix::Dyn(_) => {}
            },
            OpCode::ReadGlobal { result, offset, .. } => {
                let node = self.intern(CanonNode::ReadGlobal(*offset));
                self.record_value(point, *result, node);
            }
            OpCode::WriteWitness {
                result: Some(result),
                value,
                pinned: false,
            } => {
                let node = CanonNode::Witness(self.operand(*value));
                let node = self.intern(node);
                self.record_value(point, *result, node);
            }
            OpCode::Rangecheck { value, max_bits } => {
                let key = AssertKey::Rangecheck {
                    value: self.operand(*value),
                    max_bits: *max_bits,
                };
                self.record_assertion(point, key);
            }
            OpCode::Lookup { target, args, flag } if self.deduplicate_lookups => {
                let target = match target {
                    LookupTarget::Rangecheck(bits) => LookupKey::Rangecheck(*bits),
                    LookupTarget::DynRangecheck(bound) => {
                        LookupKey::DynRangecheck(self.operand(*bound))
                    }
                    LookupTarget::Array(array) => LookupKey::Array(self.operand(*array)),
                    LookupTarget::Spread(bits) => LookupKey::Spread(*bits),
                };
                let key = AssertKey::Lookup {
                    target,
                    args: args.iter().map(|arg| self.operand(*arg)).collect(),
                    flag: self.operand(*flag),
                };
                self.record_assertion(point, key);
            }
            OpCode::TupleProj { .. } | OpCode::TupleRefProj { .. } | OpCode::MkTuple { .. } => {
                ice_non_elided_tuple()
            }
            // Everything else is never keyed: pinned witness writes and `FreshWitness`
            // (nondeterministic advice), memory and calls, constraint emitters, aggregate
            // constructors (deferred), `Guard` wrappers (opaque), non-dedup lookups.
            _ => {}
        }
    }

    /// Build the flattened commutative-chain key for an `Add`/`Mul`/`And`/`Or`/`Xor`-family
    /// instruction producing `result` from `operands`.
    fn chain(&mut self, kind: ChainKind, result: ValueId, operands: &[ValueId]) -> NodeId {
        let ty = self.types.get_value_type(result).clone();
        let mut parts = Vec::new();
        for &operand in operands {
            self.extend_chain(kind, &ty, operand, &mut parts);
        }
        parts.sort_unstable();
        if matches!(kind, ChainKind::And | ChainKind::Or) {
            parts.dedup();
        }
        self.intern(CanonNode::Chain(kind, parts, ty))
    }

    /// Flatten `operand` into a chain of `kind` and type `ty`: a same-kind, same-type chain
    /// contributes its parts; anything else contributes one terminal leaf. The type condition keeps
    /// flattening within a single modulus.
    fn extend_chain(
        &mut self,
        kind: ChainKind,
        ty: &Type,
        operand: ValueId,
        out: &mut Vec<NodeId>,
    ) {
        let operand = self.replacements.get_replacement(operand);
        if let Some(&nid) = self.node_of.get(&operand)
            && let CanonNode::Chain(k, parts, t) = &self.nodes[nid.0 as usize]
            && *k == kind
            && t == ty
        {
            out.extend(parts.iter().copied());
            return;
        }
        out.push(self.leaf(operand));
    }

    fn binary(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
        make: impl FnOnce(NodeId, NodeId) -> CanonNode,
    ) -> NodeId {
        let node = make(self.operand(lhs), self.operand(rhs));
        self.intern(node)
    }

    /// Resolve an operand to its terminal leaf: chase pending redirects, then key by congruence
    /// class where the partition numbered the value, by the value's own structural key where one
    /// was recorded (the non-numbered ops: witness casts, `ToBits`, `ReadGlobal`, ...), and by raw
    /// identity otherwise.
    fn operand(&mut self, value: ValueId) -> NodeId {
        let value = self.replacements.get_replacement(value);
        self.leaf(value)
    }

    /// [`Self::operand`] for an already-chased value.
    fn leaf(&mut self, value: ValueId) -> NodeId {
        if let Some(class) = self.cc.class_key(self.fid, value) {
            return self.intern(CanonNode::Class(class));
        }
        if let Some(&nid) = self.node_of.get(&value) {
            return nid;
        }
        self.intern(CanonNode::Var(value))
    }

    fn intern(&mut self, node: CanonNode) -> NodeId {
        if let Some(&id) = self.ids.get(&node) {
            return id;
        }
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node.clone());
        self.ids.insert(node, id);
        id
    }

    /// Join `value` to an existing dominating *same-typed* group of `node` (recording the redirect)
    /// or start a new group with it.
    ///
    /// Occurrences arrive in domination preorder, so an earlier group leader can never be dominated
    /// by a later occurrence of its own type.
    ///
    /// The type guard scopes dedup to same-typed evaluations — one key can in principle carry
    /// occurrences of distinct types (the witness-wrapper asymmetry the motion stage also guards
    /// against). A mismatched dominating leader neither redirects this occurrence nor stops it from
    /// leading its own type's group.
    fn record_value(&mut self, point: ProgramPoint, value: ValueId, node: NodeId) {
        self.node_of.insert(value, node);
        let groups = self.value_groups.entry(node).or_default();
        for (leader_point, leader) in groups.iter() {
            if self.types.get_value_type(*leader) == self.types.get_value_type(value)
                && can_replace(self.cfg, *leader_point, point)
            {
                if self.replacements.get_replacement(value) == value {
                    self.replacements.insert(value, *leader);
                }
                return;
            }
        }
        groups.push((point, value));
    }

    /// Drop the assertion at `point` if a same-key assertion dominates it, else record it as a
    /// group leader.
    fn record_assertion(&mut self, point: ProgramPoint, key: AssertKey) {
        let groups = self.assert_groups.entry(key).or_default();
        for leader_point in groups.iter() {
            if can_replace(self.cfg, *leader_point, point) {
                self.to_remove.insert(point);
                return;
            }
        }
        groups.push(point);
    }
}

// CANONICAL EXPRESSION KEYS
// ================================================================================================

/// An interned canonical-key handle.
///
/// IDs are assigned in first-encounter order over the deterministic domination-preorder walk, so
/// they are stable across runs. Only key _equality_ is meaningful outside this module (the motion
/// stages compare them; the nodes themselves stay private).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct NodeId(u32);

/// The flattenable (associative and commutative) chain operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ChainKind {
    Add,
    Mul,
    And,
    Or,
    Xor,
}

/// A canonical expression key.
///
/// Terminal operands are congruence-class representatives ([`CanonNode::Class`]) wherever the
/// analysis numbered the value, so two keys are equal only when the computations produce equal
/// values in every run.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum CanonNode {
    /// A value known to the congruence partition, named by its class key (minimum member id).
    Class(ValueId),

    /// A value the partition did not number and no structural key exists for.
    Var(ValueId),

    /// A flattened, sorted commutative chain (`And`/`Or` additionally deduped).
    ///
    /// The result type rides in the key so chains of different widths never merge, and flattening
    /// never crosses a width change (wrapping arithmetic is associative only within one modulus).
    Chain(ChainKind, Vec<NodeId>, Type),
    Sub {
        lhs: NodeId,
        rhs: NodeId,
    },
    Div {
        lhs: NodeId,
        rhs: NodeId,
    },
    Mod {
        lhs: NodeId,
        rhs: NodeId,
    },
    Shl {
        lhs: NodeId,
        rhs: NodeId,
    },
    Shr {
        lhs: NodeId,
        rhs: NodeId,
    },
    Eq {
        lhs: NodeId,
        rhs: NodeId,
    },
    Lt {
        lhs: NodeId,
        rhs: NodeId,
    },
    Not(NodeId),
    Select {
        cond: NodeId,
        then: NodeId,
        otherwise: NodeId,
    },
    ArrayGet {
        array: NodeId,
        index: NodeId,
    },
    BitRange {
        value: NodeId,
        offset: usize,
        width: usize,
    },
    SExt {
        value: NodeId,
        from_bits: usize,
        to_bits: usize,
    },
    Cast {
        value: NodeId,
        target: CastTarget,
    },
    BitsOf {
        value: NodeId,
        endianness: Endianness,
        count: usize,
    },
    BytesOf {
        value: NodeId,
        endianness: Endianness,
        count: usize,
    },

    /// An unpinned `WriteWitness` keyed by its hint: two such writes hold the same honest value
    /// and may share a slot (merging only removes adversarial freedom).
    Witness(NodeId),
    ReadGlobal(u64),
}

/// The dominance-dedup key of a side-effecting assertion.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum AssertKey {
    Rangecheck { value: NodeId, max_bits: usize },
    Lookup { target: LookupKey, args: Vec<NodeId>, flag: NodeId },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum LookupKey {
    Rangecheck(u8),
    DynRangecheck(NodeId),
    Array(NodeId),
    Spread(u8),
}

// INTERNAL FUNCTIONALITY
// ================================================================================================

/// The instruction results channel 1 redirects to their class leaders: the pure scalar shapes
/// plus `ArrayGet`.
///
/// Aggregate constructors and call results are deliberately excluded in this stage.
fn leader_redirect_candidate(instruction: &OpCode) -> Option<ValueId> {
    match instruction {
        OpCode::BinaryArithOp { result, .. }
        | OpCode::Cmp { result, .. }
        | OpCode::MulConst { result, .. }
        | OpCode::Cast { result, .. }
        | OpCode::SExt { result, .. }
        | OpCode::BitRange { result, .. }
        | OpCode::Not { result, .. }
        | OpCode::Select { result, .. }
        | OpCode::ArrayGet { result, .. } => Some(*result),
        _ => None,
    }
}
