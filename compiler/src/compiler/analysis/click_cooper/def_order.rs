//! A dominance-consistent total order on a function's value definitions.

use crate::{
    collections::HashMap,
    compiler::{
        analysis::flow_analysis::CFG,
        ssa::{BlockId, Instruction, ValueId, hlssa::HLFunction},
    },
};

// DEF KEY
// ================================================================================================

/// A dominance-consistent total key on a value's definition: `(dominator-preorder index of the
/// defining block, in-block rank, value id)`.
///
/// Constructed so that if `def(w)` dominates `def(v)` then `key(w) < key(v)` — i.e. the order is a
/// linear extension of definition dominance.
pub(crate) type DefKey = (usize, usize, u64);

// DEFINITION ORDER
// ================================================================================================

/// Definition-site bookkeeping for one function, providing a dominance-consistent order on its
/// values.
pub(crate) struct DefOrder<'a> {
    /// Definition site of every value: `(block, rank)`, where a block parameter (φ) ranks before
    /// all instructions (`0`) and instruction `i`'s results rank `i + 1`.
    ///
    /// A value with no recorded site — an entry parameter or an interned constant referenced as an
    /// operand — is treated as defined at `(entry, 0)`, hence available throughout the function.
    def_site: HashMap<ValueId, (BlockId, usize)>,

    /// A dominator-tree pre-order index per block: a linear extension of block dominance in which
    /// an ancestor block always precedes its descendants.
    preorder: HashMap<BlockId, usize>,

    /// The entry block, the fallback definition site for an unrecorded value.
    entry: BlockId,

    /// The function's CFG, for the cross-block dominance check in [`Self::dominates_def`].
    cfg: &'a CFG,
}

impl<'a> DefOrder<'a> {
    /// Record every value's definition site and the block dominator pre-order of `function`.
    pub(crate) fn new(function: &HLFunction, cfg: &'a CFG) -> Self {
        let entry = function.get_entry_id();

        let mut def_site: HashMap<ValueId, (BlockId, usize)> = HashMap::default();
        for (bid, block) in function.get_blocks() {
            for p in block.get_parameter_values() {
                def_site.insert(*p, (*bid, 0));
            }
            for (index, instr) in block.get_instructions().enumerate() {
                for r in instr.get_results() {
                    def_site.insert(*r, (*bid, index + 1));
                }
            }
        }

        let preorder: HashMap<BlockId, usize> = cfg
            .get_domination_pre_order()
            .enumerate()
            .map(|(i, b)| (b, i))
            .collect();

        Self {
            def_site,
            preorder,
            entry,
            cfg,
        }
    }

    /// `(block, rank)` of `v`'s definition, falling back to `(entry, 0)` for an unrecorded value.
    fn site(&self, v: ValueId) -> (BlockId, usize) {
        self.def_site.get(&v).copied().unwrap_or((self.entry, 0))
    }

    /// The block `v` is defined in, with the [`Self::site`] entry-block fallback for an unrecorded
    /// value (an entry parameter or an interned constant, available throughout the function).
    pub(crate) fn def_block(&self, v: ValueId) -> BlockId {
        self.site(v).0
    }

    /// The redirect-validity threshold of `v` as a leader queried in `block`.
    ///
    /// This is `0` when `v` is defined outside `block` or bound at its entry (available at every
    /// index), else its defining instruction's rank (`d + 1` for instruction `d`, so
    /// `threshold <= use_index` iff the use is strictly after the definition).
    ///
    /// The out-of-block `0` is only a valid threshold under the caller's invariant that `v`'s
    /// defining block _dominates_ `block` (as every union-leader class member's does). Checking
    /// dominance is the **responsibility of the caller**.
    pub(crate) fn redirect_threshold_in(&self, v: ValueId, block: BlockId) -> usize {
        let (b, rank) = self.site(v);
        if b == block { rank } else { 0 }
    }

    /// The dominance-consistent total [`DefKey`] of `v` (the defining block's dominator-preorder
    /// index, the in-block rank, and the value id as a final tie-break).
    pub(crate) fn key(&self, v: ValueId) -> DefKey {
        let (b, rank) = self.site(v);
        (
            self.preorder.get(&b).copied().unwrap_or(usize::MAX),
            rank,
            v.0,
        )
    }

    /// `true` if `def(w)` dominates `def(v)`.
    ///
    /// This checks for strict block dominance across blocks, or an earlier rank within a block
    /// (with a value-id tie-break for two parameters of the same block, both available at block
    /// entry). [`CFG::dominates`] is reflexive, so it is only consulted across blocks.
    pub(crate) fn dominates_def(&self, w: ValueId, v: ValueId) -> bool {
        let (bw, rw) = self.site(w);
        let (bv, rv) = self.site(v);
        if bw == bv {
            rw < rv || (rw == rv && w.0 < v.0)
        } else {
            self.cfg.dominates(bw, bv)
        }
    }
}
