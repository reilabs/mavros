//! The bounded call-string [`Context`]: the context-sensitivity coordinate shared by
//! interprocedural analyses.
//!
//! A context is a `k`-limited string of call sites so the context space stays finite with recursion
//! folding onto its recursion head. The empty context is the polymorphic context, under which
//! `main` is specialized.

use crate::compiler::ssa::{FunctionId, ValueId};

// TYPES
// ================================================================================================

/// A call site, identified by the caller function and a stable per-call id within its body: the
/// `Call`'s first result value, or its first argument when the call has no results.
pub type CallSite = (FunctionId, ValueId);

// CONTEXT
// ================================================================================================

/// A bounded call-string identifying the context a function (or abstract object) was specialized
/// in.
#[derive(Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord, Default)]
pub struct Context(Vec<CallSite>);

impl Context {
    /// The empty (polymorphic / summary) context.
    pub fn empty() -> Self {
        Context(Vec::new())
    }

    /// Whether this is the empty context.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// This context extended by `site`, truncated to the most-recent `k` sites (k-CFA).
    ///
    /// Passing `k == 0` collapses every context to the empty one ([`Self::empty`]).
    pub fn push(&self, site: CallSite, k: usize) -> Self {
        if k == 0 {
            return Context::empty();
        }
        let mut sites = self.0.clone();
        sites.push(site);
        if sites.len() > k {
            sites.drain(0..sites.len() - k);
        }
        Context(sites)
    }
}
