//! The abstract domain of the points-to analysis: abstract objects, the cell-refined type-descent
//! skeleton, and the solver node universe.
//!
//! This is the inclusion-based (Andersen) refinement of a Steensgaard-style unification domain.
//! Where Steensgaard would collapse every alias of an object into a single equivalence class, here
//! each abstract object keeps its own per-cell points-to sets, so distinct objects merged via a
//! `Select`, phi, or array stay distinguishable.
//!
//! The skeleton mirrors `position::{Descent, Position}` but refines the array step with a [`Cell`]
//! coordinate so that statically-constant array indices can be tracked separately (the enabler for
//! array-cell splitting).

use crate::compiler::ssa::{FunctionId, ValueId};

// DESCENT AND PATHS
// ================================================================================================

/// One step down a type's pointer/aggregate shape.
#[derive(Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Descent {
    /// Through a `Ref<T>` to its pointee `T`.
    Deref,

    /// Through an `Array<T>`/`Slice<T>` to one element, named by [`Cell`].
    Elem(Cell),
}

/// Which element(s) of an array/slice a `Descent::Elem` names.
///
/// Constant indices each get their own [`Cell::Index`] field — the precision that lets the
/// splitting transform peel a single cell into its own allocation/value. Every dynamic
/// (non-constant) index collapses to the single sound [`Cell::AllElems`] field, which a
/// constant-cell *read* must also consult and a dynamic-index *write* pollutes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Cell {
    /// A statically-known constant index — its own splittable field.
    Index(usize),

    /// The collapsed "all elements" field for dynamic / non-constant indices, and the join target
    /// for whole-array copies.
    AllElems,
}

/// A path from a node's root down through `Deref`/`Elem` steps to one level of its type.
pub type Path = Vec<Descent>;

// CALL CONTEXT
// ================================================================================================

/// The bounded call-string identifying the context an abstract object was created in: context
/// sensitivity keeps two call sites of the same helper from conflating their local allocations.
pub use crate::compiler::analysis::call_string::{CallSite, Context};

// ABSTRACT OBJECTS
// ================================================================================================

/// An abstract heap object — the unit a points-to set ranges over.
///
/// `Alloc` objects are the real allocation sites; `Placeholder` objects make per-function summaries
/// parametric over caller-supplied memory; `Global` covers program globals; and `External` is the
/// single opaque sink for everything the analysis cannot resolve (the points-to analog of the
/// witness-taint module's `Top`).
#[derive(Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum AbstractObject {
    /// The object created at an `Alloc { result, .. }` site, qualified by its creation [`Context`].
    Alloc(FunctionId, ValueId, Context),

    /// Caller-owned memory reachable through ref-typed parameter `i` of a function, at pointee
    /// `Path`.
    ///
    /// A *symbolic* input the function's summary is parametric over: it is substituted with the
    /// caller's actual argument objects when the summary is instantiated at a call site, so it is
    /// not inherently escaped — whether it escapes is decided per call by the caller's objects.
    Placeholder(FunctionId, usize, Path),

    /// Memory reachable through program global slot `g` at `Path`.
    ///
    /// Conservatively escaped today (globals are init-time constants — there is no `WriteGlobal`).
    Global(usize, Path),

    /// The opaque "anything external" object: unknown/unconstrained callees and dynamic memory.
    ///
    /// Always escaped; any points-to set containing it is may-alias-anything.
    External,
}

impl AbstractObject {
    /// Whether this object is, by construction, always escaped (so never splittable).
    ///
    /// `Placeholder` is **not** here: it is a symbolic summary input, escaped only when the
    /// caller's substituted objects escape (decided per call via the summary's leak set).
    pub fn is_inherently_escaped(&self) -> bool {
        matches!(self, AbstractObject::External | AbstractObject::Global(..))
    }

    /// Whether this object denotes *unknown* memory with no analyzable contents — `External` and
    /// any program `Global` (init-time, program-wide).
    ///
    /// Loading through an opaque object may yield anything (so it yields `External`), and anything
    /// stored through one is published to the world (so it escapes). Distinct from
    /// [`Self::is_inherently_escaped`] (which answers "splittable?"): this answers "what do
    /// load/store *through* it do?", and a future global-contents pre-pass could make `Global`
    /// non-opaque while it stays escaped.
    pub fn is_opaque(&self) -> bool {
        matches!(self, AbstractObject::External | AbstractObject::Global(..))
    }
}

// SOLVER NODES
// ================================================================================================

/// The owner of a value-level node — an SSA value or a function formal.
///
/// A reduced cousin of `position::Owner`: the points-to analysis has no `Cfg`/`Top` owners (control
/// flow carries no pointers; `Top` is replaced by [`AbstractObject::External`]).
#[derive(Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Owner {
    /// An SSA value within the function.
    Value(ValueId),

    /// The `i`-th parameter of the function (a formal input).
    Param(usize),

    /// The `i`-th return value of the function (a formal output).
    Return(usize),
}

/// A node in the points-to constraint graph, with each carrying a points-to set.
///
/// `Val` names a level of an SSA value or formal; `Obj` names a cell *inside* an abstract object
/// (its pointee, an array element, a nested-ref slot). Keeping object cells as first-class nodes —
/// rather than collapsing them into the owning value, as unification does — is the whole source of
/// the precision gain.
#[derive(Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum NodeKey {
    /// A level of a value/formal: `owner` descended along `path`.
    Val(Owner, Path),

    /// A cell of an abstract object: `object` dereferenced and descended along `path`.
    Obj(AbstractObject, Path),

    /// A synthetic leaf value whose points-to set is exactly `{object}` — "a reference to
    /// `object`".
    ///
    /// Used as a store *source* when a callee summary's precise arg-out writes a concrete object
    /// into caller memory. Never descended.
    ObjRef(AbstractObject),
}

impl NodeKey {
    /// The root (top-level) node of SSA value `v`.
    pub fn value(v: ValueId) -> Self {
        NodeKey::Val(Owner::Value(v), Vec::new())
    }

    /// This node descended along `suffix`.
    pub fn extend(&self, suffix: &[Descent]) -> Self {
        match self {
            NodeKey::Val(owner, path) => NodeKey::Val(owner.clone(), join(path, suffix)),
            NodeKey::Obj(obj, path) => NodeKey::Obj(obj.clone(), join(path, suffix)),
            NodeKey::ObjRef(_) => unreachable!("ObjRef nodes are leaves and are never descended"),
        }
    }
}

fn join(prefix: &[Descent], suffix: &[Descent]) -> Path {
    let mut path = prefix.to_vec();
    path.extend_from_slice(suffix);
    path
}
