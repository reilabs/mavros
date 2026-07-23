//! Taint positions, used as the nodes of the `≥` graph.
//!
//! Witness-ness is tracked per *level* of a type. A [`Position`] names one such level as
//! `(owner, path)`, where `path` descends through `Deref` (a `Ref` pointee) and `Elem` (an
//! `Array`/`Slice` element).
//!
//! `Owner::Top` is the synthetic always-Witness source.
//!
//! Throughout the analysis, types are walked through `Type::peel_witness`: witness-ness is computed
//! by the analysis itself (via the `≥` graph), so the `WitnessOf` annotations the IR already carries
//! (on `WriteWitness` results and witness-derived values) are transparent to the type structure.

use crate::compiler::{
    ssa::{
        ValueId,
        hlssa::{Type, TypeExpr},
    },
    util::ice_non_elided_tuple,
};

// POSITION
// ================================================================================================

/// A single level of taint: `owner` descended along `path`.
#[derive(Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Position {
    pub owner: Owner,
    pub path: Vec<Descent>,
}

impl Position {
    pub fn root(owner: Owner) -> Self {
        Position {
            owner,
            path: Vec::new(),
        }
    }

    pub fn top() -> Self {
        Position::root(Owner::Top)
    }

    /// This position descended one further level.
    pub fn child(&self, d: Descent) -> Position {
        let mut path = self.path.clone();
        path.push(d);
        Position {
            owner: self.owner.clone(),
            path,
        }
    }

    /// This position with `suffix` appended.
    pub fn extend(&self, suffix: &[Descent]) -> Position {
        let mut path = self.path.clone();
        path.extend_from_slice(suffix);
        Position {
            owner: self.owner.clone(),
            path,
        }
    }

    /// Whether this position names memory behind a `Ref` (its path descends through a `Deref`).
    pub fn is_deref_descended(&self) -> bool {
        self.path.contains(&Descent::Deref)
    }
}

// UTILITY TYPES
// ================================================================================================

/// One step down a type's witness shape.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Descent {
    /// Through a `Ref<T>` to its pointee `T`.
    Deref,

    /// Through an `Array<T>`/`Slice<T>` to its element `T`.
    Elem,
}

/// What a [`Position`] belongs to.
///
/// Owners are implicitly scoped to one function: graphs and summaries are built per function, and
/// [`map_formal`](super::builder) translates every callee formal into caller positions at the call
/// site, so positions of different functions never share a container and no `FunctionId` is
/// needed to disambiguate them.
#[derive(Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Owner {
    /// The `i`-th parameter of the function (a formal input).
    Param(usize),

    /// The `i`-th return value of the function (a formal output).
    Return(usize),

    /// An SSA value within the function (internal).
    Value(ValueId),

    /// The function's cfg-witness flag — whether it is called under witness-dependent control flow
    /// (a formal input).
    Cfg,

    /// The `i`-th program global — a program-wide slot in `SSA::global_types`, shared by every
    /// function that reads or initializes it (reads/inits are not per-call formals, so global
    /// witness-ness is decided once, program-wide, rather than threaded through call summaries).
    ///
    /// Globals are init-time constants in this IR today, so `compute_witness_globals` finds none to
    /// be Witness. The position model covers cross-function global *reads*; see the module docs for
    /// the extensions mutable globals would require.
    Global(usize),

    /// The synthetic always-Witness source.
    Top,
}

// UTILITY FUNCTIONS
// ================================================================================================

/// Enumerate, in pre-order, the path to every level of `ty` (the empty path for the top level, then
/// each `Deref`/`Elem` descent).
///
/// Two values of the same type share this path set, so copying taint from value `a` to value `b` is
/// just "for every path `p`, add `b·p ≥ a·p`".
pub fn paths_of_type(ty: &Type) -> Vec<Vec<Descent>> {
    let mut out = Vec::new();
    let mut prefix = Vec::new();
    collect_paths(ty, &mut prefix, &mut out);
    out
}

fn collect_paths(ty: &Type, prefix: &mut Vec<Descent>, out: &mut Vec<Vec<Descent>>) {
    let ty = ty.peel_witness();
    out.push(prefix.clone());
    match &ty.expr {
        TypeExpr::Field
        | TypeExpr::U(_)
        | TypeExpr::I(_)
        | TypeExpr::Function
        | TypeExpr::Blob(..) => {}
        TypeExpr::Array(inner, _) | TypeExpr::Slice { elem: inner, .. } => {
            prefix.push(Descent::Elem);
            collect_paths(inner, prefix, out);
            prefix.pop();
        }
        TypeExpr::Ref(inner) => {
            prefix.push(Descent::Deref);
            collect_paths(inner, prefix, out);
            prefix.pop();
        }
        TypeExpr::WitnessOf(_) => unreachable!("peeled above"),
        TypeExpr::Tuple(_) => ice_non_elided_tuple(),
    }
}
