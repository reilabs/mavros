use std::fmt::{Debug, Display, Formatter};

use mavros_artifacts::FieldConfig;

use crate::compiler::ssa::SSAType;

// FIELD-ASSUMPTION: L6-int-representation
//
// These bound the integer _type system_ (the widest int a Noir program may use); they are
// field-independent and stay fixed regardless of field size. Integers wider than can fit into the
// field natively must still be supported. A value whose type range is >= p cannot be held
// natively in one field cell, so must be carried as multi-cell (limb-based) values end-to-end.
pub const MAX_SUPPORTED_UNSIGNED_BITS: usize = 128;

/// The widest integer a _signed_ operation may act on.
///
/// This is a bound on operations, not on types: [`TypeExpr::Int`] is just "an `n`-bit integer" and
/// tops out at [`MAX_SUPPORTED_UNSIGNED_BITS`] like any other. What is unsupported is asking a
/// signed opcode to read a pattern wider than this, because the signed lowerings and the VM's
/// `div_s64`/`lt_s64` are 64-bit. Enforce it with [`assert_signed_op_width`] at the point the
/// signed operation is chosen, never by inspecting a type.
pub const MAX_SUPPORTED_SIGNED_BITS: usize = 64;

/// Reject a signed _operation_ on a pattern wider than [`MAX_SUPPORTED_SIGNED_BITS`].
///
/// `what` names the operation for the panic, e.g. `"division"`. Call this from the arm that has
/// already decided the operation is signed — the width alone is never the problem, and an
/// `int128` that no signed opcode touches is perfectly legal.
pub fn assert_signed_op_width(bits: usize, what: &str) {
    assert!(
        bits <= MAX_SUPPORTED_SIGNED_BITS,
        "signed integers wider than i{MAX_SUPPORTED_SIGNED_BITS} are unsupported: \
         {what} on a {bits}-bit value"
    );
}

/// A type expression.
///
/// Integers carry only a width: [`TypeExpr::Int`] is "an `n`-bit integer", _not_ "a signed `n`-bit
/// integer". Signedness is a property of the operation ([`BinaryArithOpKind`], [`CmpKind`]), which
/// is where every level below HLSSA already keeps it — LLSSA's `Type::Int` with `UDiv`/`SDiv`, the
/// VM's `div_u64`/`div_s64`, LLVM's `build_int_signed_div`. Nothing may recover a sign from a type.
///
/// [`BinaryArithOpKind`]: super::BinaryArithOpKind
/// [`CmpKind`]: super::CmpKind
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeExpr {
    Field,
    Int(usize),
    WitnessOf(Box<Type>),
    Array(Box<Type>, usize),
    Slice(Box<Type>),
    Ref(Box<Type>),
    Tuple(Vec<Type>),
    Function,
    Blob(Box<Type>, usize),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Type {
    pub expr: TypeExpr,
}

impl Display for Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        match &self.expr {
            TypeExpr::Field => write!(f, "Field"),
            TypeExpr::Int(size) => write!(f, "int{}", size),
            TypeExpr::WitnessOf(inner) => write!(f, "WitnessOf({})", inner),
            TypeExpr::Array(inner, size) => write!(f, "Array<{}, {}>", inner, size),
            TypeExpr::Slice(inner) => write!(f, "Slice<{}>", inner),
            TypeExpr::Ref(inner) => write!(f, "Ref<{}>", inner),
            TypeExpr::Tuple(elements) => write!(
                f,
                "Tuple<{}>",
                elements
                    .iter()
                    .map(|e| format!("{}", e))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            TypeExpr::Function => write!(f, "Function"),
            TypeExpr::Blob(inner, len) => write!(f, "Blob<{}; {}>", inner, len),
        }
    }
}

impl Type {
    // --- Constructors ---

    pub fn field() -> Self {
        Type {
            expr: TypeExpr::Field,
        }
    }

    /// An `n`-bit integer. Signedness belongs to the operation, not here — see [`TypeExpr`].
    pub fn int(size: usize) -> Self {
        Type {
            expr: TypeExpr::Int(size),
        }
    }

    pub fn bool() -> Self {
        Type::int(1)
    }

    pub fn int32() -> Self {
        Type::int(32)
    }

    pub fn function() -> Self {
        Type {
            expr: TypeExpr::Function,
        }
    }

    pub fn blob(elem: Type, len: usize) -> Self {
        Type {
            expr: TypeExpr::Blob(Box::new(elem), len),
        }
    }

    pub fn array_of(self, size: usize) -> Self {
        Type {
            expr: TypeExpr::Array(Box::new(self), size),
        }
    }

    pub fn slice_of(self) -> Self {
        Type {
            expr: TypeExpr::Slice(Box::new(self)),
        }
    }

    pub fn ref_of(self) -> Self {
        Type {
            expr: TypeExpr::Ref(Box::new(self)),
        }
    }

    pub fn tuple_of(types: Vec<Self>) -> Self {
        Type {
            expr: TypeExpr::Tuple(types),
        }
    }

    /// Construct a WitnessOf type. Double-wrapping is a bug.
    pub fn witness_of(inner: Type) -> Self {
        assert!(
            !matches!(inner.expr, TypeExpr::WitnessOf(_)),
            "ICE: attempted to construct WitnessOf(WitnessOf(...)). \
             Double witness wrapping indicates a logic error. Inner type: {:?}",
            inner.expr
        );
        Type {
            expr: TypeExpr::WitnessOf(Box::new(inner)),
        }
    }

    /// Wrap `inner` in WitnessOf unless it already is witnessed — the idempotent variant of
    /// [`Self::witness_of`], used where a level _inherits_ witness-ness from its container and an
    /// already-witnessed inner must not be wrapped twice (witness-of-witness collapses).
    pub fn witness_of_collapsed(inner: Type) -> Self {
        if inner.is_witness_of() {
            inner
        } else {
            Type::witness_of(inner)
        }
    }

    // --- Predicates ---

    pub fn is_numeric(&self) -> bool {
        match &self.expr {
            TypeExpr::Int(_) | TypeExpr::Field => true,
            TypeExpr::WitnessOf(inner) => inner.is_numeric(),
            _ => false,
        }
    }

    pub fn is_field(&self) -> bool {
        matches!(self.expr, TypeExpr::Field)
    }

    pub fn is_array(&self) -> bool {
        matches!(self.expr, TypeExpr::Array(_, _))
    }

    pub fn is_slice(&self) -> bool {
        matches!(self.expr, TypeExpr::Slice(_))
    }

    pub fn is_array_or_slice(&self) -> bool {
        matches!(self.expr, TypeExpr::Array(_, _) | TypeExpr::Slice(_))
    }

    pub fn is_witness_of(&self) -> bool {
        matches!(self.expr, TypeExpr::WitnessOf(_))
    }

    pub fn is_integer(&self) -> bool {
        matches!(self.expr, TypeExpr::Int(_))
    }

    pub fn is_int32(&self) -> bool {
        matches!(self.expr, TypeExpr::Int(32))
    }

    pub fn is_heap_allocated(&self) -> bool {
        matches!(
            self.expr,
            TypeExpr::WitnessOf(_)
                | TypeExpr::Array(_, _)
                | TypeExpr::Slice(_)
                | TypeExpr::Ref(_)
                | TypeExpr::Tuple(_)
        )
    }

    pub fn is_function(&self) -> bool {
        matches!(self.expr, TypeExpr::Function)
    }

    pub fn is_blob(&self) -> bool {
        matches!(self.expr, TypeExpr::Blob(..))
    }

    pub fn has_eq(&self) -> bool {
        matches!(self.expr, TypeExpr::Field | TypeExpr::Int(_))
    }

    pub fn is_ref(&self) -> bool {
        matches!(self.expr, TypeExpr::Ref(_))
    }

    pub fn is_tuple(&self) -> bool {
        matches!(self.expr, TypeExpr::Tuple(_))
    }

    // --- Accessors ---

    /// Element type of an array/slice/blob. An element read out of a witnessed
    /// container is itself witnessed; an already-witnessed element is never
    /// wrapped twice (witness-of-witness collapses to witness).
    pub fn get_array_element(&self) -> Self {
        match &self.expr {
            TypeExpr::Array(inner, _) => *inner.clone(),
            TypeExpr::Slice(inner) => *inner.clone(),
            TypeExpr::Blob(inner, _) => *inner.clone(),
            TypeExpr::WitnessOf(inner) => Type::witness_of_collapsed(inner.get_array_element()),
            _ => panic!("Type is not an array: {}", self),
        }
    }

    pub fn get_pointed(&self) -> Self {
        match &self.expr {
            TypeExpr::Ref(inner) => *inner.clone(),
            _ => panic!("Type is not a reference: {}", self),
        }
    }

    /// Member type of a tuple. A member of a witnessed tuple is itself
    /// witnessed; an already-witnessed member is never wrapped twice.
    pub fn get_tuple_element(&self, index: usize) -> Self {
        match &self.expr {
            TypeExpr::Tuple(elements) => elements[index].clone(),
            TypeExpr::WitnessOf(inner) => {
                Type::witness_of_collapsed(inner.get_tuple_element(index))
            }
            _ => panic!("Type is not a tuple: {}", self),
        }
    }

    /// All member types of a tuple, witnessed if the tuple itself is (see
    /// [`Self::get_tuple_element`]).
    pub fn get_tuple_elements(&self) -> Vec<Self> {
        match &self.expr {
            TypeExpr::Tuple(elements) => elements.clone(),
            TypeExpr::WitnessOf(inner) => inner
                .get_tuple_elements()
                .into_iter()
                .map(Type::witness_of_collapsed)
                .collect(),
            _ => panic!("Type is not a tuple: {}", self),
        }
    }

    pub fn get_refered(&self) -> &Self {
        match &self.expr {
            TypeExpr::Ref(inner) => inner.as_ref(),
            _ => panic!("Type is not a reference: {}", self),
        }
    }

    /// The width of a numeric type in bits. A field element is as wide as the configured field's
    /// modulus, so this needs the program's [`FieldConfig`] (reached from an `SSA` or a builder as
    /// `ssa.field()` / `b.field()`).
    pub fn get_bit_size(&self, field: FieldConfig) -> usize {
        match &self.expr {
            TypeExpr::Int(size) => *size,
            TypeExpr::Field => field.field_bit_size() as usize,
            TypeExpr::WitnessOf(inner) => inner.get_bit_size(field),
            _ => panic!("Type is not numeric: {}", self),
        }
    }

    // --- WitnessOf helpers ---

    /// Returns the inner type if this is WitnessOf, panics otherwise.
    pub fn unwrap_witness_of(&self) -> &Type {
        match &self.expr {
            TypeExpr::WitnessOf(inner) => inner,
            _ => panic!("Type is not WitnessOf: {}", self),
        }
    }

    /// Returns the inner type if this is WitnessOf, None otherwise.
    pub fn try_unwrap_witness_of(&self) -> Option<&Type> {
        match &self.expr {
            TypeExpr::WitnessOf(inner) => Some(inner),
            _ => None,
        }
    }

    /// Strip one level of WitnessOf. Returns inner if WitnessOf, self otherwise.
    pub fn strip_witness(&self) -> Self {
        match &self.expr {
            TypeExpr::WitnessOf(inner) => *inner.clone(),
            _ => self.clone(),
        }
    }

    /// Strip every top-level WitnessOf wrapper, by reference — the borrow-returning sibling of
    /// [`Self::strip_witness`] for callers that only inspect the underlying structure.
    ///
    /// With the no-nested-wrapper invariant enforced by [`Self::witness_of`] the loop runs at
    /// most once, but stacked wrappers are tolerated anyway.
    pub fn peel_witness(&self) -> &Type {
        let mut t = self;
        while let TypeExpr::WitnessOf(inner) = &t.expr {
            t = inner;
        }
        t
    }

    /// Recursively strip all WitnessOf wrappers at every level.
    pub fn strip_all_witness(&self) -> Self {
        match &self.expr {
            TypeExpr::WitnessOf(inner) => inner.strip_all_witness(),
            TypeExpr::Array(inner, size) => Type {
                expr: TypeExpr::Array(Box::new(inner.strip_all_witness()), *size),
            },
            TypeExpr::Slice(inner) => Type {
                expr: TypeExpr::Slice(Box::new(inner.strip_all_witness())),
            },
            TypeExpr::Ref(inner) => Type {
                expr: TypeExpr::Ref(Box::new(inner.strip_all_witness())),
            },
            TypeExpr::Tuple(elements) => Type {
                expr: TypeExpr::Tuple(elements.iter().map(|e| e.strip_all_witness()).collect()),
            },
            _ => self.clone(),
        }
    }

    // --- Subtyping ---

    /// Check if `self <: other` under the WitnessOf subtyping rules.
    ///
    /// The subtyping relation is:
    /// - X <: X  (reflexive)
    /// - X <: WitnessOf(Y) if X <: Y  (WitnessOf is a supertype)
    /// - WitnessOf(X) <: WitnessOf(Y) if X <: Y  (covariant)
    /// - Array/Slice/Tuple are covariant in their element types
    /// - Ref is invariant
    pub fn is_subtype_of(&self, other: &Type) -> bool {
        match (&self.expr, &other.expr) {
            // WitnessOf on both sides: covariant
            (TypeExpr::WitnessOf(inner_a), TypeExpr::WitnessOf(inner_b)) => {
                inner_a.is_subtype_of(inner_b)
            }
            // X <: WitnessOf(Y) iff X <: Y
            (_, TypeExpr::WitnessOf(inner_b)) => self.is_subtype_of(inner_b),
            // WitnessOf(X) <: Y where Y is NOT WitnessOf — impossible
            (TypeExpr::WitnessOf(_), _) => false,
            // Structural (neither is WitnessOf)
            (TypeExpr::Field, TypeExpr::Field) => true,
            (TypeExpr::Int(n), TypeExpr::Int(m)) => n == m,
            (TypeExpr::Array(x, n), TypeExpr::Array(y, m)) => n == m && x.is_subtype_of(y),
            (TypeExpr::Slice(x), TypeExpr::Slice(y)) => x.is_subtype_of(y),
            (TypeExpr::Tuple(xs), TypeExpr::Tuple(ys)) => {
                xs.len() == ys.len() && xs.iter().zip(ys.iter()).all(|(x, y)| x.is_subtype_of(y))
            }
            (TypeExpr::Ref(x), TypeExpr::Ref(y)) => x == y, // invariant
            (TypeExpr::Function, TypeExpr::Function) => true,
            (TypeExpr::Blob(x, n), TypeExpr::Blob(y, m)) => n == m && x == y,
            _ => false,
        }
    }

    /// Returns true if converting from `self` to `target` requires inserting
    /// WitnessOf cast(s). This is the case when `self` is a strict subtype of
    /// `target` (same structure but `target` has WitnessOf where `self` doesn't).
    pub fn needs_witness_cast(&self, target: &Type) -> bool {
        self != target && self.is_subtype_of(target)
    }

    // --- Join (least upper bound) ---

    /// Compute the least upper bound (join) of two types in the WitnessOf lattice.
    ///
    /// Used for merge points (phi nodes) where two branches may produce
    /// different witness-ness levels.
    pub fn join(a: &Type, b: &Type) -> Type {
        match (&a.expr, &b.expr) {
            // WitnessOf cases: unwrap and re-wrap
            (TypeExpr::WitnessOf(inner_a), TypeExpr::WitnessOf(inner_b)) => {
                Type::witness_of(Type::join(inner_a, inner_b))
            }
            (TypeExpr::WitnessOf(inner_a), _) => Type::witness_of(Type::join(inner_a, b)),
            (_, TypeExpr::WitnessOf(inner_b)) => Type::witness_of(Type::join(a, inner_b)),
            // Structural (neither is WitnessOf)
            (TypeExpr::Field, TypeExpr::Field) => Type::field(),
            (TypeExpr::Int(n), TypeExpr::Int(m)) => {
                assert_eq!(n, m, "Cannot join int({}) and int({})", n, m);
                Type::int(*n)
            }
            (TypeExpr::Array(x, n), TypeExpr::Array(y, m)) => {
                assert_eq!(
                    n, m,
                    "Cannot join arrays of different sizes: {} vs {}",
                    n, m
                );
                Type::join(x, y).array_of(*n)
            }
            (TypeExpr::Slice(x), TypeExpr::Slice(y)) => Type::join(x, y).slice_of(),
            (TypeExpr::Tuple(xs), TypeExpr::Tuple(ys)) => {
                assert_eq!(
                    xs.len(),
                    ys.len(),
                    "Cannot join tuples of different lengths"
                );
                Type::tuple_of(
                    xs.iter()
                        .zip(ys.iter())
                        .map(|(x, y)| Type::join(x, y))
                        .collect(),
                )
            }
            (TypeExpr::Ref(x), TypeExpr::Ref(y)) => Type::join(x, y).ref_of(),
            (TypeExpr::Function, TypeExpr::Function) => Type::function(),
            (TypeExpr::Blob(x, n), TypeExpr::Blob(y, m)) => {
                assert_eq!(n, m, "Cannot join Blob({}) and Blob({})", n, m);
                assert_eq!(x, y, "Cannot join blobs with different element types");
                Type::blob(*x.clone(), *n)
            }
            _ => panic!("Cannot join types {} and {}", a, b),
        }
    }

    // --- Comparison ---

    pub fn is_ref_of(&self, other: &Self) -> bool {
        match &self.expr {
            TypeExpr::Ref(inner) => inner.as_ref() == other,
            _ => false,
        }
    }

    // --- Arithmetic result type ---

    pub fn get_arithmetic_result_type(&self, other: &Self) -> Self {
        match (&self.expr, &other.expr) {
            // Both WitnessOf: unwrap both, compute, re-wrap once
            (TypeExpr::WitnessOf(a), TypeExpr::WitnessOf(b)) => {
                Type::witness_of(a.get_arithmetic_result_type(b))
            }
            // One side WitnessOf: unwrap it, compute, re-wrap
            (TypeExpr::WitnessOf(inner), _) => {
                Type::witness_of(inner.get_arithmetic_result_type(other))
            }
            (_, TypeExpr::WitnessOf(inner)) => {
                Type::witness_of(self.get_arithmetic_result_type(inner))
            }
            (TypeExpr::Field, _) | (_, TypeExpr::Field) => Type::field(),
            (TypeExpr::Int(size1), TypeExpr::Int(size2)) => Type::int(*size1.max(size2)),
            _ => panic!("Cannot perform arithmetic on types {} and {}", self, other),
        }
    }

    /// The unified type of a `Select`'s two alternatives.
    ///
    /// Numeric alternatives unify by the arithmetic rule. The two container kinds `Select` also
    /// ranges over — but on which no arithmetic is defined — unify elementwise via
    /// [`Self::join`] instead:
    ///
    /// - **Slices:** `untaint_control_flow`'s `emit_merge_select` merges witness-length physical
    ///   slices with a `Select`.
    /// - **Tuples:** `purify_witness_slices`'s `Select` arm rewrites both alternatives of a
    ///   witness-length slice select into `(physical, log_len, start)` tuples. That form is
    ///   transient — `ElideTuples` runs next and splits the select per component — but `TypeInfo`
    ///   is recomputed in between (the pass preserves no analyses, and `ElideTuples` needs types),
    ///   so it must type.
    ///
    /// `join` still asserts on arity/shape mismatch, and everything else is still refused, so this
    /// stays a real check rather than a widened [`Self::get_arithmetic_result_type`].
    pub fn get_select_result_type(&self, other: &Self) -> Self {
        match (&self.expr, &other.expr) {
            (TypeExpr::Slice(_), TypeExpr::Slice(_)) | (TypeExpr::Tuple(_), TypeExpr::Tuple(_)) => {
                Type::join(self, other)
            }
            _ => self.get_arithmetic_result_type(other),
        }
    }

    // --- Misc ---

    pub fn contains_ptrs(&self) -> bool {
        match &self.expr {
            TypeExpr::Ref(_) => true,
            TypeExpr::Array(inner, _) => inner.contains_ptrs(),
            TypeExpr::Slice(inner) => inner.contains_ptrs(),
            TypeExpr::WitnessOf(inner) => inner.contains_ptrs(),
            TypeExpr::Field => false,
            TypeExpr::Int(_) => false,
            TypeExpr::Function => false,
            TypeExpr::Blob(inner, _) => inner.contains_ptrs(),
            TypeExpr::Tuple(elements) => elements.iter().any(|e| e.contains_ptrs()),
        }
    }

    pub fn calculate_type_size(&self) -> usize {
        match &self.expr {
            TypeExpr::Field => 1,
            TypeExpr::Array(_inner, _size) => 1,
            TypeExpr::Tuple(inner_types) => {
                inner_types.iter().map(|t| t.calculate_type_size()).sum()
            }
            TypeExpr::Function => 1,
            // Blobs are by-value sequences, not pointers to heap data.
            TypeExpr::Blob(inner, n) => inner.calculate_type_size() * n,
            TypeExpr::Int(_) => 1,
            TypeExpr::WitnessOf(_) => 1, // pointer-sized (witness tape reference)
            _ => panic!("Cannot currently calculate size for type {}", self),
        }
    }
}

impl SSAType for Type {}

#[cfg(test)]
mod tests {
    use super::*;

    // --- the signed width bound ---

    #[test]
    fn the_signed_width_bound_is_about_the_operation_not_the_type() {
        // An `int128` is a perfectly ordinary type: only asking a _signed_ operation to read one
        // is unsupported. Nothing here may inspect a type to decide that -- a type is a width.
        assert_signed_op_width(64, "division");
        assert_signed_op_width(1, "division");
        assert_eq!(Type::int(128).get_bit_size(FieldConfig::bn254()), 128);
    }

    #[test]
    #[should_panic(expected = "signed integers wider than i64 are unsupported")]
    fn a_signed_operation_wider_than_i64_is_rejected() {
        assert_signed_op_width(128, "division");
    }

    // --- get_bit_size ---

    #[test]
    fn field_bit_size_comes_from_the_config() {
        let field = FieldConfig::bn254();
        assert_eq!(
            Type::field().get_bit_size(field),
            field.field_bit_size() as usize
        );
        // bn254's modulus is 254 bits wide; pinned because the corpus depends on this width.
        assert_eq!(Type::field().get_bit_size(field), 254);
        // A witnessed field is exactly as wide as the field it witnesses.
        assert_eq!(
            Type::witness_of(Type::field()).get_bit_size(field),
            Type::field().get_bit_size(field)
        );
        // Integer widths are field-independent.
        assert_eq!(Type::int(32).get_bit_size(field), 32);
        assert_eq!(Type::int(64).get_bit_size(field), 64);
    }

    // --- is_subtype_of ---

    #[test]
    fn subtype_reflexive() {
        assert!(Type::field().is_subtype_of(&Type::field()));
        assert!(Type::int(32).is_subtype_of(&Type::int(32)));
        assert!(Type::function().is_subtype_of(&Type::function()));
        let wf = Type::witness_of(Type::field());
        assert!(wf.is_subtype_of(&wf));
    }

    #[test]
    fn subtype_field_witness_of_field() {
        let f = Type::field();
        let wf = Type::witness_of(Type::field());
        assert!(f.is_subtype_of(&wf));
        assert!(!wf.is_subtype_of(&f));
    }

    #[test]
    fn subtype_u32_witness_of_u32() {
        let u = Type::int(32);
        let wu = Type::witness_of(Type::int(32));
        assert!(u.is_subtype_of(&wu));
        assert!(!wu.is_subtype_of(&u));
    }

    #[test]
    fn subtype_array_covariant() {
        let arr_f = Type::field().array_of(5);
        let arr_wf = Type::witness_of(Type::field()).array_of(5);
        assert!(arr_f.is_subtype_of(&arr_wf));
        assert!(!arr_wf.is_subtype_of(&arr_f));
    }

    #[test]
    fn subtype_array_into_witness_of_array() {
        let arr_f = Type::field().array_of(5);
        let w_arr_f = Type::witness_of(Type::field().array_of(5));
        assert!(arr_f.is_subtype_of(&w_arr_f));
        assert!(!w_arr_f.is_subtype_of(&arr_f));
    }

    #[test]
    fn subtype_incomparable_array_types() {
        // Array<WitnessOf(Field), 5> and WitnessOf(Array<Field, 5>) are incomparable
        let arr_wf = Type::witness_of(Type::field()).array_of(5);
        let w_arr_f = Type::witness_of(Type::field().array_of(5));
        assert!(!arr_wf.is_subtype_of(&w_arr_f));
        assert!(!w_arr_f.is_subtype_of(&arr_wf));
    }

    #[test]
    fn subtype_witness_of_array_covariant() {
        // WitnessOf(Array<Field,5>) <: WitnessOf(Array<WitnessOf(Field),5>)
        let w_arr_f = Type::witness_of(Type::field().array_of(5));
        let w_arr_wf = Type::witness_of(Type::witness_of(Type::field()).array_of(5));
        assert!(w_arr_f.is_subtype_of(&w_arr_wf));
        assert!(!w_arr_wf.is_subtype_of(&w_arr_f));
    }

    #[test]
    fn subtype_tuple_covariant() {
        let t1 = Type::tuple_of(vec![Type::field(), Type::int(32)]);
        let t2 = Type::tuple_of(vec![Type::witness_of(Type::field()), Type::int(32)]);
        assert!(t1.is_subtype_of(&t2));
        assert!(!t2.is_subtype_of(&t1));
    }

    #[test]
    fn subtype_ref_invariant() {
        let r1 = Type::field().ref_of();
        let r2 = Type::witness_of(Type::field()).ref_of();
        assert!(!r1.is_subtype_of(&r2));
        assert!(!r2.is_subtype_of(&r1));
    }

    #[test]
    fn subtype_different_base_types() {
        assert!(!Type::field().is_subtype_of(&Type::int(32)));
        assert!(!Type::int(32).is_subtype_of(&Type::field()));
        assert!(!Type::int(8).is_subtype_of(&Type::int(32)));
    }

    // --- needs_witness_cast ---

    #[test]
    fn needs_cast_same_type() {
        assert!(!Type::field().needs_witness_cast(&Type::field()));
        let wf = Type::witness_of(Type::field());
        assert!(!wf.needs_witness_cast(&wf));
    }

    #[test]
    fn needs_cast_field_to_witness() {
        assert!(Type::field().needs_witness_cast(&Type::witness_of(Type::field())));
    }

    #[test]
    fn needs_cast_array_element_widening() {
        let arr_f = Type::field().array_of(3);
        let arr_wf = Type::witness_of(Type::field()).array_of(3);
        assert!(arr_f.needs_witness_cast(&arr_wf));
    }

    #[test]
    fn needs_cast_incompatible() {
        assert!(!Type::field().needs_witness_cast(&Type::int(32)));
    }

    // --- join ---

    #[test]
    fn join_same_types() {
        assert_eq!(Type::join(&Type::field(), &Type::field()), Type::field());
        assert_eq!(Type::join(&Type::int(32), &Type::int(32)), Type::int(32));
    }

    #[test]
    fn join_field_witness_field() {
        let f = Type::field();
        let wf = Type::witness_of(Type::field());
        assert_eq!(Type::join(&f, &wf), wf);
        assert_eq!(Type::join(&wf, &f), wf);
    }

    #[test]
    fn join_witness_witness() {
        let wf = Type::witness_of(Type::field());
        assert_eq!(Type::join(&wf, &wf), wf);
    }

    #[test]
    fn join_array_covariant() {
        let arr_f = Type::field().array_of(5);
        let arr_wf = Type::witness_of(Type::field()).array_of(5);
        assert_eq!(Type::join(&arr_f, &arr_wf), arr_wf);
        assert_eq!(Type::join(&arr_wf, &arr_f), arr_wf);
    }

    #[test]
    fn join_incomparable_array_types() {
        // join(Array<WitnessOf(Field), 5>, WitnessOf(Array<Field, 5>))
        //   = WitnessOf(Array<WitnessOf(Field), 5>)
        let arr_wf = Type::witness_of(Type::field()).array_of(5);
        let w_arr_f = Type::witness_of(Type::field().array_of(5));
        let expected = Type::witness_of(Type::witness_of(Type::field()).array_of(5));
        assert_eq!(Type::join(&arr_wf, &w_arr_f), expected);
        assert_eq!(Type::join(&w_arr_f, &arr_wf), expected);
    }

    #[test]
    fn join_tuple() {
        let t1 = Type::tuple_of(vec![Type::field(), Type::int(32)]);
        let t2 = Type::tuple_of(vec![Type::witness_of(Type::field()), Type::int(32)]);
        let expected = Type::tuple_of(vec![Type::witness_of(Type::field()), Type::int(32)]);
        assert_eq!(Type::join(&t1, &t2), expected);
    }

    #[test]
    fn join_nested_array() {
        // join(Array<Array<Field, 3>, 2>, Array<Array<WitnessOf(Field), 3>, 2>)
        //   = Array<Array<WitnessOf(Field), 3>, 2>
        let inner_f = Type::field().array_of(3);
        let inner_wf = Type::witness_of(Type::field()).array_of(3);
        let a = inner_f.array_of(2);
        let b = inner_wf.array_of(2);
        assert_eq!(Type::join(&a, &b), b);
    }

    #[test]
    fn join_slice() {
        let sf = Type::field().slice_of();
        let swf = Type::witness_of(Type::field()).slice_of();
        assert_eq!(Type::join(&sf, &swf), swf);
    }

    // --- double-witness rejection ---

    #[test]
    #[should_panic(expected = "ICE: attempted to construct WitnessOf(WitnessOf(...))")]
    fn witness_of_double_wrap_panics() {
        let wf = Type::witness_of(Type::field());
        let _wwf = Type::witness_of(wf);
    }

    // --- element access through a witnessed container ---

    #[test]
    fn get_array_element_of_witnessed_container_wraps_once() {
        let t = Type::witness_of(Type::field().array_of(3));
        assert_eq!(t.get_array_element(), Type::witness_of(Type::field()));
    }

    #[test]
    fn get_array_element_of_witnessed_container_with_witness_elems_no_double_wrap() {
        let t = Type::witness_of(Type::witness_of(Type::field()).array_of(3));
        assert_eq!(t.get_array_element(), Type::witness_of(Type::field()));
    }

    #[test]
    fn get_tuple_element_of_witnessed_tuple_wraps_once() {
        let t = Type::witness_of(Type::tuple_of(vec![
            Type::witness_of(Type::field()),
            Type::field(),
        ]));
        assert_eq!(t.get_tuple_element(0), Type::witness_of(Type::field()));
        assert_eq!(t.get_tuple_element(1), Type::witness_of(Type::field()));
        assert_eq!(
            t.get_tuple_elements(),
            vec![
                Type::witness_of(Type::field()),
                Type::witness_of(Type::field())
            ]
        );
    }

    // --- join properties ---

    #[test]
    fn join_commutative() {
        let a = Type::field().array_of(3);
        let b = Type::witness_of(Type::field()).array_of(3);
        assert_eq!(Type::join(&a, &b), Type::join(&b, &a));
    }

    #[test]
    fn join_idempotent() {
        let t = Type::witness_of(Type::field()).array_of(5);
        assert_eq!(Type::join(&t, &t), t);
    }

    #[test]
    fn join_associative() {
        let a = Type::field().array_of(3);
        let b = Type::witness_of(Type::field()).array_of(3);
        let c = Type::witness_of(Type::field().array_of(3));
        let ab_c = Type::join(&Type::join(&a, &b), &c);
        let a_bc = Type::join(&a, &Type::join(&b, &c));
        assert_eq!(ab_c, a_bc);
    }

    // --- subtype consistent with join ---

    #[test]
    fn subtype_iff_join_equals_supertype() {
        let pairs: Vec<(Type, Type)> = vec![
            (Type::field(), Type::witness_of(Type::field())),
            (
                Type::field().array_of(3),
                Type::witness_of(Type::field()).array_of(3),
            ),
            (
                Type::field().array_of(3),
                Type::witness_of(Type::field().array_of(3)),
            ),
            (
                Type::tuple_of(vec![Type::field(), Type::int(8)]),
                Type::tuple_of(vec![Type::witness_of(Type::field()), Type::int(8)]),
            ),
        ];
        for (sub, sup) in &pairs {
            assert!(
                sub.is_subtype_of(sup),
                "{} should be subtype of {}",
                sub,
                sup
            );
            assert_eq!(
                Type::join(sub, sup),
                *sup,
                "join({}, {}) should equal the supertype",
                sub,
                sup
            );
        }
    }
}
