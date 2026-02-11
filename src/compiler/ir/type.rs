use std::fmt::{Debug, Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeExpr {
    Field,
    U(usize),
    WitnessOf(Box<Type>),
    Array(Box<Type>, usize),
    Slice(Box<Type>),
    Ref(Box<Type>),
    Tuple(Vec<Type>),
    Function,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Type {
    pub expr: TypeExpr,
}

impl Display for Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        match &self.expr {
            TypeExpr::Field => write!(f, "Field"),
            TypeExpr::U(size) => write!(f, "u{}", size),
            TypeExpr::WitnessOf(inner) => write!(f, "WitnessOf({})", inner),
            TypeExpr::Array(inner, size) => write!(f, "Array<{}, {}>", inner, size),
            TypeExpr::Slice(inner) => write!(f, "Slice<{}>", inner),
            TypeExpr::Ref(inner) => write!(f, "Ref<{}>", inner),
            TypeExpr::Tuple(elements) => write!(
                f, "Tuple<{}>",
                elements.iter().map(|e| format!("{}", e)).collect::<Vec<_>>().join(", ")
            ),
            TypeExpr::Function => write!(f, "Function"),
        }
    }
}

impl Type {
    // --- Constructors ---

    pub fn field() -> Self {
        Type { expr: TypeExpr::Field }
    }

    pub fn u(size: usize) -> Self {
        Type { expr: TypeExpr::U(size) }
    }

    pub fn bool() -> Self {
        Type::u(1)
    }

    pub fn u32() -> Self {
        Type::u(32)
    }

    pub fn function() -> Self {
        Type { expr: TypeExpr::Function }
    }

    pub fn array_of(self, size: usize) -> Self {
        Type { expr: TypeExpr::Array(Box::new(self), size) }
    }

    pub fn slice_of(self) -> Self {
        Type { expr: TypeExpr::Slice(Box::new(self)) }
    }

    pub fn ref_of(self) -> Self {
        Type { expr: TypeExpr::Ref(Box::new(self)) }
    }

    pub fn tuple_of(types: Vec<Self>) -> Self {
        Type { expr: TypeExpr::Tuple(types) }
    }

    /// Construct a WitnessOf type, enforcing idempotency:
    /// WitnessOf(WitnessOf(X)) = WitnessOf(X)
    pub fn witness_of(inner: Type) -> Self {
        match inner.expr {
            TypeExpr::WitnessOf(_) => inner, // idempotent
            _ => Type { expr: TypeExpr::WitnessOf(Box::new(inner)) },
        }
    }

    // --- Predicates ---

    pub fn is_numeric(&self) -> bool {
        match &self.expr {
            TypeExpr::U(_) | TypeExpr::Field => true,
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

    pub fn is_u(&self) -> bool {
        matches!(self.expr, TypeExpr::U(_))
    }

    pub fn is_u32(&self) -> bool {
        matches!(self.expr, TypeExpr::U(32))
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

    pub fn has_eq(&self) -> bool {
        matches!(self.expr, TypeExpr::Field | TypeExpr::U(_))
    }

    pub fn is_ref(&self) -> bool {
        matches!(self.expr, TypeExpr::Ref(_))
    }

    pub fn is_tuple(&self) -> bool {
        matches!(self.expr, TypeExpr::Tuple(_))
    }

    // --- Accessors ---

    pub fn get_array_element(&self) -> Self {
        match &self.expr {
            TypeExpr::Array(inner, _) => *inner.clone(),
            TypeExpr::Slice(inner) => *inner.clone(),
            TypeExpr::WitnessOf(inner) => {
                let elem = inner.get_array_element();
                Type::witness_of(elem)
            }
            _ => panic!("Type is not an array: {}", self),
        }
    }

    pub fn get_pointed(&self) -> Self {
        match &self.expr {
            TypeExpr::Ref(inner) => *inner.clone(),
            _ => panic!("Type is not a reference: {}", self),
        }
    }

    pub fn get_tuple_element(&self, index: usize) -> Self {
        match &self.expr {
            TypeExpr::Tuple(elements) => elements[index].clone(),
            TypeExpr::WitnessOf(inner) => {
                let elem = inner.get_tuple_element(index);
                Type::witness_of(elem)
            }
            _ => panic!("Type is not a tuple: {}", self),
        }
    }

    pub fn get_tuple_elements(&self) -> Vec<Self> {
        match &self.expr {
            TypeExpr::Tuple(elements) => elements.clone(),
            TypeExpr::WitnessOf(inner) => {
                inner.get_tuple_elements().into_iter().map(Type::witness_of).collect()
            }
            _ => panic!("Type is not a tuple: {}", self),
        }
    }

    pub fn get_refered(&self) -> &Self {
        match &self.expr {
            TypeExpr::Ref(inner) => inner.as_ref(),
            _ => panic!("Type is not a reference: {}", self),
        }
    }

    pub fn get_bit_size(&self) -> usize {
        match &self.expr {
            TypeExpr::U(size) => *size,
            TypeExpr::Field => 254, // TODO: parametrize
            TypeExpr::WitnessOf(inner) => inner.get_bit_size(),
            _ => panic!("Type is not a u: {}", self),
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

    /// Recursively strip all WitnessOf wrappers at every level.
    pub fn strip_all_witness(&self) -> Self {
        match &self.expr {
            TypeExpr::WitnessOf(inner) => inner.strip_all_witness(),
            TypeExpr::Array(inner, size) => {
                Type { expr: TypeExpr::Array(Box::new(inner.strip_all_witness()), *size) }
            }
            TypeExpr::Slice(inner) => {
                Type { expr: TypeExpr::Slice(Box::new(inner.strip_all_witness())) }
            }
            TypeExpr::Ref(inner) => {
                Type { expr: TypeExpr::Ref(Box::new(inner.strip_all_witness())) }
            }
            TypeExpr::Tuple(elements) => {
                Type { expr: TypeExpr::Tuple(elements.iter().map(|e| e.strip_all_witness()).collect()) }
            }
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
            (TypeExpr::U(n), TypeExpr::U(m)) => n == m,
            (TypeExpr::Array(x, n), TypeExpr::Array(y, m)) => n == m && x.is_subtype_of(y),
            (TypeExpr::Slice(x), TypeExpr::Slice(y)) => x.is_subtype_of(y),
            (TypeExpr::Tuple(xs), TypeExpr::Tuple(ys)) => {
                xs.len() == ys.len()
                    && xs.iter().zip(ys.iter()).all(|(x, y)| x.is_subtype_of(y))
            }
            (TypeExpr::Ref(x), TypeExpr::Ref(y)) => x == y, // invariant
            (TypeExpr::Function, TypeExpr::Function) => true,
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
            (TypeExpr::WitnessOf(inner_a), _) => {
                Type::witness_of(Type::join(inner_a, b))
            }
            (_, TypeExpr::WitnessOf(inner_b)) => {
                Type::witness_of(Type::join(a, inner_b))
            }
            // Structural (neither is WitnessOf)
            (TypeExpr::Field, TypeExpr::Field) => Type::field(),
            (TypeExpr::U(n), TypeExpr::U(m)) => {
                assert_eq!(n, m, "Cannot join U({}) and U({})", n, m);
                Type::u(*n)
            }
            (TypeExpr::Array(x, n), TypeExpr::Array(y, m)) => {
                assert_eq!(n, m, "Cannot join arrays of different sizes: {} vs {}", n, m);
                Type::join(x, y).array_of(*n)
            }
            (TypeExpr::Slice(x), TypeExpr::Slice(y)) => {
                Type::join(x, y).slice_of()
            }
            (TypeExpr::Tuple(xs), TypeExpr::Tuple(ys)) => {
                assert_eq!(xs.len(), ys.len(), "Cannot join tuples of different lengths");
                Type::tuple_of(
                    xs.iter()
                        .zip(ys.iter())
                        .map(|(x, y)| Type::join(x, y))
                        .collect(),
                )
            }
            (TypeExpr::Ref(x), TypeExpr::Ref(y)) => Type::join(x, y).ref_of(),
            (TypeExpr::Function, TypeExpr::Function) => Type::function(),
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
            // If either is WitnessOf, unwrap, compute, then re-wrap
            (TypeExpr::WitnessOf(inner), _) => {
                Type::witness_of(inner.get_arithmetic_result_type(other))
            }
            (_, TypeExpr::WitnessOf(inner)) => {
                Type::witness_of(self.get_arithmetic_result_type(inner))
            }
            (TypeExpr::Field, _) | (_, TypeExpr::Field) => Type::field(),
            (TypeExpr::U(size1), TypeExpr::U(size2)) => Type::u(*size1.max(size2)),
            _ => panic!("Cannot perform arithmetic on types {} and {}", self, other),
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
            TypeExpr::U(_) => false,
            TypeExpr::Function => false,
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
            TypeExpr::U(_) => 1,
            TypeExpr::WitnessOf(_) => 1, // pointer-sized (witness tape reference)
            _ => panic!("Cannot currently calculate size for type {}", self),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- is_subtype_of ---

    #[test]
    fn subtype_reflexive() {
        assert!(Type::field().is_subtype_of(&Type::field()));
        assert!(Type::u(32).is_subtype_of(&Type::u(32)));
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
        let u = Type::u(32);
        let wu = Type::witness_of(Type::u(32));
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
        let t1 = Type::tuple_of(vec![Type::field(), Type::u(32)]);
        let t2 = Type::tuple_of(vec![Type::witness_of(Type::field()), Type::u(32)]);
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
        assert!(!Type::field().is_subtype_of(&Type::u(32)));
        assert!(!Type::u(32).is_subtype_of(&Type::field()));
        assert!(!Type::u(8).is_subtype_of(&Type::u(32)));
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
        assert!(!Type::field().needs_witness_cast(&Type::u(32)));
    }

    // --- join ---

    #[test]
    fn join_same_types() {
        assert_eq!(Type::join(&Type::field(), &Type::field()), Type::field());
        assert_eq!(Type::join(&Type::u(32), &Type::u(32)), Type::u(32));
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
        let t1 = Type::tuple_of(vec![Type::field(), Type::u(32)]);
        let t2 = Type::tuple_of(vec![Type::witness_of(Type::field()), Type::u(32)]);
        let expected = Type::tuple_of(vec![Type::witness_of(Type::field()), Type::u(32)]);
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

    // --- idempotency ---

    #[test]
    fn witness_of_idempotent() {
        let wf = Type::witness_of(Type::field());
        let wwf = Type::witness_of(wf.clone());
        assert_eq!(wf, wwf);
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
            (Type::field().array_of(3), Type::witness_of(Type::field()).array_of(3)),
            (Type::field().array_of(3), Type::witness_of(Type::field().array_of(3))),
            (
                Type::tuple_of(vec![Type::field(), Type::u(8)]),
                Type::tuple_of(vec![Type::witness_of(Type::field()), Type::u(8)]),
            ),
        ];
        for (sub, sup) in &pairs {
            assert!(sub.is_subtype_of(sup), "{} should be subtype of {}", sub, sup);
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
