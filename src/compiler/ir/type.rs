use std::fmt::{Debug, Display, Formatter};

pub trait CommutativeMonoid {
    fn empty() -> Self;
    fn op(&self, other: &Self) -> Self;
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub struct Empty;

impl CommutativeMonoid for Empty {
    fn empty() -> Self {
        Empty
    }

    fn op(&self, _: &Self) -> Self {
        Empty
    }
}

impl Display for Empty {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeExpr<V> {
    Field,
    U(usize),
    WitnessRef,
    Array(Box<Type<V>>, usize),
    Slice(Box<Type<V>>),
    Ref(Box<Type<V>>),
    Tuple(Vec<Type<V>>),
}

impl<V> TypeExpr<V> {
    pub fn equal_up_to_annotation(&self, other: &Self) -> bool {
        match (self, other) {
            (TypeExpr::Field, TypeExpr::Field) => true,
            (TypeExpr::U(size1), TypeExpr::U(size2)) => size1 == size2,
            (TypeExpr::Array(inner1, size1), TypeExpr::Array(inner2, size2)) => {
                inner1.equal_up_to_annotation(inner2) && size1 == size2
            }
            (TypeExpr::Slice(inner1), TypeExpr::Slice(inner2)) => {
                inner1.equal_up_to_annotation(inner2)
            }
            (TypeExpr::Ref(inner1), TypeExpr::Ref(inner2)) => inner1.equal_up_to_annotation(inner2),
            (TypeExpr::WitnessRef, TypeExpr::WitnessRef) => true,
            _ => false,
        }
    }

    pub fn as_pure<V2: CommutativeMonoid>(&self) -> TypeExpr<V2> {
        match self {
            TypeExpr::Field => TypeExpr::Field,
            TypeExpr::U(size) => TypeExpr::U(*size),
            TypeExpr::WitnessRef => TypeExpr::WitnessRef,
            TypeExpr::Array(inner, size) => TypeExpr::Array(Box::new(inner.as_pure()), *size),
            TypeExpr::Slice(inner) => TypeExpr::Slice(Box::new(inner.as_pure())),
            TypeExpr::Ref(inner) => TypeExpr::Ref(Box::new(inner.as_pure())),
            TypeExpr::Tuple(_elements) => {todo!("Tuples not supported yet")}
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Type<V> {
    pub expr: TypeExpr<V>,
    pub annotation: V,
}

impl<V: Display> Display for Type<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        fn format_annotation<V: Display>(info: &V) -> String {
            let inner = format!("{}", info);
            if inner.is_empty() {
                inner
            } else {
                format!("[{}]", inner)
            }
        }

        match &self.expr {
            TypeExpr::Field => write!(f, "Field{}", format_annotation(&self.annotation)),
            TypeExpr::U(size) => write!(f, "u{}{}", size, format_annotation(&self.annotation)),
            TypeExpr::Array(inner, size) => write!(
                f,
                "Array{}<{}, {}>",
                format_annotation(&self.annotation),
                inner,
                size,
            ),
            TypeExpr::Slice(inner) => {
                write!(f, "Slice{}<{}>", format_annotation(&self.annotation), inner,)
            }
            TypeExpr::Ref(inner) => {
                write!(f, "Ref{}<{}>", format_annotation(&self.annotation), inner)
            }
            TypeExpr::WitnessRef => write!(f, "WitnessRef{}", format_annotation(&self.annotation)),
            TypeExpr::Tuple(elements) => write!(
                f, "Tuple{}<{}>", format_annotation(&self.annotation), elements.iter().map(|e| format!("{}", e)).collect::<Vec<_>>().join(", ")
            ),
        }
    }
}

impl<V: CommutativeMonoid + Display> Type<V> {
    pub fn get_arithmetic_result_type(&self, other: &Self) -> Self {
        match (&self.expr, &other.expr) {
            (TypeExpr::Field, _) | (_, TypeExpr::Field) => {
                Type::field(self.annotation.op(&other.annotation))
            }
            (TypeExpr::U(size1), TypeExpr::U(size2)) => {
                Type::u(*size1.max(size2), self.annotation.op(&other.annotation))
            }
            (TypeExpr::WitnessRef, _) | (_, TypeExpr::WitnessRef) => {
                Type::witness_ref(self.annotation.op(&other.annotation))
            }
            _ => panic!("Cannot perform arithmetic on types {} and {}", self, other),
        }
    }

    pub fn combine_with_annotation(self, other: &V) -> Self {
        Type {
            expr: self.expr,
            annotation: self.annotation.op(other),
        }
    }

    pub fn contains_ptrs(&self) -> bool {
        match &self.expr {
            TypeExpr::Ref(_) => true,
            TypeExpr::Array(inner, _) => inner.contains_ptrs(),
            TypeExpr::Slice(inner) => inner.contains_ptrs(),
            TypeExpr::Field => false,
            TypeExpr::U(_) => false,
            TypeExpr::WitnessRef => false,
            TypeExpr::Tuple(elements) => {
                for elem in elements {
                    if elem.contains_ptrs() {
                        return true;
                    }
                }
                false
            }
        }
    }
}

impl<V: Clone> Type<V> {
    pub fn get_array_element(&self) -> Self {
        match &self.expr {
            TypeExpr::Array(inner, _) => *inner.clone(),
            TypeExpr::Slice(inner) => *inner.clone(),
            _ => panic!("Type is not an array"),
        }
    }

    pub fn get_pointed(&self) -> Self {
        match &self.expr {
            TypeExpr::Ref(inner) => *inner.clone(),
            _ => panic!("Type is not a reference"),
        }
    }

    pub fn get_tuple_element(&self, index: usize) -> Self {
        match &self.expr {
            TypeExpr::Tuple(elements) => elements[index].clone(),
            _ => panic!("Type is not a tuple"),
        }
    }

    pub fn get_tuple_elements(&self) -> &Vec<Self> {
        match &self.expr {
            TypeExpr::Tuple(elements) => elements,
            _ => panic!("Type is not a tuple"),
        }
    }
}

impl<V> Type<V> {
    pub fn get_annotation(&self) -> &V {
        &self.annotation
    }

    pub fn bool(annotation: V) -> Self {
        Type {
            expr: TypeExpr::U(1),
            annotation,
        }
    }

    pub fn field(annotation: V) -> Self {
        Type {
            expr: TypeExpr::Field,
            annotation,
        }
    }

    pub fn u(size: usize, annotation: V) -> Self {
        Type {
            expr: TypeExpr::U(size),
            annotation,
        }
    }

    pub fn witness_ref(annotation: V) -> Self {
        Type {
            expr: TypeExpr::WitnessRef,
            annotation,
        }
    }

    pub fn u32(annotation: V) -> Self {
        Type::u(32, annotation)
    }

    pub fn array_of(self, size: usize, annotation: V) -> Self {
        Type {
            expr: TypeExpr::Array(Box::new(self), size),
            annotation,
        }
    }

    pub fn slice_of(self, annotation: V) -> Self {
        Type {
            expr: TypeExpr::Slice(Box::new(self)),
            annotation,
        }
    }

    pub fn ref_of(self, annotation: V) -> Self {
        Type {
            expr: TypeExpr::Ref(Box::new(self)),
            annotation,
        }
    }
    
    pub fn tuple_of(types: Vec<Self>, annotation: V) -> Self {
        Type {
            expr: TypeExpr::Tuple(types),
            annotation,
        }
    }

    pub fn is_numeric(&self) -> bool {
        matches!(self.expr, TypeExpr::U(_) | TypeExpr::Field)
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

    pub fn is_witness_ref(&self) -> bool {
        matches!(self.expr, TypeExpr::WitnessRef)
    }

    pub fn is_u(&self) -> bool {
        matches!(self.expr, TypeExpr::U(_))
    }

    pub fn is_u32(&self) -> bool {
        matches!(self.expr, TypeExpr::U(32))
    }

    pub fn is_heap_allocated(&self) -> bool {
        matches!(self.expr, TypeExpr::WitnessRef | TypeExpr::Array(_, _) | TypeExpr::Slice(_) | TypeExpr::Ref(_) | TypeExpr::Tuple(_))
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

    pub fn equal_up_to_annotation(&self, other: &Self) -> bool {
        self.expr.equal_up_to_annotation(&other.expr)
    }

    pub fn is_ref_of(&self, other: &Self) -> bool {
        matches!(&self.expr, TypeExpr::Ref(inner) if inner.as_ref().equal_up_to_annotation(other))
    }

    pub fn get_refered(&self) -> &Self {
        match &self.expr {
            TypeExpr::Ref(inner) => inner.as_ref(),
            _ => panic!("Type is not a reference"),
        }
    }

    pub fn get_bit_size(&self) -> usize {
        match &self.expr {
            TypeExpr::U(size) => *size,
            TypeExpr::Field => 254, // TODO: parametrize
            _ => panic!("Type is not a u"),
        }
    }

    pub fn as_pure<V2: CommutativeMonoid>(&self) -> Type<V2> {
        Type {
            expr: self.expr.as_pure(),
            annotation: V2::empty(),
        }
    }

    pub fn calculate_type_size(&self) -> usize {
        match &self.expr {
            TypeExpr::Field => 1,
            TypeExpr::Array(inner, size) => 1,
            TypeExpr::Tuple(inner_types) => {
                inner_types.iter().map(|t| t.calculate_type_size()).sum()
            }
            _ => panic!("Cannot currently calculate size for types other than Field, Array, and Tuple"),
        }
    }
}
