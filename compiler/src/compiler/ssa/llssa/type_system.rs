use std::fmt::{self, Display, Formatter};

use crate::compiler::ssa::SSAType;

use super::LLStruct;

/// SSA value type for LLSSA.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    /// Sized unsigned integer. Int(1) = bool, Int(8) = byte, Int(32), Int(64).
    Int(u32),
    /// Opaque pointer.
    Ptr,
    /// Multi-word aggregate, by value. Only for value-safe structs.
    Struct(LLStruct),
}

impl Type {
    pub fn i1() -> Self {
        Type::Int(1)
    }
    pub fn i32() -> Self {
        Type::Int(32)
    }
    pub fn i64() -> Self {
        Type::Int(64)
    }
    pub fn ptr() -> Self {
        Type::Ptr
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Type::Int(bits) => write!(f, "i{}", bits),
            Type::Ptr => write!(f, "ptr"),
            Type::Struct(s) => write!(f, "{}", s),
        }
    }
}

impl SSAType for Type {}
