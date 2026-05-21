//! Contains the basic identifier types for the SSA.

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ValueId(pub u64);

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct BlockId(pub u64);

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct FunctionId(pub u64);
