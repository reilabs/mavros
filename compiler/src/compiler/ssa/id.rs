//! Contains the basic identifier types for the SSA.

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct ValueId(pub u64);

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct BlockId(pub u64);

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct FunctionId(pub u64);
