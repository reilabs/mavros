//! The windows of a witnessed field element, which `ShareFieldDecompositions` and
//! `OutlineFieldBitRanges` rewrite.
//!
//! Each such window lowers to its element's canonical byte decomposition, the bulk of its cost.

use crate::compiler::{
    analysis::types::FunctionTypeInfo,
    ssa::{
        ValueId,
        hlssa::{OpCode, TypeExpr},
    },
};

/// A window of a witnessed field element: `result = bit_range(value, offset, width)`, guarded or
/// not.
///
/// A pass that moves a guarded window takes it unguarded. That is sound because
/// `LowerSideEffectFreeGuards`, the first lowering after these passes, drops a `BitRange`'s guard
/// anyway, and the canonical decomposition holds for every element, so taking one where the guard
/// is off refuses no witness.
#[derive(Clone, Copy, Debug)]
pub struct FieldWindow {
    pub result: ValueId,
    pub value: ValueId,
    pub offset: usize,
    pub width: usize,
}

impl FieldWindow {
    /// The window `op` takes, if it is a window of a witnessed field element.
    pub fn of(op: &OpCode, types: &FunctionTypeInfo) -> Option<Self> {
        let op = match op {
            OpCode::Guard { inner, .. } => inner.as_ref(),
            op => op,
        };
        let OpCode::BitRange {
            result,
            value,
            offset,
            width,
        } = *op
        else {
            return None;
        };
        let value_type = types.get_value_type(value);
        let witnessed_field = value_type.is_witness_of()
            && matches!(value_type.strip_witness().expr, TypeExpr::Field);
        witnessed_field.then_some(Self {
            result,
            value,
            offset,
            width,
        })
    }

    /// The bit just past the window.
    pub fn end(&self) -> usize {
        self.offset + self.width
    }
}
