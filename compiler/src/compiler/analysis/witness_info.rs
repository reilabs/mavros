use itertools::Itertools;

use std::fmt::Display;

use crate::{
    collections::HashMap,
    compiler::ssa::{BlockId, FunctionId, SSAAnotator, ValueId},
};

#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash)]
pub enum WitnessType {
    Pure,
    Witness,
}

impl std::fmt::Display for WitnessType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WitnessType::Pure => write!(f, "P"),
            WitnessType::Witness => write!(f, "W"),
        }
    }
}

impl WitnessType {
    pub fn is_pure(&self) -> bool {
        match self {
            WitnessType::Pure => true,
            WitnessType::Witness => false,
        }
    }

    pub fn is_witness(&self) -> bool {
        match self {
            WitnessType::Pure => false,
            WitnessType::Witness => true,
        }
    }

    pub fn join(self, other: WitnessType) -> WitnessType {
        match (self, other) {
            (WitnessType::Witness, _) | (_, WitnessType::Witness) => WitnessType::Witness,
            _ => WitnessType::Pure,
        }
    }
}

pub type WitnessInfo = WitnessType;

/// Per-level witness-ness of a value's type.
///
/// `Scalar` and `Ref` carry one; `Array` carries none; `Slice` carries one, but for its *length*
/// rather than its identity. This is necessary because:
///
/// - An array's length is static, so a witness "container identity" was never anything its
///   element taint did not already cover — nothing but the `Cast`-to-`WitnessOf` rule ever wrote
///   to that slot, and it now seeds the leaves instead (see `build_instr` in [`super::
///   witness_taint_inference::builder`]).
/// - A slice's length *is* real state, but `purify_witness_slices` moves it onto the `log_len`
///   scalar of a `(physical, log_len, start)` tuple before any type is wrapped, so the slice level
///   must report `Pure` from then on (see [`WitnessShape::toplevel_info`]).
///
/// With no container-level wrap, `WitnessOf` can _only ever sit on a leaf_. The witness-indexed
/// rebuild scans **rely on this**. `instruction_lowering::witness_array::select_leaves`, for
/// example treats `TypeExpr::WitnessOf(_)` as a scalar and would emit a single `Select` over a
/// whole array if a `WitnessOf(Array<..>)` could still reach it. Re-introducing a container taint
/// slot means fixing that consumer in the same change.
#[derive(PartialEq, Eq, Debug, Clone, Hash)]
pub enum WitnessShape {
    Scalar(WitnessInfo),
    Array(Box<WitnessShape>),
    Slice(WitnessInfo, Box<WitnessShape>),
    Ref(WitnessInfo, Box<WitnessShape>),
}

impl Display for WitnessShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WitnessShape::Scalar(info) => write!(f, "{info}"),
            WitnessShape::Array(inner) => {
                write!(f, "[{inner}]")
            }
            WitnessShape::Slice(len, elem) => {
                write!(f, "[len:{len} of {elem}]")
            }
            WitnessShape::Ref(info, inner) => {
                write!(f, "[*{info} of {inner}]")
            }
        }
    }
}

impl WitnessShape {
    /// Join two witness types (least upper bound). Eagerly computes concrete result.
    pub fn join(&self, other: &WitnessShape) -> WitnessShape {
        match (self, other) {
            (WitnessShape::Scalar(t1), WitnessShape::Scalar(t2)) => {
                WitnessShape::Scalar(t1.join(*t2))
            }
            (WitnessShape::Array(inner1), WitnessShape::Array(inner2)) => {
                WitnessShape::Array(Box::new(inner1.join(inner2)))
            }
            (WitnessShape::Slice(l1, e1), WitnessShape::Slice(l2, e2)) => {
                WitnessShape::Slice(l1.join(*l2), Box::new(e1.join(e2)))
            }
            (WitnessShape::Ref(t1, inner1), WitnessShape::Ref(t2, inner2)) => {
                WitnessShape::Ref(t1.join(*t2), Box::new(inner1.join(inner2)))
            }
            _ => panic!(
                "Cannot join different witness types: {:?} vs {:?}",
                self, other
            ),
        }
    }

    /// The witness-ness of the shape's top level — the level `apply_witness_type` mirrors into
    /// a `TypeExpr::WitnessOf` wrap.
    ///
    /// A slice answers `Pure` even when its length slot is witness: `purify_witness_slices` moves
    /// that witness-ness onto the `log_len` scalar of a `(physical, log_len, start)` tuple. So the
    /// slice value itself must never be wrapped. Consumers that need the length slot read the raw
    /// shape instead.
    pub fn toplevel_info(&self) -> WitnessType {
        match self {
            WitnessShape::Scalar(info) => *info,
            WitnessShape::Array(_) | WitnessShape::Slice(_, _) => WitnessType::Pure,
            WitnessShape::Ref(info, _) => *info,
        }
    }

    /// Whether any level of this shape is witness-typed.
    pub fn contains_witness(&self) -> bool {
        match self {
            WitnessShape::Scalar(info) => info.is_witness(),
            WitnessShape::Array(inner) => inner.contains_witness(),
            WitnessShape::Ref(info, inner) => info.is_witness() || inner.contains_witness(),
            WitnessShape::Slice(len, elem) => len.is_witness() || elem.contains_witness(),
        }
    }

    pub fn child_witness_type(&self) -> Option<WitnessShape> {
        match self {
            WitnessShape::Array(inner) => Some(*inner.clone()),
            WitnessShape::Slice(_, elem) => Some(*elem.clone()),
            WitnessShape::Ref(_, inner) => Some(*inner.clone()),
            WitnessShape::Scalar(_) => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FunctionWitnessType {
    pub returns_witness: Vec<WitnessShape>,
    pub cfg_witness: WitnessInfo,
    pub parameters: Vec<WitnessShape>,
    pub block_cfg_witness: HashMap<BlockId, WitnessInfo>,
    pub value_witness_types: HashMap<ValueId, WitnessShape>,
}

impl FunctionWitnessType {
    /// The inferred shape of `value_id`, if one was recorded.
    ///
    /// Only block parameters and instruction results are recorded; constants are not (they are
    /// always all-Pure). Callers looking up arbitrary operands must treat `None` as Pure rather
    /// than unwrapping (see `get_witness_or_pure` in `untaint_control_flow`).
    pub fn try_get_value_witness_type(&self, value_id: ValueId) -> Option<&WitnessShape> {
        self.value_witness_types.get(&value_id)
    }
}

impl SSAAnotator for FunctionWitnessType {
    fn annotate_value(&self, _: FunctionId, value_id: ValueId) -> String {
        let Some(wt) = self.value_witness_types.get(&value_id) else {
            return "".to_string();
        };
        wt.to_string()
    }

    fn annotate_block(&self, _: FunctionId, block_id: BlockId) -> String {
        let Some(info) = self.block_cfg_witness.get(&block_id) else {
            return "".to_string();
        };
        format!("cfg_witness: {}", info)
    }

    fn annotate_function(&self, _: FunctionId) -> String {
        let return_types = self
            .returns_witness
            .iter()
            .map(|t| t.to_string())
            .join(", ");
        format!(
            "returns: [{}], cfg_witness: {}",
            return_types, self.cfg_witness
        )
    }
}

impl Display for FunctionWitnessType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "returns: {:?}\nparameters: {:?}\nvalue_witness_types: {:?}\ncfg_witness: {:?}",
            self.returns_witness, self.parameters, self.value_witness_types, self.cfg_witness
        )
    }
}
