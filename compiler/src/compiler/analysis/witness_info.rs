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

#[derive(PartialEq, Eq, Debug, Clone, Hash)]
pub enum WitnessShape {
    Scalar(WitnessInfo),
    Array(WitnessInfo, Box<WitnessShape>),
    Ref(WitnessInfo, Box<WitnessShape>),
}

impl Display for WitnessShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WitnessShape::Scalar(info) => write!(f, "{info}"),
            WitnessShape::Array(info, inner) => {
                write!(f, "[{info} of {inner}]")
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
            (WitnessShape::Array(t1, inner1), WitnessShape::Array(t2, inner2)) => {
                WitnessShape::Array(t1.join(*t2), Box::new(inner1.join(inner2)))
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

    pub fn toplevel_info(&self) -> WitnessType {
        match self {
            WitnessShape::Scalar(info) => *info,
            WitnessShape::Array(info, _) => *info,
            WitnessShape::Ref(info, _) => *info,
        }
    }

    /// Whether any level of this shape is witness-typed.
    pub fn contains_witness(&self) -> bool {
        match self {
            WitnessShape::Scalar(info) => info.is_witness(),
            WitnessShape::Array(info, inner) | WitnessShape::Ref(info, inner) => {
                info.is_witness() || inner.contains_witness()
            }
        }
    }

    pub fn child_witness_type(&self) -> Option<WitnessShape> {
        match self {
            WitnessShape::Array(_, inner) => Some(*inner.clone()),
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
