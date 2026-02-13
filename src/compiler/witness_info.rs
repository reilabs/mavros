use itertools::Itertools;

use crate::compiler::ssa::{BlockId, FunctionId, SsaAnnotator, ValueId};
use std::collections::HashMap;

#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash)]
pub enum ConstantWitness {
    Pure,
    Witness,
}

impl std::fmt::Display for ConstantWitness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstantWitness::Pure => write!(f, "P"),
            ConstantWitness::Witness => write!(f, "W"),
        }
    }
}

impl ConstantWitness {
    pub fn is_pure(&self) -> bool {
        match self {
            ConstantWitness::Pure => true,
            ConstantWitness::Witness => false,
        }
    }

    pub fn is_witness(&self) -> bool {
        match self {
            ConstantWitness::Pure => false,
            ConstantWitness::Witness => true,
        }
    }

    pub fn join(self, other: ConstantWitness) -> ConstantWitness {
        match (self, other) {
            (ConstantWitness::Witness, _) | (_, ConstantWitness::Witness) => {
                ConstantWitness::Witness
            }
            _ => ConstantWitness::Pure,
        }
    }
}

/// WitnessInfo is now simply ConstantWitness — no more type variables or joins.
pub type WitnessInfo = ConstantWitness;

#[derive(PartialEq, Eq, Debug, Clone, Hash)]
pub enum WitnessType {
    Scalar(WitnessInfo),
    Array(WitnessInfo, Box<WitnessType>),
    Ref(WitnessInfo, Box<WitnessType>),
    Tuple(WitnessInfo, Vec<WitnessType>),
}

impl WitnessType {
    pub fn to_string(&self) -> String {
        match self {
            WitnessType::Scalar(info) => format!("{}", info),
            WitnessType::Array(info, inner) => {
                format!("[{} of {}]", info, inner.to_string())
            }
            WitnessType::Ref(info, inner) => {
                format!("[*{} of {}]", info, inner.to_string())
            }
            WitnessType::Tuple(info, children) => {
                format!(
                    "({} of <{}>)",
                    info,
                    children
                        .iter()
                        .map(|child| child.to_string())
                        .join(", ")
                )
            }
        }
    }

    /// Join two witness types (least upper bound). Eagerly computes concrete result.
    pub fn join(&self, other: &WitnessType) -> WitnessType {
        match (self, other) {
            (WitnessType::Scalar(t1), WitnessType::Scalar(t2)) => {
                WitnessType::Scalar(t1.join(*t2))
            }
            (WitnessType::Array(t1, inner1), WitnessType::Array(t2, inner2)) => {
                WitnessType::Array(t1.join(*t2), Box::new(inner1.join(inner2)))
            }
            (WitnessType::Ref(t1, inner1), WitnessType::Ref(t2, inner2)) => {
                WitnessType::Ref(t1.join(*t2), Box::new(inner1.join(inner2)))
            }
            (WitnessType::Tuple(t1, children1), WitnessType::Tuple(t2, children2)) => {
                let children_join = children1
                    .iter()
                    .zip(children2.iter())
                    .map(|(c1, c2)| c1.join(c2))
                    .collect();
                WitnessType::Tuple(t1.join(*t2), children_join)
            }
            _ => panic!(
                "Cannot join different witness types: {:?} vs {:?}",
                self, other
            ),
        }
    }

    /// Join two witness types, handling shape mismatches gracefully.
    /// When shapes don't match, keeps self's shape but joins in the other's toplevel
    /// witness info. Use this for block params and return type joining where self's
    /// shape is authoritative.
    pub fn try_join(&self, other: &WitnessType) -> WitnessType {
        if self.shape_compatible(other) {
            self.join(other)
        } else {
            // Keep self's shape, join in other's toplevel witness info
            self.with_toplevel_info(self.toplevel_info().join(other.toplevel_info()))
        }
    }


    /// Check if two WitnessTypes have the same shape (variant and children count).
    fn shape_compatible(&self, other: &WitnessType) -> bool {
        match (self, other) {
            (WitnessType::Scalar(_), WitnessType::Scalar(_)) => true,
            (WitnessType::Array(_, inner1), WitnessType::Array(_, inner2)) => {
                inner1.shape_compatible(inner2)
            }
            (WitnessType::Ref(_, inner1), WitnessType::Ref(_, inner2)) => {
                inner1.shape_compatible(inner2)
            }
            (WitnessType::Tuple(_, c1), WitnessType::Tuple(_, c2)) => {
                c1.len() == c2.len()
                    && c1.iter().zip(c2.iter()).all(|(a, b)| a.shape_compatible(b))
            }
            _ => false,
        }
    }

    pub fn toplevel_info(&self) -> ConstantWitness {
        match self {
            WitnessType::Scalar(info) => *info,
            WitnessType::Array(info, _) => *info,
            WitnessType::Ref(info, _) => *info,
            WitnessType::Tuple(info, _) => *info,
        }
    }

    pub fn child_witness_type(&self) -> Option<WitnessType> {
        match self {
            WitnessType::Array(_, inner) => Some(*inner.clone()),
            WitnessType::Ref(_, inner) => Some(*inner.clone()),
            WitnessType::Scalar(_) => None,
            WitnessType::Tuple(_, _) => {
                panic!("Error: child_witness_type shouldn't be called for Tuple values")
            }
        }
    }

    pub fn with_toplevel_info(&self, toplevel: WitnessInfo) -> WitnessType {
        match self {
            WitnessType::Scalar(_) => WitnessType::Scalar(toplevel),
            WitnessType::Array(_, inner) => WitnessType::Array(toplevel, inner.clone()),
            WitnessType::Ref(_, inner) => WitnessType::Ref(toplevel, inner.clone()),
            WitnessType::Tuple(_, inner) => WitnessType::Tuple(toplevel, inner.clone()),
        }
    }
}

#[derive(Clone)]
pub struct FunctionWitnessType {
    pub returns_witness: Vec<WitnessType>,
    pub cfg_witness: WitnessInfo,
    pub parameters: Vec<WitnessType>,
    pub block_cfg_witness: HashMap<BlockId, WitnessInfo>,
    pub value_witness_types: HashMap<ValueId, WitnessType>,
}

impl FunctionWitnessType {
    pub fn to_string(&self) -> String {
        format!(
            "returns: {:?}\nparameters: {:?}\nvalue_witness_types: {:?}\ncfg_witness: {:?}",
            self.returns_witness, self.parameters, self.value_witness_types, self.cfg_witness
        )
    }

    pub fn get_value_witness_type(&self, value_id: ValueId) -> &WitnessType {
        self.value_witness_types.get(&value_id).unwrap()
    }

    pub fn get_block_witness(&self, block_id: BlockId) -> &WitnessInfo {
        self.block_cfg_witness.get(&block_id).unwrap()
    }
}

impl SsaAnnotator for FunctionWitnessType {
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
