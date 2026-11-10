use super::{limb_config::LimbConfig, types::limb_types};
use crate::compiler::ssa::hlssa::{HLSSA, OpCode, Type};

/// The global slots after each global holding wide integers has one slot per limb.
///
/// A global holding wide integers as sequence elements takes the same transposed shape as a local:
/// one sequence per limb position, each in a separate slot. Every later slot moves along, so every
/// global opcode is renumbered, whatever it holds.
pub(super) struct Globals {
    /// The first new slot of each old one.
    pub(super) offsets: Vec<usize>,

    /// The old slots' types, which say how many new slots each became.
    pub(super) old_types: Vec<Type>,

    /// Whether any slot moved; if none did, nothing needs renumbering.
    pub(super) moved: bool,
}

impl Globals {
    /// Split every global in `ssa` that holds a wide integer, and record where each slot went.
    pub(super) fn split(ssa: &mut HLSSA, limb_config: LimbConfig) -> Self {
        let old_types: Vec<Type> = ssa.get_global_types().to_vec();
        let mut offsets = Vec::with_capacity(old_types.len());
        let mut new_types = Vec::new();
        for ty in &old_types {
            offsets.push(new_types.len());
            new_types.extend(limb_types(ty, limb_config));
        }
        let moved = new_types.len() != old_types.len();
        if moved {
            ssa.set_global_types(new_types);
        }
        Self {
            offsets,
            old_types,
            moved,
        }
    }

    /// The old slot a global opcode names, if `op` is one.
    pub(super) fn slot(op: &OpCode) -> Option<usize> {
        match op {
            OpCode::ReadGlobal { offset, .. } => Some(*offset as usize),
            OpCode::InitGlobal { global, .. } | OpCode::DropGlobal { global } => Some(*global),
            _ => None,
        }
    }
}
