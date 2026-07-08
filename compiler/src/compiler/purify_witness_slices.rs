//! Purifies witness-length slices into `(physical, log_len)` pairs.

use crate::compiler::{
    analysis::witness_taint_inference::WitnessTaintInference, ssa::hlssa::HLSSA,
};

pub struct PurifyWitnessSlices {}

impl PurifyWitnessSlices {
    pub fn new() -> Self {
        Self {}
    }

    pub fn run(&self, ssa: HLSSA, _witness_inference: &WitnessTaintInference) -> HLSSA {
        ssa
    }
}
