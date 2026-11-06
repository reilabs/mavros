use super::{globals::Globals, plan::plan_function, rewriter::rewrite_function};
use crate::compiler::{
    analysis::{flow_analysis::FlowAnalysis, types::TypeInfo},
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    passes::wide_witness_ints::rewriter::lowers_a_gadget,
    ssa::{BlockId, FunctionId, hlssa::HLSSA},
};
pub struct WideWitnessInts {}

impl WideWitnessInts {
    pub fn new() -> Self {
        WideWitnessInts {}
    }
}

impl Default for WideWitnessInts {
    fn default() -> Self {
        Self::new()
    }
}

impl Pass for WideWitnessInts {
    fn name(&self) -> &'static str {
        "wide_witness_ints"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![TypeInfo::id(), FlowAnalysis::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        let field = ssa.field();
        let globals = Globals::split(ssa, field);

        let type_info = store.get::<TypeInfo>();
        let cfg = store.get::<FlowAnalysis>();

        let function_ids: Vec<FunctionId> = ssa.get_function_ids().collect();

        for fid in function_ids {
            let reachable: Vec<BlockId> = cfg
                .get_function_cfg(fid)
                .get_domination_pre_order()
                .collect();
            // Only reachable blocks are rewritten, so an orphan would keep its wide values and its
            // global slots unrenumbered. None reaches here: the pipeline reclaims orphans before
            // the lowerings just ahead of this pass, which type every block against reachable-only
            // type info.
            debug_assert_eq!(
                reachable.len(),
                ssa.get_function(fid).get_blocks().count(),
                "ICE: an unreachable block reached the multi-cell representation"
            );
            let value_map = plan_function(ssa, fid, &reachable, type_info);
            // A function with no limbs can still hold an operation whose operands each fit an
            // element but whose sum or product does not; the chain, schoolbook and division lower
            // those too.
            let fti = type_info.get_function(fid);
            let lowers_here = || {
                reachable.iter().any(|bid| {
                    ssa.get_function(fid)
                        .get_block(*bid)
                        .get_instructions()
                        .any(|op| lowers_a_gadget(op, fti, field))
                })
            };
            let reads_a_moved_global = || {
                globals.moved
                    && reachable.iter().any(|bid| {
                        ssa.get_function(fid)
                            .get_block(*bid)
                            .get_instructions()
                            .any(|op| Globals::slot(op).is_some())
                    })
            };
            if value_map.values().all(|limbs| limbs.len() == 1)
                && !lowers_here()
                && !reads_a_moved_global()
            {
                continue;
            }
            rewrite_function(ssa, fid, &reachable, &value_map, type_info, field, &globals);
        }
    }

    fn preserves(&self) -> Vec<AnalysisId> {
        // Signatures, values and block parameters all change.
        vec![]
    }
}
