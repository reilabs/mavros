mod bit_range;
mod pure_guards;
mod witness_array;
mod witness_bitwise;
mod witness_integer_arith;
mod witness_integer_utils;
mod witness_spread;

use crate::compiler::{
    analysis::{
        flow_analysis::FlowAnalysis,
        types::{FunctionTypeInfo, Types},
        value_range_analysis::{FunctionValueRanges, IntInterval, ValueRangeAnalysis},
    },
    pass_manager::{AnalysisId, AnalysisStore, Pass},
    ssa::hlssa::{
        HLSSA, OpCode,
        builder::{HLBlockEmitter, HLEmitter, HLFunctionBuilder, HLSSABuilder},
    },
};

use self::{
    bit_range::LowerBitRangeOps, pure_guards::LowerPureGuards, witness_array::LowerWitnessArrayOps,
    witness_bitwise::LowerWitnessBitwiseOps, witness_integer_arith::LowerWitnessIntegerArithOps,
    witness_spread::LowerWitnessSpreadOps,
};

const ITERATION_LIMIT: usize = 32;

pub struct AlgebraicLowering {
    name: &'static str,
    lowerers: Vec<Box<dyn AlgebraicLoweringRule>>,
}

pub(super) struct LoweringContext<'a> {
    types: &'a FunctionTypeInfo,
    value_ranges: Option<&'a FunctionValueRanges>,
}

impl<'a> LoweringContext<'a> {
    pub fn new(types: &'a FunctionTypeInfo, value_ranges: Option<&'a FunctionValueRanges>) -> Self {
        Self {
            types,
            value_ranges,
        }
    }

    pub fn types(&self) -> &'a FunctionTypeInfo {
        self.types
    }

    pub fn range(&self, value: crate::compiler::ssa::ValueId) -> IntInterval {
        self.value_ranges
            .map(|ranges| ranges.get(value))
            .unwrap_or_else(IntInterval::top)
    }
}

pub(super) trait AlgebraicLoweringRule {
    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool;
}

impl AlgebraicLowering {
    pub fn new() -> Self {
        Self::with_lowerers(
            "algebraic_lowering",
            vec![
                Box::new(LowerPureGuards::new()),
                Box::new(LowerWitnessArrayOps::new()),
                Box::new(LowerWitnessIntegerArithOps::new()),
                Box::new(LowerWitnessBitwiseOps::new()),
                Box::new(LowerWitnessSpreadOps::new()),
                Box::new(LowerBitRangeOps::new()),
            ],
        )
    }

    pub fn pure_guards_only() -> Self {
        Self::with_lowerers(
            "algebraic_lowering_pure_guards",
            vec![Box::new(LowerPureGuards::new())],
        )
    }

    fn with_lowerers(name: &'static str, lowerers: Vec<Box<dyn AlgebraicLoweringRule>>) -> Self {
        Self { name, lowerers }
    }

    fn run_iteration(&self, ssa: &mut HLSSA) -> bool {
        let flow = FlowAnalysis::run(ssa);
        let types = Types::new().run(ssa, &flow);
        let value_ranges = ValueRangeAnalysis::new().run(ssa, &flow, &types);

        let function_ids: Vec<_> = ssa.get_function_ids().collect();
        let mut changed = false;
        let mut sb = HLSSABuilder::new(ssa);
        for function_id in function_ids {
            let function_type_info = types.get_function(function_id);
            let function_value_ranges = value_ranges.get_function(function_id);
            sb.modify_function(function_id, |fb| {
                changed |=
                    self.run_on_function(fb, function_type_info, Some(function_value_ranges));
            });
        }
        changed
    }

    fn run_on_function(
        &self,
        fb: &mut HLFunctionBuilder<'_>,
        function_type_info: &FunctionTypeInfo,
        function_value_ranges: Option<&FunctionValueRanges>,
    ) -> bool {
        let context = LoweringContext::new(function_type_info, function_value_ranges);
        let mut changed = false;
        let block_ids: Vec<_> = fb.function.get_blocks().map(|(bid, _)| *bid).collect();
        for block_id in block_ids {
            let (instructions, terminator) = {
                let mut block = fb.function.take_block(block_id);
                let instructions = block.take_instructions();
                let terminator = block.take_terminator();
                fb.function.put_block(block_id, block);
                (instructions, terminator)
            };

            let mut b = fb.block(block_id);
            for instruction in instructions {
                if self.try_lower_instruction(&mut b, &context, &instruction) {
                    changed = true;
                } else {
                    b.emit(instruction);
                }
            }
            if let Some(terminator) = terminator {
                b.set_terminator(terminator);
            }
        }
        changed
    }

    fn try_lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        self.lowerers
            .iter()
            .any(|lowerer| lowerer.lower_instruction(b, context, instruction))
    }
}

impl Pass for AlgebraicLowering {
    fn name(&self) -> &'static str {
        self.name
    }

    fn run(&self, ssa: &mut HLSSA, _store: &AnalysisStore) {
        for _ in 0..ITERATION_LIMIT {
            if !self.run_iteration(ssa) {
                return;
            }
        }
        panic!("algebraic lowering did not reach a fixed point");
    }

    fn preserves(&self) -> Vec<AnalysisId> {
        vec![]
    }
}
