mod bit_range;
mod pure_guards;
mod witness_array;
mod witness_assert;
mod witness_bitwise;
mod witness_compare;
mod witness_field;
mod witness_integer_arith;
mod witness_memory;
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
    witness_assert::LowerWitnessAssertOps, witness_bitwise::LowerWitnessBitwiseOps,
    witness_compare::LowerWitnessCompareOps, witness_field::LowerWitnessFieldOps,
    witness_integer_arith::LowerWitnessIntegerArithOps, witness_memory::LowerWitnessMemoryOps,
    witness_spread::LowerWitnessSpreadOps,
};

const ITERATION_LIMIT: usize = 32;

pub struct InstructionLowering {
    name: &'static str,
    lowerers: Vec<Box<dyn InstructionLoweringRule>>,
    fixed_point: bool,
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

pub(super) trait InstructionLoweringRule {
    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool;
}

impl InstructionLowering {
    pub fn witness_integer_ops() -> Self {
        Self::with_lowerers(
            "instruction_lowering_witness_integer_ops",
            vec![
                Box::new(LowerWitnessIntegerArithOps::new()),
                Box::new(LowerWitnessBitwiseOps::new()),
                Box::new(LowerWitnessSpreadOps::new()),
                Box::new(LowerBitRangeOps::new()),
                Box::new(LowerWitnessCompareOps::new()),
                Box::new(LowerWitnessAssertOps::new()),
                Box::new(LowerWitnessFieldOps::new()),
            ],
            true,
        )
    }

    pub fn pure_guards() -> Self {
        Self::with_lowerers(
            "instruction_lowering_pure_guards",
            vec![Box::new(LowerPureGuards::new())],
            false,
        )
    }

    pub fn witness_memory_ops() -> Self {
        Self::with_lowerers(
            "instruction_lowering_witness_memory_ops",
            vec![Box::new(LowerWitnessMemoryOps::new())],
            false,
        )
    }

    pub fn witness_array_access() -> Self {
        Self::with_lowerers(
            "instruction_lowering_witness_array_access",
            vec![Box::new(LowerWitnessArrayOps::new())],
            false,
        )
    }

    fn with_lowerers(
        name: &'static str,
        lowerers: Vec<Box<dyn InstructionLoweringRule>>,
        fixed_point: bool,
    ) -> Self {
        Self {
            name,
            lowerers,
            fixed_point,
        }
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
        for lowerer in &self.lowerers {
            if lowerer.lower_instruction(b, context, instruction) {
                return true;
            }
        }
        false
    }
}

impl Pass for InstructionLowering {
    fn name(&self) -> &'static str {
        self.name
    }

    fn run(&self, ssa: &mut HLSSA, _store: &AnalysisStore) {
        if !self.fixed_point {
            self.run_iteration(ssa);
            return;
        }

        for _ in 0..ITERATION_LIMIT {
            if !self.run_iteration(ssa) {
                return;
            }
        }
        panic!("instruction lowering did not reach a fixed point");
    }

    fn preserves(&self) -> Vec<AnalysisId> {
        vec![]
    }
}
