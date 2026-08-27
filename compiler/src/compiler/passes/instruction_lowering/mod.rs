mod bit_range;
mod degree_spilling;
mod guards;
mod pure_guards;
mod side_effect_free_guards;
mod slice_insert_remove;
mod slice_pop;
mod slice_select;
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
        value_range_analysis::{
            FunctionValueRanges, Interval, ValueRange, ValueRangeAnalysis, Width,
        },
    },
    pass_manager::{AnalysisId, AnalysisStore, Pass},
    ssa::{
        BlockId, ValueId,
        hlssa::{
            HLSSA, OpCode, Type, TypeExpr,
            builder::{HLBlockEmitter, HLEmitter, HLFunctionBuilder, HLSSABuilder},
        },
    },
};

use self::{
    bit_range::LowerBitRangeOps,
    degree_spilling::LowerDegreeSpillingOps,
    guards::LowerGuards,
    pure_guards::LowerPureGuards,
    side_effect_free_guards::LowerSideEffectFreeGuards,
    slice_insert_remove::{LowerSliceInsert, LowerSliceRemove},
    slice_pop::LowerSlicePop,
    slice_select::LowerSliceSelect,
    witness_array::LowerWitnessArrayOps,
    witness_assert::LowerWitnessAssertOps,
    witness_bitwise::LowerWitnessBitwiseOps,
    witness_compare::LowerWitnessCompareOps,
    witness_field::LowerWitnessFieldOps,
    witness_integer_arith::LowerWitnessIntegerArithOps,
    witness_memory::LowerWitnessMemoryOps,
    witness_spread::LowerWitnessSpreadOps,
};

const ITERATION_LIMIT: usize = 32;

// INSTRUCTION LOWERING
// ================================================================================================

pub struct InstructionLowering {
    name: &'static str,
    lowerers: Vec<Box<dyn InstructionLoweringRule>>,
    fixed_point: bool,

    /// Whether any of `lowerers` reads a value range; see
    /// [`InstructionLoweringRule::needs_value_ranges`].
    needs_value_ranges: bool,

    /// The name every instruction this pass emits is scoped under, which is what a source location
    /// in the debug sidecar attributes it to.
    ///
    /// Separate from [`Self::name`] because these passes share one: everything that has always run
    /// under this driver is scoped `instruction_lowering` collectively, and moving one of them
    /// under a per-pass scope would relabel every location it emits.
    location_scope: &'static str,
}

pub(super) struct LoweringContext<'a> {
    types: &'a FunctionTypeInfo,
    value_ranges: Option<&'a FunctionValueRanges>,
    /// The block whose instructions are being lowered, when the driver has entered one.
    ///
    /// [`Self::range`] narrows through it, so a rule that asks about an operand of the instruction
    /// it is lowering gets the range that holds _here_ rather than the one that holds everywhere.
    block: Option<BlockId>,
}

impl<'a> LoweringContext<'a> {
    pub fn new(types: &'a FunctionTypeInfo, value_ranges: Option<&'a FunctionValueRanges>) -> Self {
        Self {
            types,
            value_ranges,
            block: None,
        }
    }

    /// The same context, reading ranges as they are known inside `block`.
    pub fn in_block(&self, block: BlockId) -> Self {
        Self {
            types: self.types,
            value_ranges: self.value_ranges,
            block: Some(block),
        }
    }

    pub fn types(&self) -> &'a FunctionTypeInfo {
        self.types
    }

    /// The full range record for a value, with both readings of its bit pattern.
    pub fn range(&self, value: ValueId) -> ValueRange {
        self.value_ranges
            .map(|ranges| match self.block {
                Some(block) => ranges.get_at(block, value),
                None => ranges.get(value),
            })
            .unwrap_or_else(|| ValueRange::full(Width::NonScalar))
    }

    /// The **unsigned** reading: the raw bit pattern as a non-negative integer.
    ///
    /// This is what a rule wants whenever it is about to `cast_to_field` the value and do field
    /// arithmetic.
    pub fn urange(&self, value: ValueId) -> Interval {
        self.range(value).unsigned().clone()
    }

    /// The **signed** reading: the mathematical value of a two's-complement integer.
    pub fn srange(&self, value: ValueId) -> Interval {
        self.range(value).signed().clone()
    }
}

/// The bit width and signedness of an integer type, ignoring any `WitnessOf` wrapper, or `None`
/// if the type is not an integer.
///
/// Signedness is not among the answers: it belongs to the operation, and a rule that needs it
/// takes it from its opcode.
pub(super) fn integer_bits(ty: &Type) -> Option<usize> {
    match ty.strip_witness().expr {
        TypeExpr::Int(bits) => Some(bits),
        _ => None,
    }
}

pub(super) trait InstructionLoweringRule {
    /// Whether this rule reads the [`LoweringContext`]'s value ranges.
    ///
    /// A pass all of whose rules answer `false` skips [`ValueRangeAnalysis`] altogether, which is
    /// a whole-program fixed point it would otherwise pay for on every iteration.
    ///
    /// The default is `true` because the two mistakes are not symmetric: over-declaring costs
    /// compile time, while under-declaring makes every range read widen silently to
    /// [`ValueRange::full`] — still correct, but the rule would quietly stop discharging the
    /// checks it exists to discharge, and nothing would fail.
    fn needs_value_ranges(&self) -> bool {
        true
    }

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
                Box::new(LowerSideEffectFreeGuards::new()),
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

    pub fn slice_select() -> Self {
        Self::with_lowerers(
            "instruction_lowering_slice_select",
            vec![Box::new(LowerSliceSelect::default())],
            false,
        )
    }

    pub fn slice_ops() -> Self {
        Self::with_lowerers(
            "instruction_lowering_slice_ops",
            vec![
                Box::new(LowerSlicePop::default()),
                Box::new(LowerSliceInsert::default()),
                Box::new(LowerSliceRemove::default()),
            ],
            false,
        )
    }

    pub fn pure_guards() -> Self {
        Self::with_lowerers(
            "instruction_lowering_pure_guards",
            vec![
                Box::new(LowerSideEffectFreeGuards::new()),
                Box::new(LowerPureGuards::new()),
            ],
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

    pub fn degree_spilling() -> Self {
        Self::with_lowerers(
            "degree_spilling",
            vec![Box::new(LowerDegreeSpillingOps::new())],
            false,
        )
    }

    /// The witgen pipeline's `Guard`/`Select` lowering.
    ///
    /// The odd one out among these: every other pass here runs in `witness_spilling`, while this
    /// one runs later, in `witgen_lowering`, on the clone that becomes the witness generator. Its
    /// name stays `lower_guards` because that is what names its debug-dump directory and its
    /// `MAVROS_DUMP_PASS_SSA` key, both of which people type — and so does its **location scope**,
    /// which is what the debug sidecar attributes the instructions it emits to. Both were
    /// `lower_guards` before this rule moved under this driver, and neither is this rule's to
    /// rename.
    pub fn guards() -> Self {
        Self::with_scope(
            "lower_guards",
            "lower_guards",
            vec![Box::new(LowerGuards::new())],
            false,
        )
    }

    fn with_lowerers(
        name: &'static str,
        lowerers: Vec<Box<dyn InstructionLoweringRule>>,
        fixed_point: bool,
    ) -> Self {
        Self::with_scope(name, "instruction_lowering", lowerers, fixed_point)
    }

    fn with_scope(
        name: &'static str,
        location_scope: &'static str,
        lowerers: Vec<Box<dyn InstructionLoweringRule>>,
        fixed_point: bool,
    ) -> Self {
        let needs_value_ranges = lowerers.iter().any(|rule| rule.needs_value_ranges());
        Self {
            name,
            lowerers,
            fixed_point,
            needs_value_ranges,
            location_scope,
        }
    }

    fn run_iteration(&self, ssa: &mut HLSSA) -> bool {
        let flow = FlowAnalysis::run(ssa);
        let types = Types::new().run(ssa, &flow);
        let value_ranges = self
            .needs_value_ranges
            .then(|| ValueRangeAnalysis::new().run(ssa, &flow, &types));

        let function_ids: Vec<_> = ssa.get_function_ids().collect();
        let mut changed = false;
        let mut sb = HLSSABuilder::new(ssa);
        for function_id in function_ids {
            let function_type_info = types.get_function(function_id);
            let function_value_ranges = value_ranges
                .as_ref()
                .map(|ranges| ranges.get_function(function_id));
            sb.modify_function(function_id, |fb| {
                changed |= self.run_on_function(fb, function_type_info, function_value_ranges);
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

            // Every lowered instruction scopes its own location below; emitting outside a scope
            // is an ICE.
            let context = context.in_block(block_id);
            let mut b = fb
                .block(block_id)
                .with_scoped_source_locations(self.location_scope);
            for instruction in instructions {
                let location = instruction.location().clone();
                if b.emit_with_location(location, |b| {
                    self.try_lower_instruction(b, &context, instruction.as_ref())
                }) {
                    changed = true;
                } else {
                    b.emit_located(instruction);
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
