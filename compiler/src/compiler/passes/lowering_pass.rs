use crate::compiler::{
    analysis::{
        types::{FunctionTypeInfo, TypeInfo},
        value_range_analysis::{FunctionValueRanges, IntInterval, ValueRanges},
    },
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::hlssa::{
        HLSSA, OpCode,
        builder::{HLBlockEmitter, HLFunctionBuilder, HLSSABuilder},
    },
};

pub struct LoweringContext<'a> {
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

    pub fn value_ranges(&self) -> Option<&'a FunctionValueRanges> {
        self.value_ranges
    }

    pub fn range(&self, value: crate::compiler::ssa::ValueId) -> IntInterval {
        self.value_ranges
            .map(|ranges| ranges.get(value))
            .unwrap_or_else(IntInterval::top)
    }

    pub fn try_range(&self, value: crate::compiler::ssa::ValueId) -> Option<&'a IntInterval> {
        self.value_ranges.and_then(|ranges| ranges.try_get(value))
    }
}

pub trait LoweringPass {
    const NAME: &'static str;

    fn needs_value_ranges(&self) -> bool {
        false
    }

    fn preserved_analyses(&self) -> Vec<AnalysisId> {
        vec![]
    }

    fn process_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: OpCode,
    );
}

impl<T: LoweringPass> Pass for T {
    fn name(&self) -> &'static str {
        T::NAME
    }

    fn needs(&self) -> Vec<AnalysisId> {
        let mut needs = vec![TypeInfo::id()];
        if self.needs_value_ranges() {
            needs.push(ValueRanges::id());
        }
        needs
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        run_lowering_pass(
            self,
            ssa,
            store.get::<TypeInfo>(),
            store.try_get::<ValueRanges>(),
        );
    }

    fn preserves(&self) -> Vec<AnalysisId> {
        self.preserved_analyses()
    }
}

fn run_lowering_pass<T: LoweringPass + ?Sized>(
    pass: &T,
    ssa: &mut HLSSA,
    type_info: &TypeInfo,
    value_ranges: Option<&ValueRanges>,
) {
    let function_ids: Vec<_> = ssa.get_function_ids().collect();
    let mut sb = HLSSABuilder::new(ssa);
    for function_id in function_ids {
        let function_type_info = type_info.get_function(function_id);
        let function_value_ranges = value_ranges.map(|ranges| ranges.get_function(function_id));
        sb.modify_function(function_id, |fb| {
            run_on_function(pass, fb, function_type_info, function_value_ranges);
        });
    }
}

fn run_on_function<T: LoweringPass + ?Sized>(
    pass: &T,
    fb: &mut HLFunctionBuilder<'_>,
    function_type_info: &FunctionTypeInfo,
    function_value_ranges: Option<&FunctionValueRanges>,
) {
    let context = LoweringContext::new(function_type_info, function_value_ranges);
    let block_ids: Vec<_> = fb.function.get_blocks().map(|(bid, _)| *bid).collect();
    for block_id in block_ids {
        let (instructions, terminator) = {
            let mut block = fb.function.take_block(block_id);
            let instructions = block.take_instructions();
            let terminator = block.take_terminator();
            fb.function.put_block(block_id, block);
            (instructions, terminator)
        };

        // Every rewritten instruction scopes its own location below; emitting outside a scope
        // is an ICE.
        let mut b = fb.block(block_id).with_scoped_source_locations(T::NAME);
        for instruction in instructions {
            let location = instruction.location().clone();
            b.emit_with_location(location, |b| {
                pass.process_instruction(b, &context, instruction.payload());
            });
        }
        if let Some(terminator) = terminator {
            b.set_terminator(terminator);
        }
    }
}
