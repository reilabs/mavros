use crate::compiler::{
    analysis::types::{FunctionTypeInfo, TypeInfo},
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::hlssa::{
        HLSSA, OpCode,
        builder::{HLBlockEmitter, HLFunctionBuilder, HLSSABuilder},
    },
};

pub trait LoweringPass {
    const NAME: &'static str;

    fn preserved_analyses(&self) -> Vec<AnalysisId> {
        vec![]
    }

    fn process_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        function_type_info: &FunctionTypeInfo,
        instruction: OpCode,
    );
}

impl<T: LoweringPass> Pass for T {
    fn name(&self) -> &'static str {
        T::NAME
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![TypeInfo::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        run_lowering_pass(self, ssa, store.get::<TypeInfo>());
    }

    fn preserves(&self) -> Vec<AnalysisId> {
        self.preserved_analyses()
    }
}

fn run_lowering_pass<T: LoweringPass + ?Sized>(pass: &T, ssa: &mut HLSSA, type_info: &TypeInfo) {
    let function_ids: Vec<_> = ssa.get_function_ids().collect();
    let mut sb = HLSSABuilder::new(ssa);
    for function_id in function_ids {
        let function_type_info = type_info.get_function(function_id);
        sb.modify_function(function_id, |fb| {
            run_on_function(pass, fb, function_type_info);
        });
    }
}

fn run_on_function<T: LoweringPass + ?Sized>(
    pass: &T,
    fb: &mut HLFunctionBuilder<'_>,
    function_type_info: &FunctionTypeInfo,
) {
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
            pass.process_instruction(&mut b, function_type_info, instruction);
        }
        if let Some(terminator) = terminator {
            b.set_terminator(terminator);
        }
    }
}
