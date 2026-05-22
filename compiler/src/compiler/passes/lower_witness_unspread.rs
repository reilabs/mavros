//! Lowers witness-tainted `Unspread` into witness hints plus ordinary `Spread` operations.
//!
//! The later `ExplicitWitness` pass lowers those `Spread` operations, including any wide word
//! spilling needed by the backend. This keeps word splitting in one place.

use std::collections::HashMap;

use ark_ff::Field as _;

use crate::compiler::{
    Field,
    analysis::{
        flow_analysis::FlowAnalysis,
        types::{FunctionTypeInfo, TypeInfo},
    },
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::{
        BlockId, ValueId,
        hlssa::{
            CastTarget, HLBlock, HLSSA, OpCode, Type, TypeExpr,
            builder::{HLEmitter, HLInstrBuilder, HLSSABuilder},
        },
    },
};

pub struct LowerWitnessUnspread {}

impl Pass for LowerWitnessUnspread {
    fn name(&self) -> &'static str {
        "lower_witness_unspread"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![TypeInfo::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        self.do_run(ssa, store.get::<TypeInfo>());
    }

    fn preserves(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }
}

impl LowerWitnessUnspread {
    pub fn new() -> Self {
        Self {}
    }

    fn do_run(&self, ssa: &mut HLSSA, type_info: &TypeInfo) {
        let fids: Vec<_> = ssa.get_function_ids().collect();
        let mut sb = HLSSABuilder::new(ssa);
        for function_id in fids {
            let function_type_info = type_info.get_function(function_id);
            sb.modify_function(function_id, |fb| {
                let mut new_blocks = HashMap::<BlockId, HLBlock>::new();
                for (bid, mut block) in fb.function.take_blocks().into_iter() {
                    let mut new_instructions = Vec::new();
                    for instruction in block.take_instructions().into_iter() {
                        let b =
                            &mut HLInstrBuilder::new(fb.function, fb.ssa, &mut new_instructions);
                        self.process_instruction(b, function_type_info, instruction);
                    }
                    block.put_instructions(new_instructions);
                    new_blocks.insert(bid, block);
                }
                fb.function.put_blocks(new_blocks);
            });
        }
    }

    fn process_instruction(
        &self,
        b: &mut HLInstrBuilder<'_>,
        function_type_info: &FunctionTypeInfo,
        instruction: OpCode,
    ) {
        match instruction {
            OpCode::Unspread {
                result_odd,
                result_even,
                value,
                bits,
            } => {
                if function_type_info.get_value_type(value).is_witness_of() {
                    self.lower_witness_unspread(
                        b,
                        function_type_info,
                        result_odd,
                        result_even,
                        value,
                        bits,
                    );
                } else {
                    b.push(OpCode::Unspread {
                        result_odd,
                        result_even,
                        value,
                        bits,
                    });
                }
            }
            other => b.push(other),
        }
    }

    fn lower_witness_unspread(
        &self,
        b: &mut HLInstrBuilder<'_>,
        function_type_info: &FunctionTypeInfo,
        result_odd: ValueId,
        result_even: ValueId,
        value: ValueId,
        bits: u8,
    ) {
        let value_pure = b.value_of(value);
        let (odd_hint, even_hint) = b.unspread(value_pure, bits);

        self.write_unspread_result(b, function_type_info, result_odd, odd_hint);
        self.write_unspread_result(b, function_type_info, result_even, even_hint);

        let odd_spread = spread_as_field(b, result_odd, bits);
        let two = b.field_const(Field::from(2));
        let two_odd_spread = b.mul(two, odd_spread);
        let value_field = b.cast_to_field(value);
        let even_spread = b.sub(value_field, two_odd_spread);

        let even_field = b.cast_to_field(result_even);
        let one = b.field_const(Field::ONE);
        b.lookup_spread(bits, even_field, even_spread, one);
    }

    fn write_unspread_result(
        &self,
        b: &mut HLInstrBuilder<'_>,
        function_type_info: &FunctionTypeInfo,
        result: ValueId,
        hint: ValueId,
    ) {
        let hint_field = b.cast_to_field(hint);
        let hint_wit = b.write_witness(hint_field);
        b.push(OpCode::Cast {
            result,
            value: hint_wit,
            target: cast_target_for_type(function_type_info.get_value_type(result)),
        });
    }
}

fn cast_target_for_type(ty: &Type) -> CastTarget {
    match ty.strip_all_witness().expr {
        TypeExpr::U(bits) => CastTarget::U(bits),
        TypeExpr::I(bits) => CastTarget::I(bits),
        other => panic!("Expected integer type for Unspread result, got {:?}", other),
    }
}

fn spread_as_field(b: &mut HLInstrBuilder<'_>, value: ValueId, bits: u8) -> ValueId {
    let spread = b.spread(value, bits);
    b.cast_to_field(spread)
}
