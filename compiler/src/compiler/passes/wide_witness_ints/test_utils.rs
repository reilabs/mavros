use mavros_artifacts::{FieldConfig, FieldId};

use crate::compiler::{
    analysis::{flow_analysis::FlowAnalysis, types::Types},
    pass_manager::{Analysis, AnalysisStore, Pass},
    passes::wide_witness_ints::{WideWitnessInts, limb_config::LimbConfig},
    ssa::{
        Located, ValueId,
        hlssa::{
            BinaryArithOpKind, HLSSA, OpCode, Type,
            builder::{HLEmitter, HLSSABuilder},
        },
    },
};

pub(super) fn bn254() -> FieldConfig {
    FieldConfig::bn254()
}

/// The fields the structural audits run on.
pub(super) fn fields() -> [FieldConfig; 2] {
    [bn254(), FieldConfig::goldilocks()]
}

/// Two widths past what one element of `field` holds: one with a narrow top limb, then one with
/// a full one.
pub(super) fn limbed_widths(field: FieldConfig) -> [usize; 2] {
    match field.id() {
        FieldId::Bn254 => [254, 320],
        FieldId::Goldilocks => [100, 64],
    }
}

/// Every emitted opcode of one function, in order.
pub(super) fn emitted(ssa: &HLSSA) -> Vec<OpCode> {
    let main = ssa.get_unique_entrypoint_id();
    let function = ssa.get_function(main);
    function
        .get_block(function.get_entry_id())
        .get_instructions()
        .cloned()
        .collect()
}

/// `main(lhs: WitnessOf<int(bits)>, rhs: rhs_type) -> WitnessOf<int(bits)> { lhs op rhs }`.
pub(super) fn operation_with(kind: BinaryArithOpKind, bits: usize, rhs_type: Type) -> HLSSA {
    operation_with_in(bn254(), kind, bits, rhs_type)
}

/// [`operation_with`] over `field`.
pub(super) fn operation_with_in(
    field: FieldConfig,
    kind: BinaryArithOpKind,
    bits: usize,
    rhs_type: Type,
) -> HLSSA {
    let mut ssa = HLSSA::with_main_in(field, "main".to_string());
    let main = ssa.get_unique_entrypoint_id();
    let (lhs, rhs, result) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let mut builder = HLSSABuilder::new(&mut ssa);
    builder.modify_function(main, |fb| {
        let entry = fb.function.get_entry_id();
        let block = fb.function.get_block_mut(entry);
        block.push_parameter(lhs, Type::witness_of(Type::int(bits)));
        block.push_parameter(rhs, rhs_type);
        fb.function
            .add_return_type(Type::witness_of(Type::int(bits)));
        let mut block = fb.test_block(entry);
        block.emit(OpCode::BinaryArithOp {
            kind,
            result,
            lhs,
            rhs,
        });
        block.terminate_return(vec![result]);
    });
    ssa
}

/// Wrap every instruction of `ssa`'s entry block in a guard on a new witnessed condition, which
/// is returned.
pub(super) fn guard_every_instruction(ssa: &mut HLSSA) -> ValueId {
    let main = ssa.get_unique_entrypoint_id();
    let condition = ssa.fresh_value();
    let entry = ssa.get_function_mut(main).get_entry_mut();
    entry.push_parameter(condition, Type::witness_of(Type::int(1)));
    let instructions: Vec<_> = entry
        .take_instructions()
        .into_iter()
        .map(|op| {
            let guarded = OpCode::Guard {
                condition,
                inner: Box::new(op.as_ref().clone()),
            };
            Located::new(guarded, op.location().clone())
        })
        .collect();
    entry.put_instructions(instructions);
    condition
}

pub(super) fn run_pass(ssa: &mut HLSSA) {
    let flow = FlowAnalysis::run(ssa);
    let types = Types::new().run(ssa, &flow);
    let mut store = AnalysisStore::new();
    store.insert_with_deps(flow, vec![]);
    store.insert_with_deps(types, vec![]);
    let limb_config = LimbConfig::compute(ssa, &store);
    store.insert_with_deps(limb_config, vec![]);
    WideWitnessInts::new().run(ssa, &store);
}
