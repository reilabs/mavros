//! Moves each window of a witnessed field element that the program reads in more than one place
//! into one helper per window. This ensures that its canonical byte decomposition is emitted _once_
//! rather than at every site.
//!
//! The helper holds the `BitRange` itself and is lowered with the rest of the program, so its
//! witnesses and constraints are identical to those of the inline window it replaces.

use crate::{
    collections::HashMap,
    compiler::{
        analysis::{flow_analysis::FlowAnalysis, types::Types},
        pass_manager::{AnalysisId, AnalysisStore, Pass},
        passes::shared::field_windows::FieldWindow,
        ssa::{
            FunctionId, SourceLocation,
            hlssa::{
                CallTarget, HLSSA, OpCode, Type,
                builder::{HLEmitter, HLSSABuilder},
            },
        },
    },
};

// FIELD WINDOW OUTLINING
// ================================================================================================

pub struct OutlineFieldBitRanges {}

impl OutlineFieldBitRanges {
    pub fn new() -> Self {
        Self {}
    }
}

impl Pass for OutlineFieldBitRanges {
    fn name(&self) -> &'static str {
        "outline_field_bit_ranges"
    }

    fn run(&self, ssa: &mut HLSSA, _store: &AnalysisStore) {
        let flow = FlowAnalysis::run(ssa);
        let types = Types::new().run(ssa, &flow);
        let functions: Vec<FunctionId> = ssa.get_function_ids().collect();

        // A window is keyed by its bits alone: its result takes its operand's type, a witnessed
        // `Field`.
        let mut sites: HashMap<(usize, usize), usize> = HashMap::default();
        let mut order = Vec::new();
        for &function in &functions {
            let function_types = types.get_function(function);
            for (_, block) in ssa.get_function(function).get_blocks() {
                for op in block.get_instructions() {
                    if let Some(window) = FieldWindow::of(op, function_types) {
                        let count = sites.entry((window.offset, window.width)).or_insert(0);
                        if *count == 0 {
                            order.push((window.offset, window.width));
                        }
                        *count += 1;
                    }
                }
            }
        }

        let mut sb = HLSSABuilder::new(ssa);
        let helpers: HashMap<(usize, usize), FunctionId> = order
            .into_iter()
            .filter(|key| sites[key] > 1)
            .map(|(offset, width)| {
                let name = format!("__field_bit_range_{offset}_{width}");
                let location = SourceLocation::synthetic(&name);
                let (id, ()) = sb.add_function(name, |fb| {
                    fb.function()
                        .add_return_type(Type::witness_of(Type::field()));
                    let entry = fb.function().get_entry_id();
                    let mut e = fb.block(entry).with_source_location(location);
                    let value = e.add_parameter(Type::witness_of(Type::field()));
                    let selected = e.bit_range(value, offset, width);
                    e.terminate_return(vec![selected]);
                });
                ((offset, width), id)
            })
            .collect();
        if helpers.is_empty() {
            return;
        }

        for &function in &functions {
            let function_types = types.get_function(function);
            sb.modify_function(function, |fb| {
                for (_, block) in fb.function.get_blocks_mut() {
                    for op in block.get_instructions_mut() {
                        let Some(window) = FieldWindow::of(op, function_types) else {
                            continue;
                        };
                        if let Some(&helper) = helpers.get(&(window.offset, window.width)) {
                            *op = OpCode::Call {
                                results: vec![window.result],
                                function: CallTarget::Static(helper),
                                args: vec![window.value],
                                unconstrained: false,
                            };
                        }
                    }
                }
            });
        }
    }

    fn preserves(&self) -> Vec<AnalysisId> {
        vec![]
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use mavros_artifacts::FieldConfig;

    use super::*;
    use crate::{compiler::ssa::hlssa::TypeExpr, forgery::all_ops};

    /// `main(x: WitnessOf<Field>)` returning `bit_range(x, offset, width)` for each window.
    fn windows_of_a_witness(windows: &[(usize, usize)]) -> HLSSA {
        let mut ssa = HLSSA::with_main(FieldConfig::bn254(), "main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        HLSSABuilder::new(&mut ssa).modify_function(main_id, |b| {
            for _ in windows {
                b.function.add_return_type(Type::witness_of(Type::field()));
            }
            let entry = b.function.get_entry_id();
            let mut e = b
                .block(entry)
                .with_source_location(SourceLocation::synthetic("windows"));
            let x = e.add_parameter(Type::witness_of(Type::field()));
            let selected = windows
                .iter()
                .map(|&(offset, width)| e.bit_range(x, offset, width))
                .collect();
            e.terminate_return(selected);
        });
        ssa
    }

    fn bit_ranges(ssa: &HLSSA) -> usize {
        all_ops(ssa)
            .filter(|op| matches!(op, OpCode::BitRange { .. }))
            .count()
    }

    fn calls(ssa: &HLSSA) -> usize {
        all_ops(ssa)
            .filter(|op| matches!(op, OpCode::Call { .. }))
            .count()
    }

    /// Two reads of one window call one helper, which holds the window; a window read once stays
    /// where it is.
    #[test]
    fn a_window_read_twice_is_read_through_one_helper() {
        let mut ssa = windows_of_a_witness(&[(0, 8), (0, 8), (8, 16)]);
        OutlineFieldBitRanges::new().run(&mut ssa, &AnalysisStore::new());

        let helpers: Vec<_> = ssa
            .iter_functions()
            .filter(|(_, function)| function.get_name() == "__field_bit_range_0_8")
            .collect();
        assert_eq!(helpers.len(), 1, "one helper for the window");
        assert_eq!(calls(&ssa), 2, "both reads call it");
        assert_eq!(
            bit_ranges(&ssa),
            2,
            "the helper's window and the one read once"
        );
        let (_, helper) = helpers[0];
        assert!(matches!(
            helper.get_returns()[0].strip_witness().expr,
            TypeExpr::Field
        ));
    }

    /// Sites in two functions share the helper, and a guarded window is called unguarded, with
    /// the window's operand and result.
    #[test]
    fn windows_across_functions_and_guards_share_one_helper() {
        let mut ssa = HLSSA::with_main(FieldConfig::bn254(), "main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let (guarded, x) = (ssa.fresh_value(), ssa.fresh_value());
        let mut sb = HLSSABuilder::new(&mut ssa);
        let (other, ()) = sb.add_function("other".to_string(), |fb| {
            fb.function()
                .add_return_type(Type::witness_of(Type::field()));
            let entry = fb.function().get_entry_id();
            let mut e = fb.block(entry).with_source_location(SourceLocation::test());
            let y = e.add_parameter(Type::witness_of(Type::field()));
            let low = e.bit_range(y, 0, 8);
            e.terminate_return(vec![low]);
        });
        sb.modify_function(main_id, |b| {
            b.function.add_return_type(Type::witness_of(Type::field()));
            let entry = b.function.get_entry_id();
            b.function
                .get_block_mut(entry)
                .push_parameter(x, Type::witness_of(Type::field()));
            let mut e = b.block(entry).with_source_location(SourceLocation::test());
            let condition = e.add_parameter(Type::bool());
            e.emit(OpCode::Guard {
                condition,
                inner: Box::new(OpCode::BitRange {
                    result: guarded,
                    value: x,
                    offset: 0,
                    width: 8,
                }),
            });
            let _ = e.call(other, vec![x], 1);
            e.terminate_return(vec![guarded]);
        });
        OutlineFieldBitRanges::new().run(&mut ssa, &AnalysisStore::new());

        assert!(!all_ops(&ssa).any(|op| matches!(op, OpCode::Guard { .. })));
        let helper = ssa
            .iter_functions()
            .find(|(_, function)| function.get_name() == "__field_bit_range_0_8")
            .map(|(id, _)| *id)
            .expect("one helper for the window");
        let calls: Vec<_> = all_ops(&ssa)
            .filter_map(|op| match op {
                OpCode::Call {
                    results,
                    function: CallTarget::Static(target),
                    args,
                    ..
                } if target == helper => Some((results, args)),
                _ => None,
            })
            .collect();
        assert_eq!(calls.len(), 2, "both sites call the helper");
        assert!(
            calls.contains(&(vec![guarded], vec![x])),
            "the guarded window keeps its operand and result"
        );
    }

    /// A window of a pure element, or of a witnessed integer, has no canonical decomposition to
    /// share.
    #[test]
    fn only_windows_of_a_witnessed_element_are_outlined() {
        let mut ssa = HLSSA::with_main(FieldConfig::bn254(), "main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        HLSSABuilder::new(&mut ssa).modify_function(main_id, |b| {
            for ty in [Type::field(), Type::field()] {
                b.function.add_return_type(ty);
            }
            for _ in 0..2 {
                b.function.add_return_type(Type::witness_of(Type::int(32)));
            }
            let entry = b.function.get_entry_id();
            let mut e = b
                .block(entry)
                .with_source_location(SourceLocation::synthetic("windows"));
            let pure = e.add_parameter(Type::field());
            let int = e.add_parameter(Type::witness_of(Type::int(32)));
            let selected = [pure, pure, int, int]
                .into_iter()
                .map(|value| e.bit_range(value, 0, 8))
                .collect();
            e.terminate_return(selected);
        });
        OutlineFieldBitRanges::new().run(&mut ssa, &AnalysisStore::new());

        assert_eq!(calls(&ssa), 0);
        assert_eq!(bit_ranges(&ssa), 4);
    }
}
