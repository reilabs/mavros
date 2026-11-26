//! Takes every window of a witnessed field element from one canonical byte decomposition.
//!
//! Each window of a witnessed field element lowers to that element's canonical byte decomposition.
//! This lowering is the bulk of its cost, so we do our best to avoid two windows of one element
//! paying the decomposition tax twice.
//!
//! Where an element has more than one distinct window in a block, the widest prefix they need
//! (`bit_range(v, 0, end)`) is taken once, just before the first of them, and relabelled as an
//! `end`-bit integer. Each window is then a window of that integer, bounded by a few range checks;
//! only the prefix pays for the canonical decomposition.
//!
//! Sharing stays within a block, so the prefix dominates every window it serves, and no path pays
//! for a decomposition that the unshared program would not take. The relabel is honest because the
//! prefix is cut from the canonical decomposition, so it is below `2^end`; and `end` is at most
//! [`multi_cell_int_bits`], below the modulus, so each integer window's split is unique.

use crate::{
    collections::HashMap,
    compiler::{
        analysis::{flow_analysis::FlowAnalysis, types::Types},
        pass_manager::{AnalysisId, AnalysisStore, Pass, PassResult},
        passes::shared::{field_windows::FieldWindow, limbs::multi_cell_int_bits},
        ssa::{
            Located, SourceLocation, ValueId,
            hlssa::{CastTarget, HLBlock, HLSSA, OpCode},
        },
    },
};

// FIELD DECOMPOSITION SHARING
// ================================================================================================

pub struct ShareFieldDecompositions {}

impl ShareFieldDecompositions {
    pub fn new() -> Self {
        Self {}
    }
}

impl Pass for ShareFieldDecompositions {
    fn name(&self) -> &'static str {
        "share_field_decompositions"
    }

    fn run(&self, ssa: &mut HLSSA, _store: &AnalysisStore) -> PassResult {
        let flow = FlowAnalysis::run(ssa);
        let types = Types::new().run(ssa, &flow);

        // A wider prefix is not one integer here: wide integers have already become limbs.
        let widest = multi_cell_int_bits(ssa.field());
        let function_ids: Vec<_> = ssa.get_function_ids().collect();

        for function_id in function_ids {
            let function_types = types.get_function(function_id);
            let mut groups = Vec::new();
            for (block_id, block) in ssa.get_function(function_id).get_blocks() {
                let windows = block
                    .get_instructions_with_source_locations()
                    .enumerate()
                    .filter_map(|(index, (op, location))| {
                        FieldWindow::of(op, function_types).map(|window| (index, window, location))
                    });
                groups.extend(group_by_value(windows, widest).map(|group| (*block_id, group)));
            }
            if groups.is_empty() {
                continue;
            }

            let mut fresh = ssa.fresh_values(groups.iter().map(|(_, g)| g.fresh_needed()).sum());
            let function = ssa.get_function_mut(function_id);
            for (block_id, block) in function.get_blocks_mut() {
                let in_block: Vec<_> = groups
                    .iter()
                    .filter(|(id, _)| id == block_id)
                    .map(|(_, group)| group)
                    .collect();
                if !in_block.is_empty() {
                    rewrite_block(block, &in_block, &mut fresh);
                }
            }
        }
        Ok(())
    }

    fn preserves(&self) -> Vec<AnalysisId> {
        vec![]
    }
}

// WINDOWS
// ================================================================================================

/// The windows of one element in one block that share a prefix.
struct Group {
    value: ValueId,

    /// The bits of the prefix: the end of the furthest-reaching window.
    end: usize,

    /// The index of the first window in its block, before which the prefix is taken.
    first: usize,

    /// The first window's location, which the prefix is attributed to.
    location: SourceLocation,

    /// The windows, by their index in the block.
    windows: Vec<(usize, FieldWindow)>,
}

impl Group {
    /// The fresh values the rewrite takes: the prefix and its relabel, and one per window it
    /// takes from the integer.
    fn fresh_needed(&self) -> usize {
        2 + self
            .windows
            .iter()
            .filter(|(_, window)| (window.offset, window.width) != (0, self.end))
            .count()
    }
}

/// The groups among one block's `windows`, in block order: the windows of each element with at
/// least two distinct windows whose prefix is at most `widest` bits.
fn group_by_value<'a>(
    windows: impl Iterator<Item = (usize, FieldWindow, &'a SourceLocation)>,
    widest: usize,
) -> impl Iterator<Item = Group> {
    let mut groups: Vec<Group> = Vec::new();
    let mut of_value: HashMap<ValueId, usize> = HashMap::default();
    for (index, window, location) in windows {
        let group = *of_value.entry(window.value).or_insert_with(|| {
            groups.push(Group {
                value: window.value,
                end: 0,
                first: index,
                location: location.clone(),
                windows: Vec::new(),
            });
            groups.len() - 1
        });
        let group = &mut groups[group];
        group.end = group.end.max(window.end());
        group.windows.push((index, window));
    }
    groups.into_iter().filter(move |group| {
        let (_, first) = group.windows[0];
        let distinct = group
            .windows
            .iter()
            .any(|(_, w)| (w.offset, w.width) != (first.offset, first.width));
        distinct && group.end <= widest
    })
}

// REWRITING
// ================================================================================================

/// Takes each group's prefix before its first window, and every window of the group from it.
fn rewrite_block(block: &mut HLBlock, groups: &[&Group], fresh: &mut Vec<ValueId>) {
    // Each grouped element's prefix, as a field element and as an integer, and its width.
    let mut prefix_of: HashMap<ValueId, (ValueId, ValueId, usize)> = HashMap::default();
    let mut window_group: HashMap<usize, (FieldWindow, ValueId)> = HashMap::default();
    let mut starts: HashMap<usize, Vec<&Group>> = HashMap::default();
    for group in groups {
        starts.entry(group.first).or_default().push(group);
        for (index, window) in &group.windows {
            window_group.insert(*index, (*window, group.value));
        }
    }

    let instructions = block.take_instructions();
    let mut rewritten = Vec::with_capacity(instructions.len());
    for (index, instruction) in instructions.into_iter().enumerate() {
        for group in starts.get(&index).into_iter().flatten() {
            let (field, int) = (fresh.pop().unwrap(), fresh.pop().unwrap());
            for op in [
                OpCode::BitRange {
                    result: field,
                    value: group.value,
                    offset: 0,
                    width: group.end,
                },
                OpCode::Cast {
                    result: int,
                    value: field,
                    target: CastTarget::Int(group.end),
                },
            ] {
                rewritten.push(Located::new(op, group.location.clone()));
            }
            prefix_of.insert(group.value, (field, int, group.end));
        }

        let Some(&(window, value)) = window_group.get(&index) else {
            rewritten.push(instruction);
            continue;
        };
        let location = instruction.location().clone();
        let (field, int, end) = prefix_of[&value];
        if (window.offset, window.width) == (0, end) {
            rewritten.push(Located::new(
                OpCode::Cast {
                    result: window.result,
                    value: field,
                    target: CastTarget::Nop,
                },
                location,
            ));
            continue;
        }
        let within = fresh
            .pop()
            .expect("a fresh value for every window taken from the integer");
        for op in [
            OpCode::BitRange {
                result: within,
                value: int,
                offset: window.offset,
                width: window.width,
            },
            OpCode::Cast {
                result: window.result,
                value: within,
                target: CastTarget::Field,
            },
        ] {
            rewritten.push(Located::new(op, location.clone()));
        }
    }
    block.put_instructions(rewritten);
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use mavros_artifacts::{Bn254, FieldConfig, Goldilocks, InputValueOrdered, MavrosField};

    use super::*;
    use crate::{
        collections::HashSet,
        compiler::ssa::{
            Instruction,
            hlssa::{
                Constant, Type,
                builder::{HLEmitter, HLSSABuilder},
            },
        },
        forgery::{accepts_the_forgery, accepts_the_forgery_checked_at, all_ops},
    };

    /// `main(x: WitnessOf<Field>)` returning `bit_range(x, offset, width)` for each window, in one
    /// block.
    fn windows_of_a_witness(field: FieldConfig, windows: &[(usize, usize)]) -> (HLSSA, ValueId) {
        let mut ssa = HLSSA::with_main(field, "main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let x = HLSSABuilder::new(&mut ssa).modify_function(main_id, |b| {
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
            x
        });
        (ssa, x)
    }

    fn run(ssa: &mut HLSSA) {
        ShareFieldDecompositions::new()
            .run(ssa, &AnalysisStore::new())
            .unwrap();
    }

    /// The windows of `ssa` as `(operand, offset, width)`.
    fn windows(ssa: &HLSSA) -> Vec<(ValueId, usize, usize)> {
        all_ops(ssa)
            .filter_map(|op| match op {
                OpCode::BitRange {
                    value,
                    offset,
                    width,
                    ..
                } => Some((value, offset, width)),
                _ => None,
            })
            .collect()
    }

    /// The value each instruction result of `ssa` is relabelled from by an `Int` cast.
    fn int_relabels(ssa: &HLSSA) -> HashMap<ValueId, ValueId> {
        all_ops(ssa)
            .filter_map(|op| match op {
                OpCode::Cast {
                    result,
                    value,
                    target: CastTarget::Int(_),
                } => Some((result, value)),
                _ => None,
            })
            .collect()
    }

    /// Every operand an instruction of `ssa` reads that another instruction defines is defined
    /// earlier in the same block.
    fn assert_defined_before_use(ssa: &HLSSA) {
        let defined_anywhere: HashSet<ValueId> = all_ops(ssa)
            .flat_map(|op| op.get_results().copied().collect::<Vec<_>>())
            .collect();
        for (_, function) in ssa.iter_functions() {
            for (_, block) in function.get_blocks() {
                let mut defined: HashSet<ValueId> = HashSet::default();
                for op in block.get_instructions() {
                    for input in op.get_inputs() {
                        assert!(
                            !defined_anywhere.contains(input) || defined.contains(input),
                            "{input:?} is read before it is defined: {op:?}"
                        );
                    }
                    defined.extend(op.get_results().copied());
                }
            }
        }
    }

    /// Three windows of one element leave one window of it, the prefix they all lie in. The widest
    /// window is the prefix itself, and the other two are windows of the prefix relabelled as an
    /// integer.
    #[test]
    fn the_windows_of_an_element_share_one_decomposition() {
        let (mut ssa, x) = windows_of_a_witness(FieldConfig::bn254(), &[(0, 8), (0, 16), (4, 8)]);
        run(&mut ssa);

        let windows = windows(&ssa);
        let of_x: Vec<_> = windows.iter().filter(|(value, ..)| *value == x).collect();
        assert_eq!(of_x, [&(x, 0, 16)], "{windows:?}");
        let relabels = int_relabels(&ssa);
        let mut of_the_prefix: Vec<_> = windows
            .iter()
            .filter(|(value, ..)| *value != x)
            .map(|&(value, offset, width)| {
                assert!(
                    relabels.contains_key(&value),
                    "a window of the prefix reads its integer relabel"
                );
                (offset, width)
            })
            .collect();
        of_the_prefix.sort_unstable();
        assert_eq!(of_the_prefix, [(0, 8), (4, 8)]);
        assert!(
            all_ops(&ssa).any(|op| matches!(
                op,
                OpCode::Cast {
                    target: CastTarget::Nop,
                    ..
                }
            )),
            "the widest window is the prefix"
        );
        assert_defined_before_use(&ssa);
    }

    /// One window, or the same window twice, has nothing to share; nor does a prefix too wide to
    /// be one integer.
    #[test]
    fn a_lone_or_repeated_window_or_a_prefix_wider_than_an_integer_is_left_alone() {
        let field = FieldConfig::goldilocks();
        let widest = multi_cell_int_bits(field);
        for windows_taken in [
            vec![(0, 8)],
            vec![(0, 8), (0, 8)],
            vec![(0, 8), (0, widest + 1)],
        ] {
            let (mut ssa, _) = windows_of_a_witness(field, &windows_taken);
            let before = windows(&ssa);
            run(&mut ssa);
            assert_eq!(windows(&ssa), before, "{windows_taken:?}");
        }
    }

    /// Windows of one element in two blocks each keep a decomposition: a prefix taken in one
    /// would be paid for on paths that never reach the other.
    #[test]
    fn windows_in_different_blocks_are_not_shared() {
        let mut ssa = HLSSA::with_main(FieldConfig::bn254(), "main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        HLSSABuilder::new(&mut ssa).modify_function(main_id, |b| {
            b.function.add_return_type(Type::witness_of(Type::field()));
            b.function.add_return_type(Type::witness_of(Type::field()));
            let entry = b.function.get_entry_id();
            let next = b.function.add_block();
            let (x, low) = {
                let mut e = b.block(entry).with_source_location(SourceLocation::test());
                let x = e.add_parameter(Type::witness_of(Type::field()));
                let low = e.bit_range(x, 0, 8);
                e.terminate_jmp(next, vec![]);
                (x, low)
            };
            let mut e = b.block(next).with_source_location(SourceLocation::test());
            let wide = e.bit_range(x, 0, 16);
            e.terminate_return(vec![low, wide]);
        });
        let before = windows(&ssa);
        run(&mut ssa);
        assert_eq!(windows(&ssa), before);
    }

    /// An element an instruction defines takes its prefix after that instruction, and a guarded
    /// window is taken from the prefix unguarded.
    #[test]
    fn a_computed_element_and_a_guarded_window_share_the_prefix() {
        let mut ssa = HLSSA::with_main(FieldConfig::bn254(), "main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let (guarded, sum) = (ssa.fresh_value(), ssa.fresh_value());
        HLSSABuilder::new(&mut ssa).modify_function(main_id, |b| {
            b.function.add_return_type(Type::witness_of(Type::field()));
            b.function.add_return_type(Type::witness_of(Type::field()));
            let entry = b.function.get_entry_id();
            let mut e = b.block(entry).with_source_location(SourceLocation::test());
            let x = e.add_parameter(Type::witness_of(Type::field()));
            let condition = e.add_parameter(Type::bool());
            e.emit(OpCode::BinaryArithOp {
                kind: crate::compiler::ssa::hlssa::BinaryArithOpKind::UAdd,
                result: sum,
                lhs: x,
                rhs: x,
            });
            let low = e.bit_range(sum, 0, 8);
            e.emit(OpCode::Guard {
                condition,
                inner: Box::new(OpCode::BitRange {
                    result: guarded,
                    value: sum,
                    offset: 0,
                    width: 16,
                }),
            });
            e.terminate_return(vec![low, guarded]);
        });
        run(&mut ssa);

        assert!(
            !all_ops(&ssa).any(|op| matches!(op, OpCode::Guard { .. })),
            "the guarded window is taken from the prefix"
        );
        let of_sum: Vec<_> = windows(&ssa)
            .into_iter()
            .filter(|(value, ..)| *value == sum)
            .collect();
        assert_eq!(of_sum, [(sum, 0, 16)]);
        assert_defined_before_use(&ssa);
    }

    /// An element that is itself a shared window is shared in turn, from a prefix taken after it
    /// is defined.
    #[test]
    fn a_window_of_a_shared_window_is_shared_in_turn() {
        let mut ssa = HLSSA::with_main(FieldConfig::bn254(), "main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        HLSSABuilder::new(&mut ssa).modify_function(main_id, |b| {
            for _ in 0..3 {
                b.function.add_return_type(Type::witness_of(Type::field()));
            }
            let entry = b.function.get_entry_id();
            let mut e = b.block(entry).with_source_location(SourceLocation::test());
            let x = e.add_parameter(Type::witness_of(Type::field()));
            let y = e.bit_range(x, 0, 64);
            let low = e.bit_range(y, 0, 8);
            let mid = e.bit_range(y, 0, 16);
            let half = e.bit_range(x, 0, 32);
            e.terminate_return(vec![low, mid, half]);
        });
        run(&mut ssa);

        assert_defined_before_use(&ssa);
        let field_windows = windows(&ssa)
            .into_iter()
            .filter(|(value, ..)| !int_relabels(&ssa).contains_key(value))
            .count();
        assert_eq!(field_windows, 2, "one prefix of `x` and one of `y`");
    }

    /// A window taken from the prefix is held to its bits: moving 256 from its high part into it
    /// keeps the integer they spell, so only the window's range check refuses it.
    #[test]
    fn a_window_of_the_prefix_is_held_to_its_width() {
        let field = FieldConfig::bn254();
        let narrow = SourceLocation::synthetic("narrow");
        let mut ssa = HLSSA::with_main(field, "main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        HLSSABuilder::new(&mut ssa).modify_function(main_id, |b| {
            b.function.add_return_type(Type::field());
            b.function.add_return_type(Type::field());
            let entry = b.function.get_entry_id();
            let (x, wide) = {
                let mut e = b.block(entry).with_source_location(SourceLocation::test());
                let x = e.add_parameter(Type::field());
                (x, e.bit_range(x, 0, 16))
            };
            let mut e = b.block(entry).with_source_location(narrow.clone());
            let low = e.bit_range(x, 0, 8);
            e.terminate_return(vec![wide, low]);
        });

        // `0x1234`'s low byte and the byte above it, as the window and its high part.
        let accepts = |window: u64, high: u64| {
            let input = InputValueOrdered::Field(Bn254::from_element(field.constant(0x1234u64)));
            accepts_the_forgery::<Bn254>(ssa.clone(), &[input], |ssa| {
                let forged =
                    [window, high].map(|v| ssa.add_const(Constant::Field(field.constant(v))));
                let mut writes = 0;
                for (_, function) in ssa.iter_functions_mut() {
                    for (_, block) in function.get_blocks_mut() {
                        let mut instructions = block.take_instructions();
                        for instruction in &mut instructions {
                            if *instruction.location() != narrow {
                                continue;
                            }
                            if let OpCode::WriteWitness { value, .. } = &mut **instruction {
                                *value = forged[writes];
                                writes += 1;
                            }
                        }
                        block.put_instructions(instructions);
                    }
                }
                assert_eq!(
                    writes, 2,
                    "the window and its high part are the window's hints"
                );
            })
        };
        assert!(accepts(0x34, 0x12), "the honest hints are refused");
        assert!(!accepts(0x134, 0x11), "a window of 0x134 is accepted");
    }

    /// A window inside the prefix is held to its bits by the parts below and above it too. With
    /// `x = 0x1234`, the window `(4, 8)` of the prefix `(0, 16)` is `0x23`, over `0x4` and under
    /// `0x1`. Moving 16 from the window into the low part keeps the integer they spell, so only
    /// the low part's range check refuses it; a window of `0x24` leaves the high part a residue
    /// no 4 bits hold, so only the high part's range check does.
    #[test]
    fn a_window_inside_the_prefix_is_held_by_its_low_and_high_parts() {
        fn check<F: MavrosField>() {
            let field = F::CONFIG;
            let narrow = SourceLocation::synthetic("narrow");
            let mut ssa = HLSSA::with_main(field, "main".to_string());
            let main_id = ssa.get_unique_entrypoint_id();
            HLSSABuilder::new(&mut ssa).modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let (x, wide) = {
                    let mut e = b.block(entry).with_source_location(SourceLocation::test());
                    let x = e.add_parameter(Type::field());
                    (x, e.bit_range(x, 0, 16))
                };
                let mut e = b.block(entry).with_source_location(narrow.clone());
                let middle = e.bit_range(x, 4, 8);
                e.terminate_return(vec![wide, middle]);
            });

            let constant = |v: u64| field.constant(v);
            // The window's hints, in the order its lowering writes them: the window, then its low
            // and high parts. Its range checks are too narrow to spill, so they trap at `narrow`,
            // where nothing else can.
            let accepts = |hints: [crate::compiler::Field; 3]| {
                let input = InputValueOrdered::Field(F::from_element(constant(0x1234)));
                accepts_the_forgery_checked_at::<F>(ssa.clone(), &[input], Some("narrow"), |ssa| {
                    let forged = hints.map(|v| ssa.add_const(Constant::Field(v)));
                    let mut writes = 0;
                    for (_, function) in ssa.iter_functions_mut() {
                        for (_, block) in function.get_blocks_mut() {
                            let mut instructions = block.take_instructions();
                            for instruction in &mut instructions {
                                if *instruction.location() != narrow {
                                    continue;
                                }
                                if let OpCode::WriteWitness { value, .. } = &mut **instruction {
                                    *value = forged[writes];
                                    writes += 1;
                                }
                            }
                            block.put_instructions(instructions);
                        }
                    }
                    assert_eq!(writes, 3, "the window and its two parts are its hints");
                })
            };
            let at = format!("{:?}", F::ID);
            assert!(
                accepts([constant(0x23), constant(0x4), constant(0x1)]),
                "{at}: the honest hints are refused"
            );
            assert!(
                !accepts([constant(0x22), constant(0x14), constant(0x1)]),
                "{at}: a low part of 0x14 is accepted"
            );
            let residue = (constant(0x1234) - constant(0x4) - constant(0x24) * field.two_pow(4))
                * field.two_pow(12).inverse().unwrap();
            assert!(
                !accepts([constant(0x24), constant(0x4), residue]),
                "{at}: a window of 0x24 is accepted"
            );
        }
        check::<Bn254>();
        check::<Goldilocks>();
    }
}
