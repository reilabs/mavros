//! Preserve the source field decomposition's fit check even when its output is unused.
//!
//! Noir's field builtins take `Field` operands (`noir_stdlib/src/field/mod.nr`). Witness
//! decompositions constrain recomposition already; pure field decompositions need a guarded
//! range check before later lowerings turn them into raw hints. Integer IR decompositions
//! are not these source builtins and must not receive a field range check.
//! The standalone pass handles source bit decompositions. The frontend emits dynamic radices
//! for byte decompositions, whose fit checks are emitted by the radix lowerer after validation.

use super::{InstructionLoweringRule, LoweringContext};
use crate::compiler::ssa::{
    ValueId,
    hlssa::{
        OpCode, TypeExpr,
        builder::{HLBlockEmitter, HLEmitter},
    },
};

pub(super) struct LowerPureBitDecompositions;

impl InstructionLoweringRule for LowerPureBitDecompositions {
    fn needs_value_ranges(&self) -> bool {
        false
    }

    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        let (op, guard) = match instruction {
            OpCode::Guard { condition, inner } => (inner.as_ref(), Some(*condition)),
            op => (op, None),
        };
        let OpCode::ToBits { value, count, .. } = op else {
            return false;
        };
        if !emit_fit_check(b, context, guard, *value, *count) {
            return false;
        }
        b.emit(instruction.clone());
        true
    }
}

/// Emit the fit check for a pure field decomposition. Byte callers must first validate the
/// radix and derive the bound from it; this helper never treats an unknown radix as 256.
pub(super) fn emit_fit_check(
    b: &mut HLBlockEmitter<'_>,
    context: &LoweringContext<'_>,
    guard: Option<ValueId>,
    value: ValueId,
    max_bits: usize,
) -> bool {
    if !matches!(context.types().get_value_type(value).expr, TypeExpr::Field)
        || max_bits >= b.field().field_bit_size() as usize
    {
        return false;
    }
    // Witness decompositions already constrain recomposition. Pure ones need only a
    // runtime/constant check; inserting it after witness inference adds no circuit work.
    let check = OpCode::Rangecheck { value, max_bits };
    b.emit(match guard {
        Some(condition) => OpCode::Guard {
            condition,
            inner: Box::new(check),
        },
        None => check,
    });
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{
        pass_manager::{AnalysisStore, Pass},
        passes::instruction_lowering::InstructionLowering,
        ssa::{
            Terminator,
            hlssa::{Constant, Endianness, HLSSA, Radix, Type},
        },
    };

    #[test]
    fn checks_pure_bits_early_and_defers_radix_validation() {
        for ty in [
            Type::field(),
            Type::int(32),
            Type::witness_of(Type::field()),
        ] {
            // None is ToBits; 0 is a dynamic radix; 1 is the canonical Bytes variant.
            for radix in [None, Some(0), Some(1), Some(2), Some(256)] {
                for guarded in [false, true] {
                    let mut ssa = HLSSA::new();
                    let value = ssa.fresh_value();
                    let dynamic_radix = ssa.fresh_value();
                    let condition = ssa.fresh_value();
                    let result = ssa.fresh_value();
                    let op = match radix {
                        None => OpCode::ToBits {
                            result,
                            value,
                            endianness: Endianness::Little,
                            count: 1,
                        },
                        Some(radix) => OpCode::ToRadix {
                            result,
                            value,
                            endianness: Endianness::Little,
                            count: 1,
                            radix: match radix {
                                0 => Radix::Dyn(dynamic_radix),
                                1 => Radix::Bytes,
                                n => Radix::Dyn(ssa.add_const(Constant::int(32, n))),
                            },
                        },
                    };
                    let entry = ssa.get_unique_entrypoint_mut().get_entry_mut();
                    entry.push_parameter(value, ty.clone());
                    entry.push_parameter(dynamic_radix, Type::int(32));
                    entry.push_parameter(condition, Type::int(1));
                    entry.push_test_instruction(if guarded {
                        OpCode::Guard {
                            condition,
                            inner: Box::new(op),
                        }
                    } else {
                        op
                    });
                    entry.set_terminator(Terminator::Return(vec![]));
                    InstructionLowering::pure_bit_decompositions()
                        .run(&mut ssa, &AnalysisStore::new());
                    let checks: Vec<_> = ssa
                        .get_unique_entrypoint()
                        .get_entry()
                        .get_instructions()
                        .filter_map(|op| match op {
                            OpCode::Rangecheck { max_bits, .. } => Some((*max_bits, false)),
                            OpCode::Guard {
                                condition: guard,
                                inner,
                            } => match inner.as_ref() {
                                OpCode::Rangecheck { max_bits, .. } => {
                                    assert_eq!(*guard, condition);
                                    Some((*max_bits, true))
                                }
                                _ => None,
                            },
                            _ => None,
                        })
                        .collect();
                    let expected = ty == Type::field() && radix.is_none();
                    assert_eq!(checks.len(), usize::from(expected), "{ty:?}, {radix:?}");
                    if expected {
                        assert_eq!(checks[0], (1, guarded));
                    }
                    if ty == Type::field() && radix == Some(0) {
                        // A function parameter is checked only after the radix lowerer has
                        // established that it equals 256. The fit check keeps its guard.
                        InstructionLowering::witness_integer_ops()
                            .run(&mut ssa, &AnalysisStore::new());
                        let checks: Vec<_> = ssa
                            .get_unique_entrypoint()
                            .get_entry()
                            .get_instructions()
                            .filter_map(|op| match op {
                                OpCode::Rangecheck { max_bits, .. } => Some((*max_bits, false)),
                                OpCode::Guard {
                                    condition: guard,
                                    inner,
                                } => match inner.as_ref() {
                                    OpCode::Rangecheck { max_bits, .. } => {
                                        assert_eq!(*guard, condition);
                                        Some((*max_bits, true))
                                    }
                                    _ => None,
                                },
                                _ => None,
                            })
                            .collect();
                        if guarded {
                            // The deferred guarded check is lowered to a pure range branch
                            // asserting that an out-of-range value requires the guard to be off.
                            assert!(checks.is_empty());
                            assert!(ssa.get_unique_entrypoint().get_blocks().any(|(_, block)| {
                                block.get_instructions().any(|op| {
                                    matches!(op,
                                    OpCode::AssertCmp { lhs, .. } if *lhs == condition)
                                })
                            }));
                        } else {
                            assert_eq!(checks, vec![(8, false)]);
                        }
                    }
                }
            }
        }
    }
}
