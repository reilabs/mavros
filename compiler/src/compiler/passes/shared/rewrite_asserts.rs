//! Shared assert-rewriting machinery for [`NormalizeAsserts`](super::normalize_asserts) and
//! [`SimplifyAsserts`](super::simplify_asserts).
//!
//! Both passes walk every function and replace each `Assert` over a comparison / `u1`-`And` with a
//! specialized chain of smaller asserts (`AssertCmp{Eq}`, `AssertCmp{Lt}`, or one assert per `And`
//! operand). They differ in exactly one place: how an *equality* is emitted. `NormalizeAsserts`
//! emits a witness-agnostic `AssertCmp{Eq}` (safe before witness typing); `SimplifyAsserts` may
//! lower a `Field`-mul equality to the R1CS-native `AssertR1C`. That single difference is injected
//! as the `emit_eq` closure, so the traversal and the recursive lowering shape live here once.

use crate::{
    collections::HashMap,
    compiler::{
        analysis::types::{FunctionTypeInfo, TypeInfo},
        ssa::{
            Instruction, Located, ValueId,
            hlssa::{BinaryArithOpKind, CmpKind, HLSSA, OpCode, TypeExpr},
        },
    },
};

// ASSERT REWRITING
// ================================================================================================

/// Rewrite the asserts of every function in `ssa`.
///
/// Every `Assert{value}` is lowered through [`lower_assert`] (using `emit_eq` for an equality).
/// When `lower_assert_cmp_eq` is set, an existing `AssertCmp{Eq}` instruction is *also* lowered
/// through `emit_eq` — this is what lets `SimplifyAsserts` turn an already-normalized equality into
/// `AssertR1C`, while `NormalizeAsserts` (which clears the flag) leaves existing `AssertCmp`s
/// untouched.
///
/// `emit_eq` determines how an asserted equality `a == b` is lowered. Given the operands plus the
/// function's instruction definition map and type info, it returns the replacement instruction
/// chain.
pub(crate) fn rewrite_asserts(
    ssa: &mut HLSSA,
    type_info: &TypeInfo,
    lower_assert_cmp_eq: bool,
    emit_eq: impl Fn(ValueId, ValueId, &HashMap<ValueId, OpCode>, &FunctionTypeInfo) -> Vec<OpCode>,
) {
    for (function_id, function) in ssa.iter_functions_mut() {
        let function_type_info = type_info.get_function(*function_id);

        let mut defs: HashMap<ValueId, OpCode> = HashMap::default();
        for (_, block) in function.get_blocks() {
            for instruction in block.get_instructions() {
                for result in instruction.get_results() {
                    defs.insert(*result, instruction.clone());
                }
            }
        }

        let mut new_blocks = HashMap::default();
        for (block_id, mut block) in function.take_blocks() {
            let mut new_instructions = Vec::new();
            for instruction in block.take_instructions().into_iter() {
                let location = instruction.location().clone();
                match instruction.payload() {
                    OpCode::Assert { value } => {
                        new_instructions.extend(
                            lower_assert(value, &defs, function_type_info, &emit_eq)
                                .into_iter()
                                .map(|instr| Located::new(instr, location.clone())),
                        );
                    }
                    OpCode::AssertCmp {
                        kind: CmpKind::Eq,
                        lhs,
                        rhs,
                    } if lower_assert_cmp_eq => {
                        new_instructions.extend(
                            emit_eq(lhs, rhs, &defs, function_type_info)
                                .into_iter()
                                .map(|instr| Located::new(instr, location.clone())),
                        );
                    }
                    other => new_instructions.push(Located::new(other, location)),
                }
            }
            block.put_instructions(new_instructions);
            new_blocks.insert(block_id, block);
        }
        function.put_blocks(new_blocks);
    }
}

/// Lower an asserted boolean into a specialized assert chain, deferring the equality case to
/// `emit_eq`.
///
/// `Assert{Cmp{Eq}}` → `emit_eq`, `Assert{Cmp{Lt}}` → `AssertCmp{Lt}`, and a `u1`-`And` assert
/// splits into one assert per operand (recursing with the same `emit_eq`). Anything else stays a
/// bare `Assert`.
///
/// `emit_eq` determines how an asserted equality `a == b` is lowered. Given the operands plus the
/// function's instruction definition map and type info, it returns the replacement instruction
/// chain.
fn lower_assert(
    value: ValueId,
    defs: &HashMap<ValueId, OpCode>,
    function_type_info: &FunctionTypeInfo,
    emit_eq: impl Fn(ValueId, ValueId, &HashMap<ValueId, OpCode>, &FunctionTypeInfo) -> Vec<OpCode>
    + Copy,
) -> Vec<OpCode> {
    match defs.get(&value) {
        Some(OpCode::Cmp {
            kind: CmpKind::Eq,
            lhs,
            rhs,
            ..
        }) => emit_eq(*lhs, *rhs, defs, function_type_info),

        Some(OpCode::Cmp {
            kind: CmpKind::Lt,
            lhs,
            rhs,
            ..
        }) => vec![OpCode::AssertCmp {
            kind: CmpKind::Lt,
            lhs: *lhs,
            rhs: *rhs,
        }],

        Some(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::And,
            result,
            lhs,
            rhs,
        }) => {
            let result_type = function_type_info.get_value_type(*result);
            match result_type.strip_witness().expr {
                TypeExpr::U(1) => {
                    let mut out = lower_assert(*lhs, defs, function_type_info, emit_eq);
                    out.extend(lower_assert(*rhs, defs, function_type_info, emit_eq));
                    out
                }
                _ => vec![OpCode::Assert { value }],
            }
        }

        _ => vec![OpCode::Assert { value }],
    }
}
