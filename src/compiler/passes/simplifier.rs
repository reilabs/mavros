use std::collections::HashMap;

use ark_ff::Field as _;
use num_traits::{One, Zero};

use crate::compiler::{
    analysis::{
        types::{FunctionTypeInfo, TypeInfo},
        value_definitions::{FunctionValueDefinitions, ValueDefinition, ValueDefinitions},
    },
    flow_analysis::FlowAnalysis,
    ir::r#type::TypeExpr,
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    passes::fix_double_jumps::ValueReplacements,
    ssa::{
        BinaryArithOpKind, CastTarget, CmpKind, ConstValue, HLFunction, HLSSA, OpCode, ValueId,
    },
};

pub struct Simplifier {}

impl Pass for Simplifier {
    fn name(&self) -> &'static str {
        "simplifier"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![TypeInfo::id(), ValueDefinitions::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        self.do_run(
            ssa,
            store.get::<TypeInfo>(),
            store.get::<ValueDefinitions>(),
        );
    }

    fn preserves(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }
}

/// What to do with a single instruction during simplification.
enum Rewrite {
    /// Drop the instruction; its result becomes an alias of `target`.
    Alias { result: ValueId, target: ValueId },
    /// Replace the instruction with the given opcode (same `result` ValueId).
    Replace(OpCode),
    /// Replace the instruction with a sequence of opcodes (used for the
    /// Rangecheck → Lt+AssertEq lowering, which needs fresh result IDs).
    ReplaceMany(Vec<OpCode>),
}

impl Simplifier {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(
        &self,
        ssa: &mut HLSSA,
        type_info: &TypeInfo,
        _value_definitions: &ValueDefinitions,
    ) {
        for (function_id, function) in ssa.iter_functions_mut() {
            let function_type_info = type_info.get_function(*function_id);
            // Iterate to fixed point. Each round may expose new opportunities
            // because the alias substitution rebinds operands.
            for _ in 0..16 {
                if !self.run_function(function, function_type_info) {
                    break;
                }
            }
        }
    }

    /// One iteration over a function. Returns `true` if anything changed.
    fn run_function(
        &self,
        function: &mut HLFunction,
        function_type_info: &FunctionTypeInfo,
    ) -> bool {
        let definitions = FunctionValueDefinitions::from_ssa(function);
        let mut aliases = ValueReplacements::new();
        let mut changed = false;

        let mut new_blocks = HashMap::new();
        for (bid, mut block) in function.take_blocks().into_iter() {
            let mut new_instructions = Vec::new();
            for instruction in block.take_instructions().into_iter() {
                // Apply aliases collected so far in this iteration before
                // pattern-matching, so we see up-to-date operand identities.
                let mut instruction = instruction;
                aliases.replace_inputs(&mut instruction);

                // Rangecheck rewrite needs fresh ValueIds. We took the blocks
                // out of `function` above, so it isn't borrowed here and we
                // can pass it through to allocate.
                let rangecheck_rewrite = self.try_rangecheck(
                    &instruction,
                    &definitions,
                    function_type_info,
                    function,
                );
                let rewrite = rangecheck_rewrite.or_else(|| {
                    self.try_algebraic(&instruction, &definitions, function_type_info)
                });

                match rewrite {
                    Some(Rewrite::Alias { result, target }) => {
                        aliases.insert(result, target);
                        changed = true;
                    }
                    Some(Rewrite::Replace(new_op)) => {
                        new_instructions.push(new_op);
                        changed = true;
                    }
                    Some(Rewrite::ReplaceMany(new_ops)) => {
                        new_instructions.extend(new_ops);
                        changed = true;
                    }
                    None => {
                        new_instructions.push(instruction);
                    }
                }
            }
            block.put_instructions(new_instructions);
            new_blocks.insert(bid, block);
        }
        function.put_blocks(new_blocks);

        // Apply aliases globally. Block iteration order is non-deterministic,
        // so a block processed before its predecessor sees stale operands;
        // sweep here to fix references the in-walk substitution missed.
        for (_, block) in function.get_blocks_mut() {
            for instr in block.get_instructions_mut() {
                aliases.replace_inputs(instr);
            }
            aliases.replace_terminator(block.get_terminator_mut());
        }

        changed
    }

    /// Algebraic identities: x+0 → x, x*1 → x, x*0 → 0, etc. Conservative —
    /// only fires when the result type is pure (not WitnessOf), so we never
    /// accidentally collapse a witness slot.
    fn try_algebraic(
        &self,
        instruction: &OpCode,
        defs: &FunctionValueDefinitions,
        types: &FunctionTypeInfo,
    ) -> Option<Rewrite> {
        match instruction {
            OpCode::BinaryArithOp { kind, result, lhs, rhs } => {
                if is_witness(types, *result) {
                    return None;
                }
                match kind {
                    BinaryArithOpKind::Add => {
                        if is_zero(defs, *lhs) {
                            return Some(Rewrite::Alias { result: *result, target: *rhs });
                        }
                        if is_zero(defs, *rhs) {
                            return Some(Rewrite::Alias { result: *result, target: *lhs });
                        }
                    }
                    BinaryArithOpKind::Sub => {
                        if is_zero(defs, *rhs) {
                            return Some(Rewrite::Alias { result: *result, target: *lhs });
                        }
                        if *lhs == *rhs {
                            return materialize_zero(types, *result);
                        }
                    }
                    BinaryArithOpKind::Mul => {
                        if is_zero(defs, *lhs) {
                            return Some(Rewrite::Alias { result: *result, target: *lhs });
                        }
                        if is_zero(defs, *rhs) {
                            return Some(Rewrite::Alias { result: *result, target: *rhs });
                        }
                        if is_one(defs, *lhs) {
                            return Some(Rewrite::Alias { result: *result, target: *rhs });
                        }
                        if is_one(defs, *rhs) {
                            return Some(Rewrite::Alias { result: *result, target: *lhs });
                        }
                    }
                    BinaryArithOpKind::Div => {
                        if is_one(defs, *rhs) {
                            return Some(Rewrite::Alias { result: *result, target: *lhs });
                        }
                    }
                    BinaryArithOpKind::And => {
                        if is_zero(defs, *lhs) {
                            return Some(Rewrite::Alias { result: *result, target: *lhs });
                        }
                        if is_zero(defs, *rhs) {
                            return Some(Rewrite::Alias { result: *result, target: *rhs });
                        }
                        if *lhs == *rhs {
                            return Some(Rewrite::Alias { result: *result, target: *lhs });
                        }
                    }
                    BinaryArithOpKind::Or => {
                        if is_zero(defs, *lhs) {
                            return Some(Rewrite::Alias { result: *result, target: *rhs });
                        }
                        if is_zero(defs, *rhs) {
                            return Some(Rewrite::Alias { result: *result, target: *lhs });
                        }
                        if *lhs == *rhs {
                            return Some(Rewrite::Alias { result: *result, target: *lhs });
                        }
                    }
                    BinaryArithOpKind::Xor => {
                        if is_zero(defs, *lhs) {
                            return Some(Rewrite::Alias { result: *result, target: *rhs });
                        }
                        if is_zero(defs, *rhs) {
                            return Some(Rewrite::Alias { result: *result, target: *lhs });
                        }
                        if *lhs == *rhs {
                            return materialize_zero(types, *result);
                        }
                    }
                    BinaryArithOpKind::Shl | BinaryArithOpKind::Shr => {
                        if is_zero(defs, *rhs) {
                            return Some(Rewrite::Alias { result: *result, target: *lhs });
                        }
                    }
                    BinaryArithOpKind::Mod => {}
                }
                None
            }
            OpCode::MulConst { result, const_val, var } => {
                if is_witness(types, *result) {
                    return None;
                }
                if is_zero(defs, *const_val) {
                    return Some(Rewrite::Alias { result: *result, target: *const_val });
                }
                if is_one(defs, *const_val) {
                    return Some(Rewrite::Alias { result: *result, target: *var });
                }
                None
            }
            OpCode::Cast { result, value, target } => {
                if matches!(target, CastTarget::Nop) {
                    return Some(Rewrite::Alias { result: *result, target: *value });
                }
                // cast(cast(x, T), T) → cast(x, T)
                if let ValueDefinition::Instruction(
                    _,
                    _,
                    OpCode::Cast {
                        result: _,
                        value: _,
                        target: inner_target,
                    },
                ) = defs.get_definition(*value)
                {
                    if inner_target == target {
                        return Some(Rewrite::Alias { result: *result, target: *value });
                    }
                }
                None
            }
            OpCode::Not { result, value } => {
                // ~~x → x
                if let ValueDefinition::Instruction(
                    _,
                    _,
                    OpCode::Not {
                        result: _,
                        value: inner,
                    },
                ) = defs.get_definition(*value)
                {
                    return Some(Rewrite::Alias { result: *result, target: *inner });
                }
                None
            }
            OpCode::Select { result, cond: _, if_t, if_f } => {
                if *if_t == *if_f {
                    return Some(Rewrite::Alias { result: *result, target: *if_t });
                }
                None
            }
            OpCode::Cmp { kind: CmpKind::Eq, result, lhs, rhs } if *lhs == *rhs => {
                if is_witness(types, *result) {
                    return None;
                }
                materialize_one(types, *result)
            }
            OpCode::ValueOf { result, value } => {
                // ValueOf(ValueOf(x)) → ValueOf(x)
                if let ValueDefinition::Instruction(
                    _,
                    _,
                    OpCode::ValueOf { .. },
                ) = defs.get_definition(*value)
                {
                    return Some(Rewrite::Alias { result: *result, target: *value });
                }
                // ValueOf(WriteWitness(_, hint, _)) → hint (the slot's hint
                // value IS what ValueOf strips back to). Sound by witgen
                // semantics.
                if let ValueDefinition::Instruction(
                    _,
                    _,
                    OpCode::WriteWitness {
                        result: _,
                        value: hint,
                        pinned: _,
                    },
                ) = defs.get_definition(*value)
                {
                    return Some(Rewrite::Alias { result: *result, target: *hint });
                }
                None
            }
            OpCode::WriteWitness {
                result: Some(result),
                value,
                pinned: false,
            } => {
                // WriteWitness(ValueOf(x)) → x. The new slot's hint is x's
                // value, so honest prover fills both identically; merging is
                // equivalent to the always-true constraint new == x. This
                // requires hint chains in explicit_witness gadgets to compute
                // on PURE values (value_of operands at the boundary), so the
                // collapsed expression contains no witness-typed compute.
                if let ValueDefinition::Instruction(
                    _,
                    _,
                    OpCode::ValueOf {
                        result: _,
                        value: inner,
                    },
                ) = defs.get_definition(*value)
                {
                    return Some(Rewrite::Alias { result: *result, target: *inner });
                }
                None
            }
            _ => None,
        }
    }

    /// The original peephole: `Rangecheck(Cast<v: U(s) | I(s), Field>, bits)` →
    /// `AssertCmp::Eq(Cmp::Lt(v, 2^bits), 1)`. Replaces the rangecheck with a
    /// bounded comparison on the original integer, avoiding a fresh range
    /// decomposition.
    fn try_rangecheck(
        &self,
        instruction: &OpCode,
        defs: &FunctionValueDefinitions,
        types: &FunctionTypeInfo,
        function: &mut HLFunction,
    ) -> Option<Rewrite> {
        let OpCode::Rangecheck { value: v, max_bits: bits } = instruction else {
            return None;
        };
        let v_def = defs.get_definition(*v);
        let ValueDefinition::Instruction(
            _,
            _,
            OpCode::Cast { result: _, value: inner, target: CastTarget::Field },
        ) = v_def
        else {
            return None;
        };
        let inner = *inner;
        let v_type = types.get_value_type(inner);
        if v_type.is_witness_of() {
            panic!("Rangecheck on impure value");
        }
        let s = match &v_type.expr {
            TypeExpr::U(s) | TypeExpr::I(s) => *s,
            _ => panic!("Rangecheck on a cast of a non-u value {}", v_type),
        };
        let cst = function.fresh_value();
        let t = function.fresh_value();
        let r = function.fresh_value();
        Some(Rewrite::ReplaceMany(vec![
            OpCode::Const {
                result: cst,
                value: ConstValue::U(s, 1u128 << bits),
            },
            OpCode::Const {
                result: t,
                value: ConstValue::U(1, 1),
            },
            OpCode::Cmp {
                kind: CmpKind::Lt,
                result: r,
                lhs: inner,
                rhs: cst,
            },
            OpCode::AssertCmp {
                kind: CmpKind::Eq,
                lhs: r,
                rhs: t,
            },
        ]))
    }
}

/// Type-aware materialization of the additive identity for `result`'s type.
fn materialize_zero(types: &FunctionTypeInfo, result: ValueId) -> Option<Rewrite> {
    let ty = types.get_value_type(result);
    if ty.is_witness_of() {
        return None;
    }
    let value = match &ty.expr {
        TypeExpr::U(s) => ConstValue::U(*s, 0),
        TypeExpr::I(s) => ConstValue::I(*s, 0),
        TypeExpr::Field => ConstValue::Field(ark_bn254::Fr::zero()),
        _ => return None,
    };
    Some(Rewrite::Replace(OpCode::Const { result, value }))
}

/// Type-aware materialization of the multiplicative identity for `result`'s type.
fn materialize_one(types: &FunctionTypeInfo, result: ValueId) -> Option<Rewrite> {
    let ty = types.get_value_type(result);
    if ty.is_witness_of() {
        return None;
    }
    let value = match &ty.expr {
        TypeExpr::U(s) => ConstValue::U(*s, 1),
        TypeExpr::I(s) => ConstValue::I(*s, 1),
        TypeExpr::Field => ConstValue::Field(ark_bn254::Fr::one()),
        _ => return None,
    };
    Some(Rewrite::Replace(OpCode::Const { result, value }))
}

fn is_witness(types: &FunctionTypeInfo, v: ValueId) -> bool {
    types.get_value_type(v).is_witness_of()
}

fn is_zero(defs: &FunctionValueDefinitions, v: ValueId) -> bool {
    let def = defs.get_definition(v);
    if let ValueDefinition::Instruction(_, _, OpCode::Const { value, .. }) = def {
        match value {
            ConstValue::U(_, 0) | ConstValue::I(_, 0) => true,
            ConstValue::Field(f) => f.is_zero(),
            _ => false,
        }
    } else {
        false
    }
}

fn is_one(defs: &FunctionValueDefinitions, v: ValueId) -> bool {
    let def = defs.get_definition(v);
    if let ValueDefinition::Instruction(_, _, OpCode::Const { value, .. }) = def {
        match value {
            ConstValue::U(_, 1) | ConstValue::I(_, 1) => true,
            ConstValue::Field(f) => f.is_one(),
            _ => false,
        }
    } else {
        false
    }
}
