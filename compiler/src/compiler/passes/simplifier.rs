//! Performs both peephole optimization and algebraic simplification on the SSA IR, running until it
//! reaches an iteration limit or a fixed point.

use std::collections::HashMap;

use num_traits::{One, Zero};

use crate::compiler::{
    analysis::{
        flow_analysis::FlowAnalysis,
        types::{FunctionTypeInfo, Types},
        value_definitions::{FunctionValueDefinitions, ValueDefinition},
    },
    pass_manager::{AnalysisId, AnalysisStore, Pass},
    passes::fix_double_jumps::ValueReplacements,
    ssa::{
        FunctionId, ValueId,
        hlssa::{
            BinaryArithOpKind, CastTarget, CmpKind, ConstValue, HLSSA, OpCode, Type, TypeExpr,
            builder::{HLFunctionBuilder, HLSSABuilder},
        },
    },
};

pub struct Simplifier {
    max_iterations: usize,
}

impl Pass for Simplifier {
    fn name(&self) -> &'static str {
        "simplifier"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        self.do_run(ssa, store.get::<FlowAnalysis>());
    }

    fn preserves(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }
}

/// What to do with a single instruction during simplification.
enum Rewrite {
    /// Drop the instruction; its result becomes an alias of `target`.
    Alias { result: ValueId, target: ValueId },

    /// Replace the instruction with the given sequence (the last opcode produces the original
    /// `result` ValueId).
    Replace(Vec<OpCode>),
}

impl Simplifier {
    pub fn new() -> Self {
        Self::with_max_iterations(16)
    }

    pub fn with_max_iterations(max_iterations: usize) -> Self {
        Self { max_iterations }
    }

    pub fn do_run(&self, ssa: &mut HLSSA, flow: &FlowAnalysis) {
        // Function signatures don't change here; build the global signature
        // map once and reuse it for per-function type inference. Owned so
        // we can mutate functions afterwards without borrow conflicts.
        let owned_sigs: HashMap<FunctionId, (Vec<Type>, Vec<Type>)> = ssa
            .iter_functions()
            .map(|(id, func)| (*id, (func.get_param_types(), func.get_returns().to_vec())))
            .collect();
        let function_types: HashMap<FunctionId, (Vec<Type>, &[Type])> = owned_sigs
            .iter()
            .map(|(id, (params, returns))| (*id, (params.clone(), returns.as_slice())))
            .collect();
        let fids: Vec<_> = ssa.get_function_ids().collect();
        let mut sb = HLSSABuilder::new(ssa);
        for function_id in fids {
            let cfg = flow.get_function_cfg(function_id);
            sb.modify_function(function_id, |fb| {
                for _ in 0..self.max_iterations {
                    // Recompute locally so newly emitted opcodes (the Const+Cast
                    // pair from `materialize_const`) appear with the right types.
                    let fti = Types::new().run_function(fb.function, &function_types, cfg);
                    if !self.run_function(fb, &fti) {
                        break;
                    }
                }
            });
        }
    }

    /// One iteration over a function. Returns `true` if anything changed.
    fn run_function(
        &self,
        fb: &mut HLFunctionBuilder<'_>,
        function_type_info: &FunctionTypeInfo,
    ) -> bool {
        let definitions = FunctionValueDefinitions::from_ssa(fb.function);
        let mut aliases = ValueReplacements::new();
        let mut changed = false;

        let mut new_blocks = HashMap::new();
        for (bid, mut block) in fb.function.take_blocks().into_iter() {
            let mut new_instructions = Vec::new();
            for instruction in block.take_instructions().into_iter() {
                // Apply aliases collected so far in this iteration before
                // pattern-matching, so we see up-to-date operand identities.
                let mut instruction = instruction;
                aliases.replace_inputs(&mut instruction);

                let rewrite =
                    self.try_algebraic(&instruction, &definitions, function_type_info, fb);

                match rewrite {
                    Some(Rewrite::Alias { result, target }) => {
                        aliases.insert(result, target);
                        changed = true;
                    }
                    Some(Rewrite::Replace(new_ops)) => {
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
        fb.function.put_blocks(new_blocks);

        // Apply aliases globally. Block iteration order is non-deterministic,
        // so a block processed before its predecessor sees stale operands;
        // sweep here to fix references the in-walk substitution missed.
        for (_, block) in fb.function.get_blocks_mut() {
            for instr in block.get_instructions_mut() {
                aliases.replace_inputs(instr);
            }
            aliases.replace_terminator(block.get_terminator_mut());
        }

        changed
    }

    /// Algebraic identities: x+0 → x, x*1 → x, x*0 → 0, etc.
    fn try_algebraic(
        &self,
        instruction: &OpCode,
        defs: &FunctionValueDefinitions,
        types: &FunctionTypeInfo,
        fb: &mut HLFunctionBuilder<'_>,
    ) -> Option<Rewrite> {
        match instruction {
            OpCode::BinaryArithOp {
                kind,
                result,
                lhs,
                rhs,
            } => {
                match kind {
                    BinaryArithOpKind::Add => {
                        if is_zero(defs, *lhs) {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *rhs,
                            });
                        }
                        if is_zero(defs, *rhs) {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *lhs,
                            });
                        }
                    }
                    BinaryArithOpKind::Sub => {
                        if is_zero(defs, *rhs) {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *lhs,
                            });
                        }
                        if *lhs == *rhs {
                            return materialize_zero(types, *result, fb);
                        }
                    }
                    BinaryArithOpKind::Mul => {
                        if is_zero(defs, *lhs) {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *lhs,
                            });
                        }
                        if is_zero(defs, *rhs) {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *rhs,
                            });
                        }
                        if is_one(defs, *lhs) {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *rhs,
                            });
                        }
                        if is_one(defs, *rhs) {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *lhs,
                            });
                        }
                    }
                    BinaryArithOpKind::Div => {
                        if is_one(defs, *rhs) {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *lhs,
                            });
                        }
                    }
                    BinaryArithOpKind::And => {
                        if is_zero(defs, *lhs) {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *lhs,
                            });
                        }
                        if is_zero(defs, *rhs) {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *rhs,
                            });
                        }
                        if *lhs == *rhs {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *lhs,
                            });
                        }
                    }
                    BinaryArithOpKind::Or => {
                        if is_zero(defs, *lhs) {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *rhs,
                            });
                        }
                        if is_zero(defs, *rhs) {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *lhs,
                            });
                        }
                        if *lhs == *rhs {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *lhs,
                            });
                        }
                    }
                    BinaryArithOpKind::Xor => {
                        if is_zero(defs, *lhs) {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *rhs,
                            });
                        }
                        if is_zero(defs, *rhs) {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *lhs,
                            });
                        }
                        if *lhs == *rhs {
                            return materialize_zero(types, *result, fb);
                        }
                    }
                    BinaryArithOpKind::Shl | BinaryArithOpKind::Shr => {
                        if is_zero(defs, *rhs) {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *lhs,
                            });
                        }
                        if matches!(kind, BinaryArithOpKind::Shr) {
                            if let Some(offset) = const_as_usize(defs, *rhs) {
                                let lhs_type = types.get_value_type(*lhs);
                                let lhs_inner = lhs_type.strip_witness();
                                if let TypeExpr::U(bits) = lhs_inner.expr {
                                    if offset < bits {
                                        return Some(Rewrite::Replace(vec![OpCode::BitRange {
                                            result: *result,
                                            value: *lhs,
                                            offset,
                                            width: bits - offset,
                                        }]));
                                    }
                                }
                            }
                        }
                    }
                    BinaryArithOpKind::Mod => {}
                }
                None
            }
            OpCode::MulConst {
                result,
                const_val,
                var,
            } => {
                if is_zero(defs, *const_val) {
                    return Some(Rewrite::Alias {
                        result: *result,
                        target: *const_val,
                    });
                }
                if is_one(defs, *const_val) {
                    return Some(Rewrite::Alias {
                        result: *result,
                        target: *var,
                    });
                }
                None
            }
            OpCode::Cast {
                result,
                value,
                target,
            } => {
                if matches!(target, CastTarget::Nop) {
                    return Some(Rewrite::Alias {
                        result: *result,
                        target: *value,
                    });
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
                        return Some(Rewrite::Alias {
                            result: *result,
                            target: *value,
                        });
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
                    return Some(Rewrite::Alias {
                        result: *result,
                        target: *inner,
                    });
                }
                None
            }
            OpCode::Select {
                result,
                cond: _,
                if_t,
                if_f,
            } => {
                if *if_t == *if_f {
                    return Some(Rewrite::Alias {
                        result: *result,
                        target: *if_t,
                    });
                }
                None
            }
            OpCode::Cmp {
                kind: CmpKind::Eq,
                result,
                lhs,
                rhs,
            } if *lhs == *rhs => materialize_one(types, *result, fb),
            OpCode::ValueOf { result, value } => {
                // ValueOf(ValueOf(x)) → ValueOf(x)
                if let ValueDefinition::Instruction(_, _, OpCode::ValueOf { .. }) =
                    defs.get_definition(*value)
                {
                    return Some(Rewrite::Alias {
                        result: *result,
                        target: *value,
                    });
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
                    return Some(Rewrite::Alias {
                        result: *result,
                        target: *hint,
                    });
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
                // equivalent to the always-true constraint new == x.
                if let ValueDefinition::Instruction(
                    _,
                    _,
                    OpCode::ValueOf {
                        result: _,
                        value: inner,
                    },
                ) = defs.get_definition(*value)
                {
                    return Some(Rewrite::Alias {
                        result: *result,
                        target: *inner,
                    });
                }
                None
            }
            _ => None,
        }
    }
}

fn materialize_const(
    types: &FunctionTypeInfo,
    result: ValueId,
    fb: &mut HLFunctionBuilder<'_>,
    pure_value: impl Fn(&TypeExpr) -> Option<ConstValue>,
) -> Option<Rewrite> {
    let ty = types.get_value_type(result).clone();
    let inner = ty.strip_witness();
    let value = pure_value(&inner.expr)?;
    if ty.is_witness_of() {
        let tmp = fb.fresh_value();
        Some(Rewrite::Replace(vec![
            OpCode::Const { result: tmp, value },
            OpCode::Cast {
                result,
                value: tmp,
                target: CastTarget::WitnessOf,
            },
        ]))
    } else {
        Some(Rewrite::Replace(vec![OpCode::Const { result, value }]))
    }
}

fn materialize_zero(
    types: &FunctionTypeInfo,
    result: ValueId,
    fb: &mut HLFunctionBuilder<'_>,
) -> Option<Rewrite> {
    materialize_const(types, result, fb, |t| match t {
        TypeExpr::U(s) => Some(ConstValue::U(*s, 0)),
        TypeExpr::I(s) => Some(ConstValue::I(*s, 0)),
        TypeExpr::Field => Some(ConstValue::Field(ark_bn254::Fr::zero())),
        _ => None,
    })
}

fn materialize_one(
    types: &FunctionTypeInfo,
    result: ValueId,
    fb: &mut HLFunctionBuilder<'_>,
) -> Option<Rewrite> {
    materialize_const(types, result, fb, |t| match t {
        TypeExpr::U(s) => Some(ConstValue::U(*s, 1)),
        TypeExpr::I(s) => Some(ConstValue::I(*s, 1)),
        TypeExpr::Field => Some(ConstValue::Field(ark_bn254::Fr::one())),
        _ => None,
    })
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

fn const_as_usize(defs: &FunctionValueDefinitions, v: ValueId) -> Option<usize> {
    let def = defs.get_definition(v);
    if let ValueDefinition::Instruction(_, _, OpCode::Const { value, .. }) = def {
        match value {
            ConstValue::U(_, value) | ConstValue::I(_, value) => (*value).try_into().ok(),
            ConstValue::Field(_) | ConstValue::FnPtr(_) => None,
        }
    } else {
        None
    }
}
