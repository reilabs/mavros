//! Performs both peephole optimization and algebraic simplification on the SSA IR, running until it
//! reaches an iteration limit or a fixed point.

use ark_ff::Field as _;
use num_traits::{One, Zero};

use crate::{
    collections::HashMap,
    compiler::{
        analysis::{
            flow_analysis::FlowAnalysis,
            types::{FunctionTypeInfo, Types, const_value_type},
            value_definitions::{FunctionValueDefinitions, ValueDefinition},
        },
        pass_manager::{AnalysisId, AnalysisStore, Pass},
        passes::fix_double_jumps::{ReplaceScope, ValueReplacements},
        ssa::{
            FunctionId, Located, ValueId,
            hlssa::{
                BinaryArithOpKind, CastTarget, CmpKind, Constant, HLSSA, OpCode, Type, TypeExpr,
                builder::{HLFunctionBuilder, HLSSABuilder},
            },
        },
        util::bit_mask,
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
                    let constant_types: HashMap<ValueId, Type> = fb
                        .ssa
                        .const_snapshot()
                        .iter()
                        .map(|(vid, cv)| (*vid, const_value_type(cv)))
                        .collect();
                    let fti = Types::new().run_function(
                        fb.function,
                        &function_types,
                        &constant_types,
                        cfg,
                    );
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
        let definitions = FunctionValueDefinitions::from_function(fb.function);
        let mut aliases = ValueReplacements::new();
        let mut changed = false;

        let mut new_blocks = HashMap::default();
        for (bid, mut block) in fb.function.take_blocks().into_iter() {
            let mut new_instructions = Vec::new();
            for mut instruction in block.take_located_instructions().into_iter() {
                // Apply aliases collected so far in this iteration before pattern-matching, so we
                // see up-to-date operands.
                aliases.replace_inputs(&mut *instruction);

                let rewrite =
                    self.try_algebraic(&instruction, &definitions, function_type_info, fb);

                match rewrite {
                    Some(Rewrite::Alias { result, target }) => {
                        aliases.insert(result, target);
                        changed = true;
                    }
                    Some(Rewrite::Replace(new_ops)) => {
                        let location = instruction.location().clone();
                        new_instructions.extend(
                            new_ops
                                .into_iter()
                                .map(|new_op| Located::new(new_op, location.clone())),
                        );
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

        // Apply aliases globally. Block iteration order is arbitrary (deterministic, but not a
        // CFG order), so a block processed before its predecessor sees stale operands; sweep here
        // to fix references the in-walk substitution missed.
        aliases.apply_to_function(fb.function, ReplaceScope::Inputs);

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
                        if is_zero(fb.ssa, *lhs) {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *rhs,
                            });
                        }
                        if is_zero(fb.ssa, *rhs) {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *lhs,
                            });
                        }
                    }
                    BinaryArithOpKind::Sub => {
                        if is_zero(fb.ssa, *rhs) {
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
                        if is_zero(fb.ssa, *lhs) {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *lhs,
                            });
                        }
                        if is_zero(fb.ssa, *rhs) {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *rhs,
                            });
                        }
                        if is_one(fb.ssa, *lhs) {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *rhs,
                            });
                        }
                        if is_one(fb.ssa, *rhs) {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *lhs,
                            });
                        }
                    }
                    BinaryArithOpKind::Div => {
                        if is_one(fb.ssa, *rhs) {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *lhs,
                            });
                        }
                        let result_type = types.get_value_type(*result);
                        if result_type.strip_witness().is_field()
                            && let Some(Constant::Field(denom)) = fb.ssa.get_const(*rhs).as_deref()
                            && !denom.is_zero()
                        {
                            let inv = fb
                                .ssa
                                .add_const(Constant::Field((*denom).inverse().unwrap()));
                            return Some(Rewrite::Replace(vec![OpCode::BinaryArithOp {
                                kind: BinaryArithOpKind::Mul,
                                result: *result,
                                lhs: *lhs,
                                rhs: inv,
                            }]));
                        }
                    }
                    BinaryArithOpKind::And => {
                        if is_zero(fb.ssa, *lhs) {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *lhs,
                            });
                        }
                        if is_zero(fb.ssa, *rhs) {
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
                        if is_all_ones(fb.ssa, *lhs) {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *rhs,
                            });
                        }
                        if is_all_ones(fb.ssa, *rhs) {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *lhs,
                            });
                        }
                    }
                    BinaryArithOpKind::Or => {
                        if is_zero(fb.ssa, *lhs) {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *rhs,
                            });
                        }
                        if is_zero(fb.ssa, *rhs) {
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
                        if is_zero(fb.ssa, *lhs) {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *rhs,
                            });
                        }
                        if is_zero(fb.ssa, *rhs) {
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
                        if is_zero(fb.ssa, *rhs) {
                            return Some(Rewrite::Alias {
                                result: *result,
                                target: *lhs,
                            });
                        }
                        if matches!(kind, BinaryArithOpKind::Shr) {
                            if let Some(offset) = const_as_usize(fb.ssa, *rhs) {
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
                if is_zero(fb.ssa, *const_val) {
                    return Some(Rewrite::Alias {
                        result: *result,
                        target: *const_val,
                    });
                }
                if is_one(fb.ssa, *const_val) {
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
                if let Some(ValueDefinition::Instruction(
                    _,
                    _,
                    OpCode::Cast {
                        result: _,
                        value: _,
                        target: inner_target,
                    },
                )) = defs.get_definition(*value)
                {
                    if inner_target == target {
                        return Some(Rewrite::Alias {
                            result: *result,
                            target: *value,
                        });
                    }
                }
                // ValueOf(WriteWitness(_, hint, _)) → hint (the slot's hint
                // value IS what ValueOf strips back to). Sound by witgen
                // semantics.
                if matches!(target, CastTarget::ValueOf)
                    && let Some(ValueDefinition::Instruction(
                        _,
                        _,
                        OpCode::WriteWitness {
                            result: _,
                            value: hint,
                            pinned: _,
                        },
                    )) = defs.get_definition(*value)
                {
                    return Some(Rewrite::Alias {
                        result: *result,
                        target: *hint,
                    });
                }
                None
            }
            OpCode::Not { result, value } => {
                // ~~x → x
                if let Some(ValueDefinition::Instruction(
                    _,
                    _,
                    OpCode::Not {
                        result: _,
                        value: inner,
                    },
                )) = defs.get_definition(*value)
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
            OpCode::BitRange {
                result,
                value,
                offset,
                width,
            } => {
                if *offset == 0 && *width == types.get_value_type(*value).get_bit_size() {
                    return Some(Rewrite::Alias {
                        result: *result,
                        target: *value,
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
            OpCode::WriteWitness {
                result: Some(result),
                value,
                pinned: false,
            } => {
                // WriteWitness(ValueOf(x)) → x. The new slot's hint is x's
                // value, so honest prover fills both identically; merging is
                // equivalent to the always-true constraint new == x.
                if let Some(ValueDefinition::Instruction(
                    _,
                    _,
                    OpCode::Cast {
                        result: _,
                        value: inner,
                        target: CastTarget::ValueOf,
                    },
                )) = defs.get_definition(*value)
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
    pure_value: impl Fn(&TypeExpr) -> Option<Constant>,
) -> Option<Rewrite> {
    let ty = types.get_value_type(result).clone();
    let inner = ty.strip_witness();
    let value = pure_value(&inner.expr)?;
    if ty.is_witness_of() {
        let tmp = fb.ssa.add_const(value);
        Some(Rewrite::Replace(vec![OpCode::Cast {
            result,
            value: tmp,
            target: CastTarget::WitnessOf,
        }]))
    } else {
        let canon = fb.ssa.add_const(value);
        Some(Rewrite::Alias {
            result,
            target: canon,
        })
    }
}

fn materialize_zero(
    types: &FunctionTypeInfo,
    result: ValueId,
    fb: &mut HLFunctionBuilder<'_>,
) -> Option<Rewrite> {
    materialize_const(types, result, fb, |t| match t {
        TypeExpr::U(s) => Some(Constant::U(*s, 0)),
        TypeExpr::I(s) => Some(Constant::I(*s, 0)),
        TypeExpr::Field => Some(Constant::Field(ark_bn254::Fr::zero())),
        _ => None,
    })
}

fn materialize_one(
    types: &FunctionTypeInfo,
    result: ValueId,
    fb: &mut HLFunctionBuilder<'_>,
) -> Option<Rewrite> {
    materialize_const(types, result, fb, |t| match t {
        TypeExpr::U(s) => Some(Constant::U(*s, 1)),
        TypeExpr::I(s) => Some(Constant::I(*s, 1)),
        TypeExpr::Field => Some(Constant::Field(ark_bn254::Fr::one())),
        _ => None,
    })
}

fn is_zero(ssa: &HLSSA, v: ValueId) -> bool {
    match ssa.get_const(v).as_deref() {
        Some(Constant::U(_, 0) | Constant::I(_, 0)) => true,
        Some(Constant::Field(f)) => f.is_zero(),
        _ => false,
    }
}

fn is_one(ssa: &HLSSA, v: ValueId) -> bool {
    match ssa.get_const(v).as_deref() {
        Some(Constant::U(_, 1) | Constant::I(_, 1)) => true,
        Some(Constant::Field(f)) => f.is_one(),
        _ => false,
    }
}

fn is_all_ones(ssa: &HLSSA, v: ValueId) -> bool {
    match ssa.get_const(v).as_deref() {
        Some(Constant::U(bits, value) | Constant::I(bits, value)) => *value == bit_mask(*bits),
        _ => false,
    }
}

fn const_as_usize(ssa: &HLSSA, v: ValueId) -> Option<usize> {
    match ssa.get_const(v).as_deref() {
        Some(Constant::U(_, value) | Constant::I(_, value)) => (*value).try_into().ok(),
        _ => None,
    }
}
