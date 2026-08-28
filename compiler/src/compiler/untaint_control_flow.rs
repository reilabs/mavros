//! Linearizes witness-dependent control flow into a form safe to lower into a ZK circuit.

use tracing::{Level, instrument};

use crate::{
    collections::HashMap,
    compiler::{
        analysis::{
            flow_analysis::{CFG, FlowAnalysis},
            types::{FunctionTypeInfo, TypeInfo, Types},
            value_definitions::{FunctionValueDefinitions, ValueDefinition},
            witness_info::{FunctionWitnessType, WitnessInfo, WitnessShape, WitnessType},
            witness_taint_inference::WitnessTaintInference,
        },
        ssa::{
            BlockId, FunctionId, SourceLocation, Terminator, ValueId,
            hlssa::{
                BinaryArithOpKind, CallTarget, CastTarget, Constant, HLBlock, HLFunction, HLSSA,
                LocatedOpCode, OpCode, SequenceTargetType, Type, TypeExpr,
                builder::{HLEmitter, HLInstrBuilder},
            },
        },
        util::ice_non_elided_tuple,
    },
};

pub struct UntaintControlFlow {}

/// Look up the witness level for a value, defaulting to Pure for values
/// not present in the witness type map (e.g., values created after type inference).
fn get_witness_or_pure(
    function_wt: &FunctionWitnessType,
    v: crate::compiler::ssa::ValueId,
) -> WitnessType {
    function_wt
        .value_witness_types
        .get(&v)
        .map(|wt| wt.toplevel_info())
        .unwrap_or(WitnessType::Pure)
}

/// Push an instruction, wrapping in Guard if block is tainted.
fn maybe_guard(
    instrs: &mut Vec<LocatedOpCode>,
    taint: Option<ValueId>,
    instr: OpCode,
    location: &SourceLocation,
) {
    let instruction = match taint {
        Some(taint) => OpCode::Guard {
            condition: taint,
            inner: Box::new(instr),
        },
        None => instr,
    };
    instrs.push(instruction.locate(location.clone()));
}

impl UntaintControlFlow {
    pub fn new() -> Self {
        Self {}
    }

    // -----------------------------------------------------------------------
    // Step 1: Type application — bake WitnessOf into SSA types
    //
    // Walks every function that has witness type info and rewrites SSA types
    // (block params, instruction result types, return types) to include
    // WitnessOf wrappers where witness inference determined a value is witness-
    // dependent. This must run before cast insertion / linearization so that
    // the type info pass can see the WitnessOf types.
    // -----------------------------------------------------------------------

    #[instrument(skip_all, name = "UntaintControlFlow::apply_types")]
    fn apply_types(&self, ssa: HLSSA, witness_inference: &WitnessTaintInference) -> HLSSA {
        let (mut result_ssa, functions, old_global_types) = ssa.prepare_rebuild();
        result_ssa.set_global_types(old_global_types);

        for (function_id, function) in functions.into_iter() {
            if let Some(function_wt) = witness_inference.try_get_function_witness_type(function_id)
            {
                let new_function = self.apply_types_to_function(function, function_wt);
                result_ssa.put_function(function_id, new_function);
            } else {
                result_ssa.put_function(function_id, function);
            }
        }

        result_ssa
    }

    fn apply_types_to_function(
        &self,
        function: HLFunction,
        function_wt: &FunctionWitnessType,
    ) -> HLFunction {
        let (mut function, blocks, returns) = function.prepare_rebuild();

        for (block_id, mut block) in blocks.into_iter() {
            let mut new_block = HLBlock::empty();

            let mut new_parameters = Vec::new();
            for (value_id, typ) in block.take_parameters() {
                let wt = function_wt
                    .try_get_value_witness_type(value_id)
                    .expect("ICE: block parameter without an inferred witness shape");
                new_parameters.push((value_id, apply_witness_type(typ, wt)));
            }
            new_block.put_parameters(new_parameters);

            let mut new_instructions = Vec::new();
            for instruction in block.take_instructions() {
                let location = instruction.location().clone();
                let new = match instruction.payload() {
                    instr @ OpCode::Alloc { .. } => instr,
                    instr @ OpCode::FreshWitness { .. } => instr,
                    OpCode::MkSeq {
                        result: r,
                        elems: l,
                        seq_type: stp,
                        elem_type: tp,
                    } => {
                        let r_wt = function_wt
                            .try_get_value_witness_type(r)
                            .expect("ICE: instruction result without an inferred witness shape")
                            .child_witness_type()
                            .unwrap();
                        OpCode::MkSeq {
                            result: r,
                            elems: l,
                            seq_type: stp,
                            elem_type: apply_witness_type(tp, &r_wt),
                        }
                    }
                    OpCode::MkSeqOfBlob {
                        result: r,
                        element_type: tp,
                        blob,
                    } => {
                        let r_wt = function_wt
                            .try_get_value_witness_type(r)
                            .expect("ICE: instruction result without an inferred witness shape")
                            .child_witness_type()
                            .unwrap();
                        OpCode::MkSeqOfBlob {
                            result: r,
                            element_type: apply_witness_type(tp, &r_wt),
                            blob,
                        }
                    }
                    OpCode::MkRepeated {
                        result: r,
                        element,
                        seq_type,
                        count,
                        elem_type: tp,
                    } => {
                        let r_wt = function_wt
                            .try_get_value_witness_type(r)
                            .expect("ICE: instruction result without an inferred witness shape")
                            .child_witness_type()
                            .unwrap();
                        OpCode::MkRepeated {
                            result: r,
                            element,
                            seq_type,
                            count,
                            elem_type: apply_witness_type(tp, &r_wt),
                        }
                    }
                    OpCode::MkTuple { .. }
                    | OpCode::TupleProj { .. }
                    | OpCode::TupleRefProj { .. } => ice_non_elided_tuple(),
                    // Guards are introduced by this pass itself (step 2) and by later passes,
                    // never before witness inference; a Guard slipping through here would
                    // silently skip the elem_type rewrite of a wrapped Alloc/MkSeq/... below.
                    OpCode::Guard { .. } => {
                        panic!("ICE: Guard should not be present during witness type application")
                    }
                    OpCode::ReadGlobal {
                        result: r,
                        offset: l,
                        result_type: tp,
                    } => OpCode::ReadGlobal {
                        result: r,
                        offset: l,
                        result_type: tp,
                    },
                    OpCode::Todo {
                        payload,
                        results,
                        result_types,
                    } => OpCode::Todo {
                        payload,
                        results,
                        result_types,
                    },
                    other => other,
                };
                new_instructions.push(new.locate(location));
            }
            new_block.put_instructions(new_instructions);
            new_block.set_terminator(block.take_terminator().unwrap());
            function.put_block(block_id, new_block);
        }

        for (ret, ret_wt) in returns.into_iter().zip(function_wt.returns_witness.iter()) {
            let ret_typ = apply_witness_type(ret, ret_wt);
            function.add_return_type(ret_typ);
        }

        function
    }

    fn cast_alloc_inits(
        &self,
        ssa: &mut HLSSA,
        type_info: &TypeInfo,
        witness_inference: &WitnessTaintInference,
    ) {
        let function_ids: Vec<FunctionId> = ssa.get_function_ids().collect();
        for function_id in function_ids {
            let Some(function_wt) = witness_inference.try_get_function_witness_type(function_id)
            else {
                continue;
            };
            if !type_info.has_function(function_id) {
                continue;
            }
            let func_type_info = type_info.get_function(function_id);
            let mut function = ssa.take_function(function_id);

            let block_ids: Vec<BlockId> = function.get_blocks().map(|(bid, _)| *bid).collect();
            for block_id in block_ids {
                let instructions = function.get_block_mut(block_id).take_instructions();
                let mut new_instructions = Vec::new();
                for instruction in instructions {
                    let (instruction, location) = instruction.take();
                    let OpCode::Alloc { result, value } = instruction else {
                        new_instructions.push(instruction.locate(location));
                        continue;
                    };
                    let cell_shape = function_wt
                        .try_get_value_witness_type(result)
                        .expect("ICE: Alloc result without an inferred witness shape")
                        .child_witness_type()
                        .expect("ICE: Alloc result of a non-ref witness shape");
                    let cell_type = apply_witness_type(
                        func_type_info.get_value_type(value).strip_all_witness(),
                        &cell_shape,
                    );
                    let mut cast_instructions = Vec::new();
                    let converted = {
                        let mut builder = HLInstrBuilder::new(
                            &mut function,
                            ssa,
                            &mut cast_instructions,
                            location.clone(),
                        );
                        convert_if_needed(value, &cell_type, func_type_info, &mut builder)
                    };
                    for mut cast_instruction in cast_instructions {
                        *cast_instruction.location_mut() = location.clone();
                        new_instructions.push(cast_instruction);
                    }
                    new_instructions.push(
                        OpCode::Alloc {
                            result,
                            value: converted,
                        }
                        .locate(location),
                    );
                }
                function
                    .get_block_mut(block_id)
                    .put_instructions(new_instructions);
            }

            ssa.put_function(function_id, function);
        }
    }

    // -----------------------------------------------------------------------
    // Step 2: Cast insertion + control flow linearization
    //
    // After types are baked in and flow/type analysis is recomputed:
    //  - Linearizes witness-conditional branches: witness JmpIf is replaced
    //    by unconditional Jmp + Select at merge points; instructions in
    //    tainted blocks are wrapped in Guard.
    //  - Inserts WitnessOf casts at typed-slot boundaries (MkSeq elems,
    //    ArraySet value, SlicePush values, Store value, Select operands,
    //    Jmp args, Return values) where the actual type doesn't match the
    //    expected slot type.
    //  - Pushes cfg_witness arg to constrained calls; strips WitnessOf from
    //    unconstrained call args via ValueOf.
    // -----------------------------------------------------------------------

    #[instrument(skip_all, name = "UntaintControlFlow::run")]
    pub fn run(&mut self, ssa: HLSSA, witness_inference: &WitnessTaintInference) -> HLSSA {
        // Step 1: bake WitnessOf into SSA types
        let mut ssa = self.apply_types(ssa, witness_inference);

        // Recompute flow + type info (types changed in step 1)
        let flow_analysis = FlowAnalysis::run(&ssa);
        let type_info = Types::new().run(&ssa, &flow_analysis);

        self.cast_alloc_inits(&mut ssa, &type_info, witness_inference);
        let flow_analysis = FlowAnalysis::run(&ssa);
        let type_info = Types::new().run(&ssa, &flow_analysis);

        // Step 2: cast insertion + control flow linearization
        let function_ids: Vec<_> = ssa.get_function_ids().collect();
        for function_id in function_ids {
            if let Some(function_wt) = witness_inference.try_get_function_witness_type(function_id)
            {
                let func_type_info = if type_info.has_function(function_id) {
                    Some(type_info.get_function(function_id))
                } else {
                    None
                };
                let mut function = ssa.take_function(function_id);
                self.run_function(
                    function_id,
                    &mut function,
                    &mut ssa,
                    function_wt,
                    &flow_analysis,
                    func_type_info,
                );
                ssa.put_function(function_id, function);
            }
        }

        ssa
    }

    #[instrument(skip_all, name = "UntaintControlFlow::run_function", level = Level::DEBUG, fields(function = function.get_name()))]
    fn run_function(
        &mut self,
        function_id: FunctionId,
        function: &mut HLFunction,
        ssa: &mut HLSSA,
        function_wt: &FunctionWitnessType,
        flow_analysis: &FlowAnalysis,
        type_info: Option<&FunctionTypeInfo>,
    ) {
        let cfg = flow_analysis.get_function_cfg(function_id);
        // Snapshot before linearization rebuilds branch blocks in place. Reuse the compiler-wide
        // definition analysis so this recognition cannot drift from other definition consumers.
        let value_definitions = FunctionValueDefinitions::from_function(function);

        let cfg_witness_param = if matches!(function_wt.cfg_witness, WitnessInfo::Witness) {
            let entry_id = function.get_entry_id();
            let id = ssa.fresh_value();
            function
                .get_block_mut(entry_id)
                .push_parameter(id, Type::witness_of(Type::int(1)));
            Some(id)
        } else {
            None
        };

        // Collect block param types for Jmp cast insertion
        let block_param_types: HashMap<BlockId, Vec<Type>> = function
            .get_blocks()
            .map(|(bid, block)| {
                let types = block.get_parameters().map(|(_, tp)| tp.clone()).collect();
                (*bid, types)
            })
            .collect();

        let return_types: Vec<Type> = function.get_returns().to_vec();

        let mut block_taint_vars = HashMap::default();
        for (block_id, _) in function.get_blocks() {
            block_taint_vars.insert(*block_id, cfg_witness_param);
        }

        for block_id in cfg.get_blocks_bfs() {
            self.process_block(
                block_id,
                function,
                ssa,
                cfg,
                function_wt,
                &mut block_taint_vars,
                &block_param_types,
                return_types.as_slice(),
                type_info,
                &value_definitions,
            );
        }
    }

    fn process_block(
        &self,
        block_id: BlockId,
        function: &mut HLFunction,
        ssa: &mut HLSSA,
        cfg: &CFG,
        function_wt: &FunctionWitnessType,
        block_taint_vars: &mut HashMap<BlockId, Option<ValueId>>,
        block_param_types: &HashMap<BlockId, Vec<Type>>,
        return_types: &[Type],
        type_info: Option<&FunctionTypeInfo>,
        value_definitions: &FunctionValueDefinitions,
    ) {
        let mut block = function.take_block(block_id);
        let block_taint = *block_taint_vars.get(&block_id).unwrap();

        let old_instructions = block.take_instructions();
        // Blocks holding only a terminator have no location to anchor the taint plumbing to.
        let block_source_location = old_instructions
            .last()
            .map(|instruction| instruction.location().clone())
            .unwrap_or_else(|| SourceLocation::synthetic("untaint_control_flow"));
        let mut new_instructions = Vec::new();

        for instruction in old_instructions {
            let (instruction, location) = instruction.take();
            self.process_instruction(
                instruction,
                function,
                ssa,
                type_info,
                block_taint,
                &location,
                &mut new_instructions,
            );
        }

        // Handle terminator
        match block.get_terminator().cloned() {
            Some(Terminator::JmpIf(cond, if_true, if_false)) => {
                let cond_wt = get_witness_or_pure(function_wt, cond);
                match cond_wt {
                    WitnessType::Pure => {
                        // Pure JmpIf: insert casts at Jmp boundaries in branch blocks
                        // (handled when those blocks are processed)
                    }
                    WitnessType::Witness => {
                        // The linearization below assumes acyclic if/else structure: a
                        // witness-dependent back edge (a loop whose trip count depends on the
                        // witness) cannot be linearized — the `merge == if_*` shortcuts would
                        // emit an unconditional jump into the loop body, i.e. an infinite
                        // loop. Constrained Noir cannot produce one (loop bounds are
                        // compile-time), so fail loudly instead of silently mis-compiling.
                        assert!(
                            !cfg.is_loop_entry(block_id)
                                && !cfg.dominates(if_true, block_id)
                                && !cfg.dominates(if_false, block_id),
                            "ICE: witness-dependent branch condition on a loop edge \
                             (block {block_id:?}); witness loop bounds cannot be linearized"
                        );
                        // The then branch is taken when `cond` is true, the
                        // else branch when it's false. Each branch must run
                        // under a different guard, so compute both taints —
                        // `parent_taint AND cond` for then, `parent_taint AND
                        // NOT cond` for else — and assign per body block by
                        // which branch dominates it.
                        let then_taint = match block_taint {
                            Some(tnt) => {
                                let result_val = ssa.fresh_value();
                                new_instructions.push(LocatedOpCode::new(
                                    OpCode::BinaryArithOp {
                                        kind: BinaryArithOpKind::And,
                                        result: result_val,
                                        lhs: tnt,
                                        rhs: cond,
                                    },
                                    block_source_location.clone(),
                                ));
                                result_val
                            }
                            None => cond,
                        };
                        let not_cond = {
                            let nv = ssa.fresh_value();
                            new_instructions.push(LocatedOpCode::new(
                                OpCode::Not {
                                    result: nv,
                                    value: cond,
                                },
                                block_source_location.clone(),
                            ));
                            nv
                        };
                        let else_taint = match block_taint {
                            Some(tnt) => {
                                let result_val = ssa.fresh_value();
                                new_instructions.push(LocatedOpCode::new(
                                    OpCode::BinaryArithOp {
                                        kind: BinaryArithOpKind::And,
                                        result: result_val,
                                        lhs: tnt,
                                        rhs: not_cond,
                                    },
                                    block_source_location.clone(),
                                ));
                                result_val
                            }
                            None => not_cond,
                        };
                        let body = cfg.get_if_body(block_id);
                        for body_bid in body {
                            let taint = if cfg.dominates(if_true, body_bid) {
                                then_taint
                            } else if cfg.dominates(if_false, body_bid) {
                                else_taint
                            } else {
                                panic!(
                                    "untaint_cf: block {:?} in if-body is dominated by neither \
                                     then-branch {:?} nor else-branch {:?}",
                                    body_bid, if_true, if_false
                                );
                            };
                            block_taint_vars.insert(body_bid, Some(taint));
                        }

                        let merge = cfg.get_merge_point(block_id);

                        if merge == if_true {
                            block.set_terminator(Terminator::Jmp(if_false, vec![]));
                        } else if merge == if_false {
                            block.set_terminator(Terminator::Jmp(if_true, vec![]));
                        } else {
                            block.set_terminator(Terminator::Jmp(if_true, vec![]));

                            if merge == function.get_entry_id() {
                                panic!(
                                    "TODO: jump back into entry not supported yet. Is it even possible?"
                                )
                            }

                            let jumps = cfg.get_jumps_into_merge_from_branch(if_true, merge);
                            if jumps.len() != 1 {
                                panic!(
                                    "TODO: handle multiple jumps into merge {:?} {:?} {:?} {:?}",
                                    block_id, if_true, merge, jumps
                                );
                            }
                            let out_true_block = jumps[0];

                            let merge_params = function.get_block_mut(merge).take_parameters();

                            let args_passed_from_lhs = match function
                                .get_block_mut(out_true_block)
                                .take_terminator()
                            {
                                Some(Terminator::Jmp(_, args)) => args,
                                _ => panic!(
                                    "Impossible – out jump must be a JMP, otherwise the join point wouldn't be a join point"
                                ),
                            };

                            function
                                .get_block_mut(out_true_block)
                                .set_terminator(Terminator::Jmp(if_false, vec![]));

                            let jumps = cfg.get_jumps_into_merge_from_branch(if_false, merge);
                            if jumps.len() != 1 {
                                panic!(
                                    "TODO: handle multiple jumps into merge {:?} {:?} {:?} {:?}",
                                    block_id, if_false, merge, jumps
                                );
                            }
                            let out_false_block = jumps[0];
                            let args_passed_from_rhs = match function
                                .get_block_mut(out_false_block)
                                .take_terminator()
                            {
                                Some(Terminator::Jmp(_, args)) => args,
                                _ => panic!(
                                    "Impossible – out jump must be a JMP, otherwise the join point wouldn't be a join point"
                                ),
                            };

                            let merger_block = function.add_block();
                            function
                                .get_block_mut(out_false_block)
                                .set_terminator(Terminator::Jmp(merger_block, vec![]));
                            function
                                .get_block_mut(merger_block)
                                .set_terminator(Terminator::Jmp(merge, vec![]));

                            if !args_passed_from_lhs.is_empty() {
                                let mut instrs = Vec::new();
                                {
                                    let mut builder = HLInstrBuilder::new(
                                        function,
                                        ssa,
                                        &mut instrs,
                                        block_source_location.clone(),
                                    );
                                    for ((res, typ), (lhs, rhs)) in merge_params.iter().zip(
                                        args_passed_from_lhs
                                            .iter()
                                            .zip(args_passed_from_rhs.iter()),
                                    ) {
                                        let lhs_type = type_info
                                            .map(|ti| ti.get_value_type(*lhs).clone())
                                            .unwrap_or_else(|| typ.clone());
                                        let rhs_type = type_info
                                            .map(|ti| ti.get_value_type(*rhs).clone())
                                            .unwrap_or_else(|| typ.clone());
                                        emit_merge_select(
                                            &mut builder,
                                            cond,
                                            then_taint,
                                            else_taint,
                                            *lhs,
                                            *rhs,
                                            Some(*res),
                                            typ,
                                            &lhs_type,
                                            &rhs_type,
                                            value_definitions,
                                            type_info,
                                        );
                                    }
                                }
                                for instr in instrs {
                                    function.get_block_mut(merger_block).push_instruction(instr);
                                }
                            }
                        }
                    }
                }
            }
            Some(Terminator::Jmp(target, args)) => {
                // Insert casts at Jmp boundaries
                if let (Some(ti), Some(param_types)) = (type_info, block_param_types.get(&target)) {
                    let mut cast_instrs = Vec::new();
                    let new_args: Vec<_> = {
                        let mut builder = HLInstrBuilder::new(
                            function,
                            ssa,
                            &mut cast_instrs,
                            block_source_location.clone(),
                        );
                        args.iter()
                            .zip(param_types.iter())
                            .map(|(arg, expected_type)| {
                                convert_if_needed(*arg, expected_type, ti, &mut builder)
                            })
                            .collect()
                    };
                    flush_conversion_instrs_located(
                        &mut new_instructions,
                        block_taint,
                        cast_instrs,
                    );
                    block.set_terminator(Terminator::Jmp(target, new_args));
                }
            }
            Some(Terminator::Return(values)) => {
                if let Some(ti) = type_info {
                    let mut cast_instrs = Vec::new();
                    let new_values: Vec<_> = {
                        let mut builder = HLInstrBuilder::new(
                            function,
                            ssa,
                            &mut cast_instrs,
                            block_source_location.clone(),
                        );
                        values
                            .iter()
                            .zip(return_types.iter())
                            .map(|(val, expected_type)| {
                                convert_if_needed(*val, expected_type, ti, &mut builder)
                            })
                            .collect()
                    };
                    flush_conversion_instrs_located(
                        &mut new_instructions,
                        block_taint,
                        cast_instrs,
                    );
                    block.set_terminator(Terminator::Return(new_values));
                }
            }
            None => {}
        };

        block.put_instructions(new_instructions);
        function.put_block(block_id, block);
    }

    /// Process a single instruction: apply cast insertion, then Guard-wrap if tainted.
    fn process_instruction(
        &self,
        instruction: OpCode,
        function: &mut HLFunction,
        ssa: &mut HLSSA,
        type_info: Option<&FunctionTypeInfo>,
        block_taint: Option<ValueId>,
        location: &SourceLocation,
        new_instructions: &mut Vec<LocatedOpCode>,
    ) {
        match instruction {
            // -- Constrained Call: push cfg_witness arg --
            OpCode::Call {
                results: ret,
                function: CallTarget::Static(tgt),
                mut args,
                unconstrained: false,
            } => {
                if let Some(arg) = block_taint {
                    args.push(arg);
                }
                new_instructions.push(
                    OpCode::Call {
                        results: ret,
                        function: CallTarget::Static(tgt),
                        args,
                        unconstrained: false,
                    }
                    .locate(location.clone()),
                );
            }
            // -- Unconstrained Call: strip WitnessOf from args --
            OpCode::Call {
                results,
                function: CallTarget::Static(tgt),
                args,
                unconstrained: true,
            } => {
                if let Some(ti) = type_info {
                    let mut cast_instrs = Vec::new();
                    let new_args: Vec<_> = {
                        let mut builder =
                            HLInstrBuilder::new(function, ssa, &mut cast_instrs, location.clone());
                        args.into_iter()
                            .map(|arg| {
                                let arg_type = ti.get_value_type(arg);
                                let pure_type = arg_type.strip_all_witness();
                                if *arg_type != pure_type {
                                    emit_strip_witness(arg, arg_type, &pure_type, &mut builder)
                                } else {
                                    arg
                                }
                            })
                            .collect()
                    };
                    flush_conversion_instrs_located(new_instructions, block_taint, cast_instrs);
                    new_instructions.push(
                        OpCode::Call {
                            results,
                            function: CallTarget::Static(tgt),
                            args: new_args,
                            unconstrained: true,
                        }
                        .locate(location.clone()),
                    );
                } else {
                    new_instructions.push(
                        OpCode::Call {
                            results,
                            function: CallTarget::Static(tgt),
                            args,
                            unconstrained: true,
                        }
                        .locate(location.clone()),
                    );
                }
            }
            OpCode::Call {
                function: CallTarget::Dynamic(_),
                ..
            } => {
                panic!("Dynamic call targets are not supported in untaint_control_flow")
            }
            // -- Cast insertion for MkSeq --
            OpCode::MkSeq {
                result: r,
                elems: vs,
                seq_type: s,
                elem_type: ref tp,
            } if type_info.is_some() => {
                let ti = type_info.unwrap();
                let target_elem_type = tp.clone();
                let mut cast_instrs = Vec::new();
                let new_vs: Vec<_> = {
                    let mut builder =
                        HLInstrBuilder::new(function, ssa, &mut cast_instrs, location.clone());
                    vs.iter()
                        .map(|v| convert_if_needed(*v, &target_elem_type, ti, &mut builder))
                        .collect()
                };
                flush_conversion_instrs_located(new_instructions, block_taint, cast_instrs);
                maybe_guard(
                    new_instructions,
                    block_taint,
                    OpCode::MkSeq {
                        result: r,
                        elems: new_vs,
                        seq_type: s,
                        elem_type: target_elem_type,
                    },
                    location,
                );
            }
            // -- Cast insertion for MkRepeated --
            OpCode::MkRepeated {
                result: r,
                element,
                seq_type,
                count,
                elem_type: ref tp,
            } if type_info.is_some() => {
                let ti = type_info.unwrap();
                let target_elem_type = tp.clone();
                let mut cast_instrs = Vec::new();
                let new_element = {
                    let mut builder =
                        HLInstrBuilder::new(function, ssa, &mut cast_instrs, location.clone());
                    convert_if_needed(element, &target_elem_type, ti, &mut builder)
                };
                flush_conversion_instrs_located(new_instructions, block_taint, cast_instrs);
                maybe_guard(
                    new_instructions,
                    block_taint,
                    OpCode::MkRepeated {
                        result: r,
                        element: new_element,
                        seq_type,
                        count,
                        elem_type: target_elem_type,
                    },
                    location,
                );
            }
            // -- Cast insertion for ArraySet --
            OpCode::ArraySet {
                result,
                array,
                index,
                value,
            } if type_info.is_some() => {
                let ti = type_info.unwrap();
                let result_type = ti.get_value_type(result);
                let expected_elem_type = match &result_type.expr {
                    TypeExpr::Array(inner, _) => inner.as_ref().clone(),
                    TypeExpr::Slice(inner) => inner.as_ref().clone(),
                    _ => panic!("ArraySet on non-array type"),
                };
                let mut cast_instrs = Vec::new();
                let (converted_array, converted_value) = {
                    let mut builder =
                        HLInstrBuilder::new(function, ssa, &mut cast_instrs, location.clone());
                    let ca = convert_if_needed(array, result_type, ti, &mut builder);
                    let cv = convert_if_needed(value, &expected_elem_type, ti, &mut builder);
                    (ca, cv)
                };
                flush_conversion_instrs_located(new_instructions, block_taint, cast_instrs);
                maybe_guard(
                    new_instructions,
                    block_taint,
                    OpCode::ArraySet {
                        result,
                        array: converted_array,
                        index,
                        value: converted_value,
                    },
                    location,
                );
            }
            // -- Cast insertion for SlicePush --
            OpCode::SlicePush {
                dir,
                result,
                slice,
                values,
            } if type_info.is_some() => {
                let ti = type_info.unwrap();
                let result_slice_type = ti.get_value_type(result);
                let expected_elem_type = match &result_slice_type.expr {
                    TypeExpr::Slice(inner) => inner.as_ref().clone(),
                    _ => panic!("SlicePush on non-slice type"),
                };
                let mut cast_instrs = Vec::new();
                let (new_slice, new_values) = {
                    let mut builder =
                        HLInstrBuilder::new(function, ssa, &mut cast_instrs, location.clone());
                    let new_slice = convert_if_needed(slice, result_slice_type, ti, &mut builder);
                    let new_values: Vec<_> = values
                        .iter()
                        .map(|v| convert_if_needed(*v, &expected_elem_type, ti, &mut builder))
                        .collect();
                    (new_slice, new_values)
                };
                flush_conversion_instrs_located(new_instructions, block_taint, cast_instrs);
                maybe_guard(
                    new_instructions,
                    block_taint,
                    OpCode::SlicePush {
                        dir,
                        result,
                        slice: new_slice,
                        values: new_values,
                    },
                    location,
                );
            }
            // -- Cast insertion for Store --
            OpCode::Store { ptr, value } if type_info.is_some() => {
                let ti = type_info.unwrap();
                let ptr_type = ti.get_value_type(ptr);
                let target_type = ptr_type.get_pointed();
                let mut cast_instrs = Vec::new();
                let converted = {
                    let mut builder =
                        HLInstrBuilder::new(function, ssa, &mut cast_instrs, location.clone());
                    convert_if_needed(value, &target_type, ti, &mut builder)
                };
                flush_conversion_instrs_located(new_instructions, block_taint, cast_instrs);
                maybe_guard(
                    new_instructions,
                    block_taint,
                    OpCode::Store {
                        ptr,
                        value: converted,
                    },
                    location,
                );
            }
            // -- Cast insertion for Select --
            OpCode::Select {
                result: r,
                cond,
                if_t,
                if_f,
            } if type_info.is_some() => {
                let ti = type_info.unwrap();
                let if_t_type = ti.get_value_type(if_t);
                let if_f_type = ti.get_value_type(if_f);
                let target_type = if_t_type.get_select_result_type(if_f_type);
                let mut cast_instrs = Vec::new();
                let (new_if_t, new_if_f) = {
                    let mut builder =
                        HLInstrBuilder::new(function, ssa, &mut cast_instrs, location.clone());
                    let t = convert_if_needed(if_t, &target_type, ti, &mut builder);
                    let f = convert_if_needed(if_f, &target_type, ti, &mut builder);
                    (t, f)
                };
                flush_conversion_instrs_located(new_instructions, block_taint, cast_instrs);
                maybe_guard(
                    new_instructions,
                    block_taint,
                    OpCode::Select {
                        result: r,
                        cond,
                        if_t: new_if_t,
                        if_f: new_if_f,
                    },
                    location,
                );
            }
            // -- All other non-Call ops: Guard-wrap when tainted --
            other => maybe_guard(new_instructions, block_taint, other, location),
        }
    }
}

// ---------------------------------------------------------------------------
// Cast insertion helpers
// ---------------------------------------------------------------------------

fn convert_if_needed(
    value: ValueId,
    target_type: &Type,
    type_info: &FunctionTypeInfo,
    builder: &mut HLInstrBuilder<'_>,
) -> ValueId {
    let value_type = type_info.get_value_type(value);
    if *value_type == *target_type {
        return value;
    }
    emit_value_conversion(value, value_type, target_type, builder)
}

/// Convert a value from source_type to target_type. Scalar witness injections
/// become a single `WitnessOf` cast; arrays and slices become one composite
/// `Map` cast, lowered to a loop late by `LowerMapCasts` (and erased entirely
/// in the witgen pipeline by `StripWitnessOf`). Conversions are pure — the
/// result is a fresh value — so they are safe to execute unconditionally,
/// including in guarded (tainted) regions.
fn emit_value_conversion(
    value: ValueId,
    source_type: &Type,
    target_type: &Type,
    builder: &mut HLInstrBuilder<'_>,
) -> ValueId {
    match CastTarget::conversion(source_type, target_type) {
        None => value,
        Some(target) => builder.cast_to(target, value),
    }
}

/// Recursively strip WitnessOf from a value (for unconstrained call args).
fn emit_strip_witness(
    value: ValueId,
    source_type: &Type,
    target_type: &Type,
    builder: &mut HLInstrBuilder<'_>,
) -> ValueId {
    if source_type == target_type {
        return value;
    }
    // Toplevel WitnessOf(X) → X: emit ValueOf, then keep stripping inside.
    if let TypeExpr::WitnessOf(inner) = &source_type.expr {
        let unwrapped = builder.value_of(value);
        return emit_strip_witness(unwrapped, inner, target_type, builder);
    }
    match CastTarget::strip_conversion(source_type, target_type) {
        None => value,
        Some(target) => builder.cast_to(target, value),
    }
}

fn flush_conversion_instrs_located(
    instrs: &mut Vec<LocatedOpCode>,
    taint: Option<ValueId>,
    cast_instrs: Vec<LocatedOpCode>,
) {
    for instr in cast_instrs {
        let (instr, location) = instr.take();
        maybe_guard(instrs, taint, instr, &location);
    }
}

fn index_is_statically_in_bounds(builder: &HLInstrBuilder<'_>, index: ValueId, len: usize) -> bool {
    matches!(
        builder.ssa.get_const(index).as_deref(),
        Some(Constant::Int(_, value)) if *value < len as u128
    )
}

/// If `updated_value` is itself an update of `base[index]`, return the original element value.
/// Reusing that value lets sparse lowering continue recursively through a nested array update.
fn nested_update_base_element(
    updated_value: ValueId,
    base: ValueId,
    index: ValueId,
    definitions: &FunctionValueDefinitions,
) -> Option<ValueId> {
    let ValueDefinition::Instruction(
        _,
        _,
        OpCode::ArraySet {
            array: nested_base, ..
        },
    ) = definitions.get_definition(updated_value)?
    else {
        return None;
    };
    match definitions.get_definition(*nested_base) {
        Some(ValueDefinition::Instruction(
            _,
            _,
            OpCode::ArrayGet {
                array: get_array,
                index: get_index,
                ..
            },
        )) if *get_array == base && *get_index == index => Some(*nested_base),
        _ => None,
    }
}

fn guarded_array_get(
    builder: &mut HLInstrBuilder<'_>,
    guard: ValueId,
    array: ValueId,
    index: ValueId,
) -> ValueId {
    let result = builder.fresh_value();
    builder.push(OpCode::Guard {
        condition: guard,
        inner: Box::new(OpCode::ArrayGet {
            result,
            array,
            index,
        }),
    });
    result
}

/// Lower a merge where exactly one arm is a single functional update of the other.
///
/// ```text
/// select(c, ArraySet(a, i, v), a)
///   => ArraySet(a, i, select(c, v, ArrayGet(a, i)))
/// ```
///
/// The symmetric false-arm form is handled as well. Potentially out-of-bounds operations retain
/// the updated arm's control-flow guard; this is essential because moving an unguarded `ArraySet`
/// out of its branch would make an inactive out-of-bounds update fail.
#[allow(clippy::too_many_arguments)]
fn try_emit_sparse_array_merge(
    builder: &mut HLInstrBuilder<'_>,
    cond: ValueId,
    lhs_guard: ValueId,
    rhs_guard: ValueId,
    lhs: ValueId,
    rhs: ValueId,
    result: Option<ValueId>,
    result_type: &Type,
    lhs_type: &Type,
    rhs_type: &Type,
    result_elem_type: &Type,
    size: usize,
    definitions: &FunctionValueDefinitions,
    type_info: Option<&FunctionTypeInfo>,
) -> Option<ValueId> {
    // A sparse ArraySet must start from a value with the exact result representation. The general
    // element-wise path below remains responsible for merges that need aggregate map-casts.
    if lhs_type != result_type || rhs_type != result_type {
        return None;
    }

    let (base, index, updated_value, update_is_lhs, update_guard) = match (
        definitions.get_definition(lhs),
        definitions.get_definition(rhs),
    ) {
        (
            Some(ValueDefinition::Instruction(
                _,
                _,
                OpCode::ArraySet {
                    array,
                    index,
                    value,
                    ..
                },
            )),
            _,
        ) if *array == rhs => (rhs, *index, *value, true, lhs_guard),
        (
            _,
            Some(ValueDefinition::Instruction(
                _,
                _,
                OpCode::ArraySet {
                    array,
                    index,
                    value,
                    ..
                },
            )),
        ) if *array == lhs => (lhs, *index, *value, false, rhs_guard),
        _ => return None,
    };

    let statically_in_bounds = index_is_statically_in_bounds(builder, index, size);

    let existing_base_element = nested_update_base_element(updated_value, base, index, definitions);
    let base_element = existing_base_element.unwrap_or_else(|| {
        if statically_in_bounds {
            builder.array_get(base, index)
        } else {
            guarded_array_get(builder, update_guard, base, index)
        }
    });
    let updated_value_type = type_info
        .map(|ti| ti.get_value_type(updated_value).clone())
        .unwrap_or_else(|| result_elem_type.clone());
    let base_element_type = existing_base_element
        .and_then(|value| type_info.map(|ti| ti.get_value_type(value).clone()))
        .unwrap_or_else(|| result_elem_type.clone());

    let selected_element = if update_is_lhs {
        emit_merge_select(
            builder,
            cond,
            lhs_guard,
            rhs_guard,
            updated_value,
            base_element,
            None,
            result_elem_type,
            &updated_value_type,
            &base_element_type,
            definitions,
            type_info,
        )
    } else {
        emit_merge_select(
            builder,
            cond,
            lhs_guard,
            rhs_guard,
            base_element,
            updated_value,
            None,
            result_elem_type,
            &base_element_type,
            &updated_value_type,
            definitions,
            type_info,
        )
    };

    let result = result.unwrap_or_else(|| builder.fresh_value());
    let array_set = OpCode::ArraySet {
        result,
        array: base,
        index,
        value: selected_element,
    };
    if statically_in_bounds {
        builder.push(array_set);
    } else {
        builder.push(OpCode::Guard {
            condition: update_guard,
            inner: Box::new(array_set),
        });
    }
    Some(result)
}

/// Emit selects for merge point values, handling type conversion between branch values and the
/// expected merge param type. Single-index array updates are kept sparse (and guarded when their
/// index may fail); unrelated arrays use the general unrolled element-wise select + cast path.
#[allow(clippy::too_many_arguments)]
fn emit_merge_select(
    builder: &mut HLInstrBuilder<'_>,
    cond: ValueId,
    lhs_guard: ValueId,
    rhs_guard: ValueId,
    lhs: ValueId,
    rhs: ValueId,
    result: Option<ValueId>,
    result_type: &Type,
    lhs_type: &Type,
    rhs_type: &Type,
    value_definitions: &FunctionValueDefinitions,
    type_info: Option<&FunctionTypeInfo>,
) -> ValueId {
    if lhs == rhs && lhs_type == result_type && rhs_type == result_type {
        return match result {
            Some(result) if result != lhs => {
                builder.push(OpCode::Cast {
                    result,
                    value: lhs,
                    target: CastTarget::Nop,
                });
                result
            }
            _ => lhs,
        };
    }

    match &result_type.expr {
        TypeExpr::Array(result_elem_type, size) => {
            if let Some(result) = try_emit_sparse_array_merge(
                builder,
                cond,
                lhs_guard,
                rhs_guard,
                lhs,
                rhs,
                result,
                result_type,
                lhs_type,
                rhs_type,
                result_elem_type,
                *size,
                value_definitions,
                type_info,
            ) {
                return result;
            }

            let lhs_elem_type = match &lhs_type.expr {
                TypeExpr::Array(e, _) => e.as_ref(),
                _ => panic!(
                    "emit_merge_select: expected array for lhs, got {:?}",
                    lhs_type
                ),
            };
            let rhs_elem_type = match &rhs_type.expr {
                TypeExpr::Array(e, _) => e.as_ref(),
                _ => panic!(
                    "emit_merge_select: expected array for rhs, got {:?}",
                    rhs_type
                ),
            };
            let mut elems = Vec::with_capacity(*size);
            for i in 0..*size {
                let idx = builder.int_const(32, i as u128);
                let lhs_elem = builder.array_get(lhs, idx);
                let rhs_elem = builder.array_get(rhs, idx);
                let selected = emit_merge_select(
                    builder,
                    cond,
                    lhs_guard,
                    rhs_guard,
                    lhs_elem,
                    rhs_elem,
                    None,
                    result_elem_type,
                    lhs_elem_type,
                    rhs_elem_type,
                    value_definitions,
                    type_info,
                );
                elems.push(selected);
            }
            let result = result.unwrap_or_else(|| builder.fresh_value());
            builder.push(OpCode::MkSeq {
                result,
                elems,
                seq_type: SequenceTargetType::Array(*size),
                elem_type: *result_elem_type.clone(),
            });
            result
        }
        TypeExpr::Tuple(_) => ice_non_elided_tuple(),
        TypeExpr::WitnessOf(_) => {
            // Cast operands to WitnessOf if they aren't already
            let lhs = if !lhs_type.is_witness_of() {
                builder.cast_to_witness_of(lhs)
            } else {
                lhs
            };
            let rhs = if !rhs_type.is_witness_of() {
                builder.cast_to_witness_of(rhs)
            } else {
                rhs
            };
            let result = result.unwrap_or_else(|| builder.fresh_value());
            builder.push(OpCode::Select {
                result,
                cond,
                if_t: lhs,
                if_f: rhs,
            });
            result
        }
        TypeExpr::Field | TypeExpr::Int(_) => {
            let result = result.unwrap_or_else(|| builder.fresh_value());
            builder.push(OpCode::Select {
                result,
                cond,
                if_t: lhs,
                if_f: rhs,
            });
            result
        }
        TypeExpr::Ref(_) => panic!("Witness select on Ref type not supported"),
        TypeExpr::Slice(_) => {
            let lhs = emit_value_conversion(lhs, lhs_type, result_type, builder);
            let rhs = emit_value_conversion(rhs, rhs_type, result_type, builder);
            let result = result.unwrap_or_else(|| builder.fresh_value());
            builder.push(OpCode::Select {
                result,
                cond,
                if_t: lhs,
                if_f: rhs,
            });
            result
        }
        TypeExpr::Function => panic!("Witness select on Function type not supported"),
        TypeExpr::Blob(..) => panic!("Witness select on Blob type not supported"),
    }
}

// ---------------------------------------------------------------------------
// Type application helper
// ---------------------------------------------------------------------------

fn apply_witness_type(typ: Type, wt: &WitnessShape) -> Type {
    match (typ.expr, wt) {
        (TypeExpr::Field, WitnessShape::Scalar(info)) => {
            let base = Type::field();
            if info.is_witness() {
                Type::witness_of(base)
            } else {
                base
            }
        }
        (TypeExpr::Int(size), WitnessShape::Scalar(info)) => {
            let base = Type::int(size);
            if info.is_witness() {
                Type::witness_of(base)
            } else {
                base
            }
        }
        (TypeExpr::Array(inner, size), WitnessShape::Array(inner_wt)) => {
            apply_witness_type(*inner, inner_wt.as_ref()).array_of(size)
        }
        // A slice's `top`/`len` taint is deliberately *not* applied as the earlier
        // `purify_witness_slices` pass has already moved it onto a `log_len` scalar.
        (TypeExpr::Slice(inner), WitnessShape::Slice(_, inner_wt)) => {
            let elem_base = apply_witness_type(*inner, inner_wt.as_ref());
            elem_base.slice_of()
        }
        (TypeExpr::Ref(inner), WitnessShape::Ref(top, inner_wt)) => {
            let base = apply_witness_type(*inner, inner_wt.as_ref()).ref_of();
            if top.is_witness() {
                Type::witness_of(base)
            } else {
                base
            }
        }
        (TypeExpr::Tuple(_), _) => ice_non_elided_tuple(),
        (TypeExpr::Blob(elem, n), wt @ WitnessShape::Scalar(info)) => {
            // Blobs hold raw constant/input data; they can never carry witness
            // values at any level. The taint inference models them as opaque
            // scalar leaves.
            assert!(
                !info.is_witness(),
                "ICE: Blob type inferred as witness-bearing: {wt}"
            );
            Type::blob(*elem, n)
        }
        (tp, wt) => panic!("Unexpected type {:?} with witness type {:?}", tp, wt),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ssa::hlssa::builder::HLSSABuilder;

    fn scalar(w: WitnessType) -> WitnessShape {
        WitnessShape::Scalar(w)
    }
    fn array(inner: WitnessShape) -> WitnessShape {
        WitnessShape::Array(Box::new(inner))
    }
    fn reference(top: WitnessType, inner: WitnessShape) -> WitnessShape {
        WitnessShape::Ref(top, Box::new(inner))
    }
    use WitnessType::{Pure, Witness};

    /// A witness-element array materializes as `Array(WitnessOf(Field), N)`
    #[test]
    fn witness_array_wraps_leaves_not_container() {
        let ty = Type::field().array_of(3);
        let shape = array(scalar(Witness));
        let got = apply_witness_type(ty, &shape);
        assert_eq!(got, Type::witness_of(Type::field()).array_of(3));
        assert_eq!(got.get_array_element(), Type::witness_of(Type::field()));
    }

    /// Nested witness array `[[Field; 2]; 3]`: the wrapper lands on the scalar
    /// only; no container level is wrapped.
    #[test]
    fn nested_witness_array_wraps_scalar_level_only() {
        let ty = Type::field().array_of(2).array_of(3);
        let shape = array(array(scalar(Witness)));
        let got = apply_witness_type(ty, &shape);
        assert_eq!(got, Type::witness_of(Type::field()).array_of(2).array_of(3));
        assert_eq!(
            got.get_array_element(),
            Type::witness_of(Type::field()).array_of(2)
        );
    }

    /// Array of refs `[&Field; 2]`: the wrapper lands on the ref
    /// elements.
    #[test]
    fn witness_ref_elements_wrap_refs_not_container() {
        let ty = Type::field().ref_of().array_of(2);
        let shape = array(reference(Witness, scalar(Pure)));
        let got = apply_witness_type(ty, &shape);
        assert_eq!(got, Type::witness_of(Type::field().ref_of()).array_of(2));
        assert_eq!(
            got.get_array_element(),
            Type::witness_of(Type::field().ref_of())
        );
    }

    /// A fully-pure array stays pure.
    #[test]
    fn pure_array_is_unchanged() {
        let ty = Type::field().array_of(3);
        let shape = array(scalar(Pure));
        assert_eq!(apply_witness_type(ty, &shape), Type::field().array_of(3));
    }

    /// A nested constant-index update must stay sparse through witness-control-flow merging. The
    /// old lowering rebuilt every scalar leaf of the outer array, making this cost proportional to
    /// the entire aggregate for every loop iteration.
    #[test]
    fn nested_array_set_merge_selects_only_the_updated_leaf() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let nested_type = Type::int(64).array_of(3).array_of(8);
        let (cond, base, updated, incremented) = {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.test_block(entry);
                let cond = e.add_parameter(Type::bool());
                let base = e.add_parameter(nested_type.clone());
                let outer_index = e.int_const(32, 4);
                let inner = e.array_get(base, outer_index);
                let inner_index = e.int_const(32, 0);
                let old = e.array_get(inner, inner_index);
                let one = e.int_const(64, 1);
                let incremented = e.uadd(old, one);
                let updated_inner = e.array_set(inner, inner_index, incremented);
                let updated = e.array_set(base, outer_index, updated_inner);
                e.terminate_return(vec![updated]);
                (cond, base, updated, incremented)
            })
        };

        let mut function = ssa.take_function(main_id);
        let definitions = FunctionValueDefinitions::from_function(&function);
        let result = ssa.fresh_value();
        let mut instructions = Vec::new();
        {
            let mut builder = HLInstrBuilder::new(
                &mut function,
                &mut ssa,
                &mut instructions,
                SourceLocation::test(),
            );
            assert_eq!(
                emit_merge_select(
                    &mut builder,
                    cond,
                    cond,
                    cond,
                    updated,
                    base,
                    Some(result),
                    &nested_type,
                    &nested_type,
                    &nested_type,
                    &definitions,
                    None,
                ),
                result
            );
        }

        let select_count = instructions
            .iter()
            .filter(|op| matches!(op.as_ref(), OpCode::Select { .. }))
            .count();
        let array_set_count = instructions
            .iter()
            .filter(|op| matches!(op.as_ref(), OpCode::ArraySet { .. }))
            .count();
        let mk_seq_count = instructions
            .iter()
            .filter(|op| matches!(op.as_ref(), OpCode::MkSeq { .. }))
            .count();
        assert_eq!(select_count, 1, "only the changed scalar leaf is selected");
        assert_eq!(array_set_count, 2, "the nested update chain is preserved");
        assert_eq!(mk_seq_count, 0, "neither array level is rebuilt densely");
        assert!(instructions.iter().any(|op| matches!(
            op.as_ref(),
            OpCode::Select { if_t, .. } if *if_t == incremented
        )));
        assert!(instructions.iter().any(|op| matches!(
            op.as_ref(),
            OpCode::ArraySet { result: r, array, .. } if *r == result && *array == base
        )));
    }

    /// The update may be on the false arm; the selected scalar operands must follow the branch
    /// orientation while retaining the same sparse ArraySet chain.
    #[test]
    fn nested_array_set_merge_handles_false_arm_update() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let nested_type = Type::int(64).array_of(2).array_of(4);
        let (cond, base, updated, replacement) = {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.test_block(entry);
                let cond = e.add_parameter(Type::bool());
                let base = e.add_parameter(nested_type.clone());
                let outer_index = e.int_const(32, 2);
                let inner = e.array_get(base, outer_index);
                let inner_index = e.int_const(32, 1);
                let replacement = e.int_const(64, 9);
                let updated_inner = e.array_set(inner, inner_index, replacement);
                let updated = e.array_set(base, outer_index, updated_inner);
                e.terminate_return(vec![updated]);
                (cond, base, updated, replacement)
            })
        };

        let mut function = ssa.take_function(main_id);
        let definitions = FunctionValueDefinitions::from_function(&function);
        let mut instructions = Vec::new();
        {
            let mut builder = HLInstrBuilder::new(
                &mut function,
                &mut ssa,
                &mut instructions,
                SourceLocation::test(),
            );
            emit_merge_select(
                &mut builder,
                cond,
                cond,
                cond,
                base,
                updated,
                None,
                &nested_type,
                &nested_type,
                &nested_type,
                &definitions,
                None,
            );
        }

        assert_eq!(
            instructions
                .iter()
                .filter(|op| matches!(op.as_ref(), OpCode::Select { .. }))
                .count(),
            1
        );
        assert_eq!(
            instructions
                .iter()
                .filter(|op| matches!(op.as_ref(), OpCode::ArraySet { .. }))
                .count(),
            2
        );
        assert!(instructions.iter().any(|op| matches!(
            op.as_ref(),
            OpCode::Select { if_f, .. } if *if_f == replacement
        )));
    }

    /// A dynamic index can fail at runtime. Its sparse ArrayGet and ArraySet must remain guarded so
    /// an inactive out-of-bounds update still passes through the original array.
    #[test]
    fn dynamic_array_set_index_keeps_failure_guards() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let array_type = Type::field().array_of(4);
        let (cond, base, updated) = {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.test_block(entry);
                let cond = e.add_parameter(Type::bool());
                let base = e.add_parameter(array_type.clone());
                let dynamic_index = e.add_parameter(Type::int(32));
                let replacement = e.field_const(e.field().constant(7u64));
                let updated = e.array_set(base, dynamic_index, replacement);
                e.terminate_return(vec![updated]);
                (cond, base, updated)
            })
        };

        let mut function = ssa.take_function(main_id);
        let definitions = FunctionValueDefinitions::from_function(&function);
        let mut instructions = Vec::new();
        {
            let mut builder = HLInstrBuilder::new(
                &mut function,
                &mut ssa,
                &mut instructions,
                SourceLocation::test(),
            );
            emit_merge_select(
                &mut builder,
                cond,
                cond,
                cond,
                updated,
                base,
                None,
                &array_type,
                &array_type,
                &array_type,
                &definitions,
                None,
            );
        }

        assert_eq!(
            instructions
                .iter()
                .filter(|op| matches!(op.as_ref(), OpCode::Select { .. }))
                .count(),
            1
        );
        assert_eq!(
            instructions
                .iter()
                .filter(|op| matches!(op.as_ref(), OpCode::MkSeq { .. }))
                .count(),
            0
        );
        assert_eq!(
            instructions
                .iter()
                .filter(|op| matches!(op.as_ref(), OpCode::ArraySet { .. }))
                .count(),
            0
        );
        assert_eq!(
            instructions
                .iter()
                .filter(|op| matches!(op.as_ref(), OpCode::Guard { .. }))
                .count(),
            2,
            "the dynamic ArrayGet and ArraySet both retain the branch guard"
        );
        assert!(instructions.iter().any(|op| matches!(
            op.as_ref(),
            OpCode::Guard { inner, .. } if matches!(inner.as_ref(), OpCode::ArrayGet { .. })
        )));
        assert!(instructions.iter().any(|op| matches!(
            op.as_ref(),
            OpCode::Guard { inner, .. } if matches!(inner.as_ref(), OpCode::ArraySet { .. })
        )));
    }
}
