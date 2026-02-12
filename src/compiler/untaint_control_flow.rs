use std::collections::HashMap;

use tracing::{Level, instrument};

use crate::compiler::{
    flow_analysis::FlowAnalysis,
    ir::r#type::{Type, TypeExpr},
    passes::fix_double_jumps::ValueReplacements,
    ssa::{BinaryArithOpKind, Block, CallTarget, Function, FunctionId, OpCode, SSA, Terminator, TupleIdx},
    witness_info::{ConstantWitness, FunctionWitnessType, WitnessInfo, WitnessType},
    witness_type_inference::WitnessTypeInference,
};

pub struct UntaintControlFlow {}

impl UntaintControlFlow {
    pub fn new() -> Self {
        Self {}
    }

    #[instrument(skip_all, name = "UntaintControlFlow::run")]
    pub fn run(
        &mut self,
        ssa: SSA,
        witness_inference: &WitnessTypeInference,
        flow_analysis: &FlowAnalysis,
    ) -> SSA {
        let (mut result_ssa, functions, old_global_types) = ssa.prepare_rebuild();

        // Convert global types from Empty to ConstantWitness
        let new_global_types: Vec<_> = old_global_types.into_iter()
            .map(|t| self.pure_taint_for_type(t))
            .collect();
        result_ssa.set_global_types(new_global_types);

        for (function_id, function) in functions.into_iter() {
            let function_wt = witness_inference.get_function_witness_type(function_id);
            let new_function =
                self.run_function(function_id, function, function_wt, flow_analysis);
            result_ssa.put_function(function_id, new_function);
        }

        // Post-process: wrapper_main's entry params should be plain Field/U types,
        // not WitnessOf. The VM writes concrete values to these positions.
        // Insert WriteWitness to bridge Field → WitnessOf(Field) before passing
        // to original_main.
        Self::fix_main_entry_params(&mut result_ssa);

        result_ssa
    }

    #[instrument(skip_all, name = "UntaintControlFlow::run_function", level = Level::DEBUG, fields(function = function.get_name()))]
    fn run_function(
        &mut self,
        function_id: FunctionId,
        function: Function,
        function_wt: &FunctionWitnessType,
        flow_analysis: &FlowAnalysis,
    ) -> Function {
        let cfg = flow_analysis.get_function_cfg(function_id);

        let (mut function, blocks, returns) = function.prepare_rebuild();

        for (block_id, mut block) in blocks.into_iter() {
            let mut new_block = Block::empty();

            let mut new_parameters = Vec::new();
            for (value_id, typ) in block.take_parameters() {
                let wt = function_wt.get_value_witness_type(value_id);
                new_parameters.push((value_id, self.apply_witness_type(typ, wt)));
            }
            new_block.put_parameters(new_parameters);

            let mut new_instructions = Vec::<OpCode>::new();
            for instruction in block.take_instructions() {
                let new = match instruction {
                    OpCode::BinaryArithOp {
                        kind,
                        result: r,
                        lhs: l,
                        rhs: h,
                    } => OpCode::BinaryArithOp {
                        kind: kind,
                        result: r,
                        lhs: l,
                        rhs: h,
                    },
                    OpCode::Cmp {
                        kind,
                        result: r,
                        lhs: l,
                        rhs: h,
                    } => OpCode::Cmp {
                        kind: kind,
                        result: r,
                        lhs: l,
                        rhs: h,
                    },
                    OpCode::Store { ptr: r, value: l } => OpCode::Store { ptr: r, value: l },
                    OpCode::Load { result: r, ptr: l } => OpCode::Load { result: r, ptr: l },
                    OpCode::ArrayGet {
                        result: r,
                        array: l,
                        index: h,
                    } => OpCode::ArrayGet {
                        result: r,
                        array: l,
                        index: h,
                    },
                    OpCode::ArraySet {
                        result: r,
                        array: l,
                        index: h,
                        value: j,
                    } => OpCode::ArraySet {
                        result: r,
                        array: l,
                        index: h,
                        value: j,
                    },
                    OpCode::SlicePush {
                        dir: d,
                        result: r,
                        slice: s,
                        values: v,
                    } => OpCode::SlicePush {
                        dir: d,
                        result: r,
                        slice: s,
                        values: v,
                    },
                    OpCode::SliceLen {
                        result: r,
                        slice: s,
                    } => OpCode::SliceLen {
                        result: r,
                        slice: s,
                    },
                    OpCode::Alloc {
                        result: r,
                        elem_type: l,
                    } => {
                        let r_wt = function_wt.get_value_witness_type(r);
                        let child = r_wt.child_witness_type().unwrap();
                        let child_typ = self.apply_witness_type(l, &child);
                        OpCode::Alloc {
                            result: r,
                            elem_type: child_typ,
                        }
                    }
                    OpCode::Call {
                        results: r,
                        function: CallTarget::Static(l),
                        args: h,
                    } => OpCode::Call {
                        results: r,
                        function: CallTarget::Static(l),
                        args: h,
                    },
                    OpCode::Call { function: CallTarget::Dynamic(_), .. } => {
                        panic!("Dynamic call targets are not supported in untaint_control_flow")
                    }
                    OpCode::AssertEq { lhs: r, rhs: l } => OpCode::AssertEq { lhs: r, rhs: l },
                    OpCode::AssertR1C { a: r, b: l, c: h } => {
                        OpCode::AssertR1C { a: r, b: l, c: h }
                    }
                    OpCode::Select {
                        result: r,
                        cond: l,
                        if_t: h,
                        if_f: j,
                    } => OpCode::Select {
                        result: r,
                        cond: l,
                        if_t: h,
                        if_f: j,
                    },
                    OpCode::WriteWitness {
                        result: r,
                        value: l,
                    } => OpCode::WriteWitness {
                        result: r,
                        value: l,
                    },
                    OpCode::FreshWitness {
                        result: r,
                        result_type: tp,
                    } => {
                        let wt = function_wt.get_value_witness_type(r);
                        let new_tp = self.apply_witness_type(tp, wt);
                        OpCode::FreshWitness {
                            result: r,
                            result_type: new_tp,
                        }
                    }
                    OpCode::Constrain { a, b, c } => OpCode::Constrain { a: a, b: b, c: c },
                    OpCode::NextDCoeff { result: a } => OpCode::NextDCoeff { result: a },
                    OpCode::BumpD {
                        matrix: a,
                        variable: b,
                        sensitivity: c,
                    } => OpCode::BumpD {
                        matrix: a,
                        variable: b,
                        sensitivity: c,
                    },
                    OpCode::MkSeq {
                        result: r,
                        elems: l,
                        seq_type: stp,
                        elem_type: tp,
                    } => {
                        let r_wt = function_wt
                            .get_value_witness_type(r)
                            .child_witness_type()
                            .unwrap();
                        OpCode::MkSeq {
                            result: r,
                            elems: l,
                            seq_type: stp,
                            elem_type: self.apply_witness_type(tp, &r_wt),
                        }
                    }
                    OpCode::Cast {
                        result: r,
                        value: l,
                        target: t,
                    } => OpCode::Cast {
                        result: r,
                        value: l,
                        target: t,
                    },
                    OpCode::Truncate {
                        result: r,
                        value: l,
                        to_bits: out_bits,
                        from_bits: in_bits,
                    } => OpCode::Truncate {
                        result: r,
                        value: l,
                        to_bits: out_bits,
                        from_bits: in_bits,
                    },
                    OpCode::Not {
                        result: r,
                        value: l,
                    } => OpCode::Not {
                        result: r,
                        value: l,
                    },
                    OpCode::ToBits {
                        result: r,
                        value: l,
                        endianness: e,
                        count: s,
                    } => OpCode::ToBits {
                        result: r,
                        value: l,
                        endianness: e,
                        count: s,
                    },
                    OpCode::ToRadix {
                        result: r,
                        value: l,
                        radix,
                        endianness: e,
                        count: s,
                    } => OpCode::ToRadix {
                        result: r,
                        value: l,
                        radix: radix,
                        endianness: e,
                        count: s,
                    },
                    OpCode::MemOp { kind, value } => OpCode::MemOp {
                        kind: kind,
                        value: value,
                    },
                    OpCode::MulConst {
                        result: r,
                        const_val: l,
                        var: c,
                    } => OpCode::MulConst {
                        result: r,
                        const_val: l,
                        var: c,
                    },
                    OpCode::Rangecheck {
                        value: val,
                        max_bits,
                    } => OpCode::Rangecheck {
                        value: val,
                        max_bits: max_bits,
                    },
                    OpCode::ReadGlobal {
                        result: r,
                        offset: l,
                        result_type: tp,
                    } => OpCode::ReadGlobal {
                        result: r,
                        offset: l,
                        result_type: self.pure_taint_for_type(tp),
                    },
                    OpCode::Lookup {
                        target,
                        keys,
                        results,
                    } => OpCode::Lookup {
                        target,
                        keys,
                        results,
                    },
                    OpCode::DLookup {
                        target,
                        keys,
                        results,
                    } => OpCode::DLookup {
                        target,
                        keys,
                        results,
                    },
                    OpCode::TupleProj {
                        result,
                        tuple,
                        idx,
                    } => {
                        match &idx {
                            TupleIdx::Static(sz) => {
                                OpCode::TupleProj {
                                    result,
                                    tuple,
                                    idx: TupleIdx::Static(*sz),
                                }
                            }
                            TupleIdx::Dynamic{..} => {
                                panic!("Dynamic TupleProj should not appear here")
                            }
                        }
                    }
                    OpCode::MkTuple {
                        result: r,
                        elems: l,
                        element_types: tps,
                    } => {
                        let r_wt = function_wt.get_value_witness_type(r);
                        let child_wts = if let WitnessType::Tuple(_, children) = r_wt {
                            children
                        } else {
                            panic!("MkTuple result should have Tuple witness type")
                        };
                        OpCode::MkTuple {
                            result: r,
                            elems: l,
                            element_types: tps
                                .iter()
                                .zip(child_wts.iter())
                                .map(|(tp, wt)| self.apply_witness_type(tp.clone(), wt))
                                .collect(),
                        }
                    }
                    OpCode::Todo { payload, results, result_types } => OpCode::Todo {
                        payload,
                        results,
                        result_types: result_types.iter().map(|tp| self.pure_taint_for_type(tp.clone())).collect(),
                    },
                    OpCode::InitGlobal { global, value } => OpCode::InitGlobal { global, value },
                    OpCode::DropGlobal { global } => OpCode::DropGlobal { global },
                    OpCode::ValueOf { .. } => panic!("ICE: ValueOf should not appear at this stage"),
                };
                new_instructions.push(new);
            }
            new_block.put_instructions(new_instructions);

            new_block.set_terminator(block.take_terminator().unwrap());

            function.put_block(block_id, new_block);
        }

        for (ret, ret_wt) in returns.into_iter().zip(function_wt.returns_witness.iter()) {
            let ret_typ = self.apply_witness_type(ret, ret_wt);
            function.add_return_type(ret_typ);
        }

        let cfg_witness_param = if matches!(
            function_wt.cfg_witness,
            WitnessInfo::Witness
        ) {
            Some(
                function.add_parameter(function.get_entry_id(), Type::witness_of(Type::u(1))),
            )
        } else {
            None
        };

        let mut block_taint_vars = HashMap::new();
        for (block_id, _) in function.get_blocks() {
            block_taint_vars.insert(*block_id, cfg_witness_param.clone());
        }

        for block_id in cfg.get_blocks_bfs() {
            let mut block = function.take_block(block_id);
            let block_taint = block_taint_vars.get(&block_id).unwrap().clone();

            let old_instructions = block.take_instructions();
            let mut new_instructions = Vec::new();

            for instruction in old_instructions {
                match instruction {
                    OpCode::BinaryArithOp {
                        kind: _,
                        result: _,
                        lhs: _,
                        rhs: _,
                    }
                    | OpCode::Cmp {
                        kind: _,
                        result: _,
                        lhs: _,
                        rhs: _,
                    }
                    | OpCode::MkSeq {
                        result: _,
                        elems: _,
                        seq_type: _,
                        elem_type: _,
                    }
                    | OpCode::MkTuple {..}
                    | OpCode::Cast {
                        result: _,
                        value: _,
                        target: _,
                    }
                    | OpCode::Truncate {
                        result: _,
                        value: _,
                        to_bits: _,
                        from_bits: _,
                    }
                    | OpCode::Not {
                        result: _,
                        value: _,
                    }
                    | OpCode::Rangecheck {
                        value: _,
                        max_bits: _,
                    }
                    | OpCode::ToBits {
                        result: _,
                        value: _,
                        endianness: _,
                        count: _,
                    }
                    | OpCode::ToRadix {
                        result: _,
                        value: _,
                        radix: _,
                        endianness: _,
                        count: _,
                    }
                    | OpCode::ReadGlobal {
                        result: _,
                        offset: _,
                        result_type: _,
                    } => {
                        new_instructions.push(instruction);
                    }
                    OpCode::AssertEq { lhs, rhs } => {
                        match block_taint {
                            Some(taint) => {
                                let new_rhs = function.fresh_value();
                                new_instructions.push(OpCode::Select {
                                    result: new_rhs,
                                    cond: taint,
                                    if_t: rhs,
                                    if_f: lhs,
                                });
                                new_instructions.push(OpCode::AssertEq {
                                    lhs: lhs,
                                    rhs: new_rhs,
                                })
                            }
                            None => new_instructions.push(instruction),
                        }
                    }
                    OpCode::Store { ptr, value: v } => {
                        let ptr_wt = function_wt
                            .get_value_witness_type(ptr)
                            .toplevel_info()
                            .expect_constant();
                        // writes to dynamic ptr not supported
                        assert_eq!(ptr_wt, ConstantWitness::Pure);

                        match block_taint {
                            Some(taint) => {
                                let old_value = function.fresh_value();
                                new_instructions.push(OpCode::Load {
                                    result: old_value,
                                    ptr: ptr,
                                });

                                let new_value = function.fresh_value();
                                new_instructions.push(OpCode::Select {
                                    result: new_value,
                                    cond: taint,
                                    if_t: v,
                                    if_f: old_value,
                                });

                                new_instructions.push(OpCode::Store {
                                    ptr: ptr,
                                    value: new_value,
                                });
                            }
                            None => new_instructions.push(instruction),
                        }
                    }
                    OpCode::Load { result: _, ptr } => {
                        let ptr_wt = function_wt
                            .get_value_witness_type(ptr)
                            .toplevel_info()
                            .expect_constant();
                        // reads from dynamic ptr not supported
                        assert_eq!(ptr_wt, ConstantWitness::Pure);
                        new_instructions.push(instruction);
                    }
                    OpCode::ArrayGet {
                        result: _,
                        array: arr,
                        index: _,
                    } => {
                        let arr_wt = function_wt
                            .get_value_witness_type(arr)
                            .toplevel_info()
                            .expect_constant();
                        // dynamic array access not supported
                        assert_eq!(arr_wt, ConstantWitness::Pure);
                        new_instructions.push(instruction);
                    }
                    OpCode::ArraySet {
                        result: _,
                        array: arr,
                        index: idx,
                        value: _,
                    } => {
                        let arr_wt = function_wt
                            .get_value_witness_type(arr)
                            .toplevel_info()
                            .expect_constant();
                        let idx_wt = function_wt
                            .get_value_witness_type(idx)
                            .toplevel_info()
                            .expect_constant();
                        // dynamic array access not supported
                        assert_eq!(arr_wt, ConstantWitness::Pure);
                        assert_eq!(idx_wt, ConstantWitness::Pure);
                        new_instructions.push(instruction);
                    }
                    OpCode::SlicePush {
                        dir: _,
                        result: _,
                        slice: sl,
                        values: _,
                    } => {
                        let slice_wt = function_wt
                            .get_value_witness_type(sl)
                            .toplevel_info()
                            .expect_constant();
                        // Slice must always be Pure witness
                        assert_eq!(slice_wt, ConstantWitness::Pure);
                        new_instructions.push(instruction);
                    }
                    OpCode::SliceLen {
                        result: _,
                        slice: sl,
                    } => {
                        let slice_wt = function_wt
                            .get_value_witness_type(sl)
                            .toplevel_info()
                            .expect_constant();
                        // Slice must always be Pure witness
                        assert_eq!(slice_wt, ConstantWitness::Pure);
                        new_instructions.push(instruction);
                    }
                    OpCode::Alloc {
                        result: _,
                        elem_type: _,
                    } => {
                        new_instructions.push(instruction);
                    }

                    OpCode::Call {
                        results: ret,
                        function: CallTarget::Static(tgt),
                        mut args,
                    } => {
                        match block_taint {
                            Some(arg) => {
                                args.push(arg);
                            }
                            None => {}
                        }
                        new_instructions.push(OpCode::Call {
                            results: ret,
                            function: CallTarget::Static(tgt),
                            args: args,
                        });
                    }
                    OpCode::Call { function: CallTarget::Dynamic(_), .. } => {
                        panic!("Dynamic call targets are not supported in untaint_control_flow")
                    }

                    OpCode::Todo { payload, results, result_types } => {
                        new_instructions.push(OpCode::Todo {
                            payload,
                            results,
                            result_types
                        });
                    }

                    OpCode::TupleProj {..} => {
                        new_instructions.push(instruction);
                    }
                    OpCode::InitGlobal { .. } | OpCode::DropGlobal { .. } => {
                        new_instructions.push(instruction);
                    }
                    _ => {
                        panic!("Unhandled instruction {:?}", instruction);
                    }
                }
            }

            match block.get_terminator().cloned() {
                Some(Terminator::JmpIf(cond, if_true, if_false)) => {
                    let cond_wt = function_wt
                        .get_value_witness_type(cond)
                        .toplevel_info()
                        .expect_constant();
                    match cond_wt {
                        ConstantWitness::Pure => {}
                        ConstantWitness::Witness => {
                            let child_block_taint = match block_taint {
                                Some(tnt) => {
                                    let result_val = function.fresh_value();
                                    new_instructions.push(OpCode::BinaryArithOp {
                                        kind: BinaryArithOpKind::And,
                                        result: result_val,
                                        lhs: tnt,
                                        rhs: cond,
                                    });
                                    result_val
                                }
                                None => cond,
                            };
                            let body = cfg.get_if_body(block_id);
                            for block_id in body {
                                block_taint_vars.insert(block_id, Some(child_block_taint));
                            }

                            let merge = cfg.get_merge_point(block_id);

                            // If one of the branches is empty, we just jump to the other and
                            // there's no need to merge any values – this is purely side-effectful,
                            // which the instruction rewrites will handle.
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

                                // We remove the parameters from the merge block – they will be un-phi-fied
                                let merge_params = function.get_block_mut(merge).take_parameters();

                                for (_, typ) in &merge_params {
                                    match typ {
                                        Type {
                                            expr: TypeExpr::Array(_, _),
                                            ..
                                        } => {
                                            panic!("TODO: Witness array not supported yet")
                                        }
                                        Type {
                                            expr: TypeExpr::Ref(_),
                                            ..
                                        } => {
                                            panic!("TODO: Witness ref not supported yet")
                                        }
                                        _ => {}
                                    }
                                }

                                let args_passed_from_lhs = match function
                                    .get_block_mut(out_true_block)
                                    .take_terminator()
                                {
                                    Some(Terminator::Jmp(_, args)) => args,
                                    _ => panic!(
                                        "Impossible – out jump must be a JMP, otherwise the join point wouldn't be a join point"
                                    ),
                                };

                                // Jump straight to the false block
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

                                if args_passed_from_lhs.len() > 0 {
                                    for ((res, _), (lhs, rhs)) in merge_params.iter().zip(
                                        args_passed_from_lhs
                                            .iter()
                                            .zip(args_passed_from_rhs.iter()),
                                    ) {
                                        function.get_block_mut(merger_block).push_instruction(
                                            OpCode::Select {
                                                result: *res,
                                                cond: cond,
                                                if_t: *lhs,
                                                if_f: *rhs,
                                            },
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            };

            block.put_instructions(new_instructions);
            function.put_block(block_id, block);
        }

        function
    }

    /// Fix wrapper_main entry params: strip WitnessOf from param types
    /// and insert WriteWitness instructions to convert Field → WitnessOf(Field).
    /// This is needed because the VM writes concrete Field values (4 limbs) to
    /// entry params, but WitnessOf(Field) is pointer-sized (1 slot).
    fn fix_main_entry_params(ssa: &mut SSA) {
        let main_id = ssa.get_main_id();
        let main_fn = ssa.get_function_mut(main_id);
        let entry_id = main_fn.get_entry_id();
        let entry_block = main_fn.get_block_mut(entry_id);

        let old_params = entry_block.take_parameters();
        let old_instructions = entry_block.take_instructions();

        let mut new_params = Vec::new();
        let mut write_witness_instructions = Vec::new();
        let mut replacements = ValueReplacements::new();

        for (value_id, typ) in &old_params {
            if typ.is_witness_of() {
                // Strip WitnessOf from param type — entry params are concrete values
                let inner_type = typ.strip_witness();
                new_params.push((*value_id, inner_type));

                // Create WriteWitness to bridge Field → WitnessOf(Field)
                let witness_val = main_fn.fresh_value();
                write_witness_instructions.push(OpCode::WriteWitness {
                    result: Some(witness_val),
                    value: *value_id,
                });
                replacements.insert(*value_id, witness_val);
            } else {
                new_params.push((*value_id, typ.clone()));
            }
        }

        // Rebuild entry block: params + WriteWitness instructions + original instructions
        let entry_block = main_fn.get_block_mut(entry_id);

        let mut new_instructions = write_witness_instructions;
        for mut instruction in old_instructions {
            replacements.replace_instruction(&mut instruction);
            new_instructions.push(instruction);
        }

        entry_block.put_parameters(new_params);
        // Also fix the terminator (e.g., return args)
        replacements.replace_terminator(entry_block.get_terminator_mut());
        entry_block.put_instructions(new_instructions);
    }

    fn apply_witness_type(&self, typ: Type, wt: &WitnessType) -> Type {
        match (typ.expr, wt) {
            (TypeExpr::Field, WitnessType::Scalar(info)) => {
                let base = Type::field();
                if info.expect_constant().is_witness() { Type::witness_of(base) } else { base }
            },
            (TypeExpr::U(size), WitnessType::Scalar(info)) => {
                let base = Type::u(size);
                if info.expect_constant().is_witness() { Type::witness_of(base) } else { base }
            },
            (TypeExpr::Array(inner, size), WitnessType::Array(top, inner_wt)) => {
                let base = self.apply_witness_type(*inner, inner_wt.as_ref()).array_of(size);
                if top.expect_constant().is_witness() { Type::witness_of(base) } else { base }
            },
            (TypeExpr::Slice(inner), WitnessType::Array(top, inner_wt)) => {
                let base = self.apply_witness_type(*inner, inner_wt.as_ref()).slice_of();
                if top.expect_constant().is_witness() { Type::witness_of(base) } else { base }
            },
            (TypeExpr::Ref(inner), WitnessType::Ref(top, inner_wt)) => {
                let base = self.apply_witness_type(*inner, inner_wt.as_ref()).ref_of();
                if top.expect_constant().is_witness() { Type::witness_of(base) } else { base }
            },
            (TypeExpr::Tuple(child_types), WitnessType::Tuple(top, child_wts)) => {
                let base = Type::tuple_of(
                    child_types.iter().zip(child_wts.iter()).map(|(child_type, child_wt)| self.apply_witness_type(child_type.clone(), child_wt)).collect()
                );
                if top.expect_constant().is_witness() { Type::witness_of(base) } else { base }
            },
            (tp, wt) => panic!("Unexpected type {:?} with witness type {:?}", tp, wt),
        }
    }

    fn pure_taint_for_type(&self, typ: Type) -> Type {
        // Types no longer have annotations; a "pure" type is just the type itself
        typ
    }
}
