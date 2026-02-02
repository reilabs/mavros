use std::collections::HashMap;

use tracing::{Level, instrument};

use crate::compiler::{
    flow_analysis::FlowAnalysis,
    ir::r#type::{Empty, Type, TypeExpr},
    ssa::{BinaryArithOpKind, Block, Function, FunctionId, OpCode, SSA, Terminator, TupleIdx},
    taint_analysis::{ConstantTaint, FunctionTaint, Taint, TaintAnalysis, TaintType},
};

pub struct UntaintControlFlow {}

impl UntaintControlFlow {
    pub fn new() -> Self {
        Self {}
    }

    #[instrument(skip_all, name = "UntaintControlFlow::run")]
    pub fn run(
        &mut self,
        ssa: SSA<Empty>,
        taint_analysis: &TaintAnalysis,
        flow_analysis: &FlowAnalysis,
    ) -> SSA<ConstantTaint> {
        let (mut result_ssa, functions) = ssa.prepare_rebuild::<ConstantTaint>();

        for (function_id, function) in functions.into_iter() {
            let function_taint = taint_analysis.get_function_taint(function_id);
            let new_function =
                self.run_function(function_id, function, function_taint, flow_analysis);
            result_ssa.put_function(function_id, new_function);
        }
        result_ssa
    }

    #[instrument(skip_all, name = "UntaintControlFlow::run_function", level = Level::DEBUG, fields(function = function.get_name()))]
    fn run_function(
        &mut self,
        function_id: FunctionId,
        function: Function<Empty>,
        function_taint: &FunctionTaint,
        flow_analysis: &FlowAnalysis,
    ) -> Function<ConstantTaint> {
        let cfg = flow_analysis.get_function_cfg(function_id);

        let (mut function, blocks, returns) = function.prepare_rebuild::<ConstantTaint>();

        for (block_id, mut block) in blocks.into_iter() {
            let mut new_block = Block::empty();

            let mut new_parameters = Vec::new();
            for (value_id, typ) in block.take_parameters() {
                let taint = function_taint.get_value_taint(value_id);
                new_parameters.push((value_id, self.typify_taint(typ, taint)));
            }
            new_block.put_parameters(new_parameters);

            let mut new_instructions = Vec::<OpCode<ConstantTaint>>::new();
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
                        result_annotation: _,
                    } => {
                        let r_taint = function_taint.get_value_taint(r);
                        let child = r_taint.child_taint_type().unwrap();
                        let child_typ = self.typify_taint(l, &child);
                        let self_taint = r_taint.toplevel_taint().expect_constant();
                        OpCode::Alloc {
                            result: r,
                            elem_type: child_typ,
                            result_annotation: self_taint,
                        }
                    }
                    OpCode::Call {
                        results: r,
                        function: l,
                        args: h,
                    } => OpCode::Call {
                        results: r,
                        function: l,
                        args: h,
                    },
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
                        witness_annotation: _,
                    } => OpCode::WriteWitness {
                        result: r,
                        value: l,
                        witness_annotation: ConstantTaint::Witness,
                    },
                    OpCode::FreshWitness {
                        result: r,
                        result_type: tp,
                    } => {
                        let taint = function_taint.get_value_taint(r);
                        let new_tp = self.typify_taint(tp, taint);
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
                        let r_taint = function_taint
                            .get_value_taint(r)
                            .child_taint_type()
                            .unwrap();
                        OpCode::MkSeq {
                            result: r,
                            elems: l,
                            seq_type: stp,
                            elem_type: self.typify_taint(tp, &r_taint),
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
                    OpCode::PureToWitnessRef {
                        result: r,
                        value: l,
                        result_annotation: _c,
                    } => OpCode::PureToWitnessRef {
                        result: r,
                        value: l,
                        result_annotation: function_taint
                            .get_value_taint(r)
                            .toplevel_taint()
                            .expect_constant(),
                    },
                    OpCode::UnboxField {
                        result: r,
                        value: l,
                    } => OpCode::UnboxField {
                        result: r,
                        value: l,
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
                        let r_taint = function_taint.get_value_taint(r);
                        let child_taints = if let TaintType::Tuple(_, children) = r_taint {
                            children
                        } else {
                            panic!("MkTuple result should have Tuple taint type")
                        };
                        OpCode::MkTuple {
                            result: r,
                            elems: l,
                            element_types: tps
                                .iter()
                                .zip(child_taints.iter())
                                .map(|(tp, taint)| self.typify_taint(tp.clone(), taint))
                                .collect(),
                        }
                    }
                    OpCode::Todo { payload, results, result_types } => OpCode::Todo {
                        payload,
                        results,
                        result_types: result_types.iter().map(|tp| self.pure_taint_for_type(tp.clone())).collect(),
                    },
                };
                new_instructions.push(new);
            }
            new_block.put_instructions(new_instructions);

            new_block.set_terminator(block.take_terminator().unwrap());

            function.put_block(block_id, new_block);
        }

        for (ret, ret_taint) in returns.into_iter().zip(function_taint.returns_taint.iter()) {
            let ret_typ = self.typify_taint(ret, ret_taint);
            function.add_return_type(ret_typ);
        }

        let cfg_taint_param = if matches!(
            function_taint.cfg_taint,
            Taint::Constant(ConstantTaint::Witness)
        ) {
            Some(
                function.add_parameter(function.get_entry_id(), Type::u(1, ConstantTaint::Witness)),
            )
        } else {
            None
        };

        let mut block_taint_vars = HashMap::new();
        for (block_id, _) in function.get_blocks() {
            block_taint_vars.insert(*block_id, cfg_taint_param.clone());
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
                        let ptr_taint = function_taint
                            .get_value_taint(ptr)
                            .toplevel_taint()
                            .expect_constant();
                        // writes to dynamic ptr not supported
                        assert_eq!(ptr_taint, ConstantTaint::Pure);

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
                        let ptr_taint = function_taint
                            .get_value_taint(ptr)
                            .toplevel_taint()
                            .expect_constant();
                        // reads from dynamic ptr not supported
                        assert_eq!(ptr_taint, ConstantTaint::Pure);
                        new_instructions.push(instruction);
                    }
                    OpCode::ArrayGet {
                        result: _,
                        array: arr,
                        index: _,
                    } => {
                        let arr_taint = function_taint
                            .get_value_taint(arr)
                            .toplevel_taint()
                            .expect_constant();
                        // dynamic array access not supported
                        assert_eq!(arr_taint, ConstantTaint::Pure);
                        new_instructions.push(instruction);
                    }
                    OpCode::ArraySet {
                        result: _,
                        array: arr,
                        index: idx,
                        value: _,
                    } => {
                        let arr_taint = function_taint
                            .get_value_taint(arr)
                            .toplevel_taint()
                            .expect_constant();
                        let idx_taint = function_taint
                            .get_value_taint(idx)
                            .toplevel_taint()
                            .expect_constant();
                        // dynamic array access not supported
                        assert_eq!(arr_taint, ConstantTaint::Pure);
                        assert_eq!(idx_taint, ConstantTaint::Pure);
                        new_instructions.push(instruction);
                    }
                    OpCode::SlicePush {
                        dir: _,
                        result: _,
                        slice: sl,
                        values: _,
                    } => {
                        let slice_taint = function_taint
                            .get_value_taint(sl)
                            .toplevel_taint()
                            .expect_constant();
                        // Slice must always be Pure taint
                        assert_eq!(slice_taint, ConstantTaint::Pure);
                        new_instructions.push(instruction);
                    }
                    OpCode::SliceLen {
                        result: _,
                        slice: sl,
                    } => {
                        let slice_taint = function_taint
                            .get_value_taint(sl)
                            .toplevel_taint()
                            .expect_constant();
                        // Slice must always be Pure taint
                        assert_eq!(slice_taint, ConstantTaint::Pure);
                        new_instructions.push(instruction);
                    }
                    OpCode::Alloc {
                        result: _,
                        elem_type: _,
                        result_annotation: _,
                    } => {
                        new_instructions.push(instruction);
                    }

                    OpCode::Call {
                        results: ret,
                        function: tgt,
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
                            function: tgt,
                            args: args,
                        });
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
                    _ => {
                        panic!("Unhandled instruction {:?}", instruction);
                    }
                }
            }

            match block.get_terminator().cloned() {
                Some(Terminator::JmpIf(cond, if_true, if_false)) => {
                    let cond_taint = function_taint
                        .get_value_taint(cond)
                        .toplevel_taint()
                        .expect_constant();
                    match cond_taint {
                        ConstantTaint::Pure => {}
                        ConstantTaint::Witness => {
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

    fn typify_taint(&self, typ: Type<Empty>, taint: &TaintType) -> Type<ConstantTaint> {
        match (typ.expr, taint) {
            (TypeExpr::Field, TaintType::Primitive(taint)) => Type {
                expr: TypeExpr::Field,
                annotation: taint.expect_constant(),
            },
            (TypeExpr::U(size), TaintType::Primitive(taint)) => Type {
                expr: TypeExpr::U(size),
                annotation: taint.expect_constant(),
            },
            (TypeExpr::Array(inner, size), TaintType::NestedImmutable(top, inner_taint)) => Type {
                expr: TypeExpr::Array(
                    Box::new(self.typify_taint(*inner, inner_taint.as_ref())),
                    size,
                ),
                annotation: top.expect_constant(),
            },
            (TypeExpr::Slice(inner), TaintType::NestedImmutable(top, inner_taint)) => Type {
                expr: TypeExpr::Slice(Box::new(self.typify_taint(*inner, inner_taint.as_ref()))),
                annotation: top.expect_constant(),
            },
            (TypeExpr::Ref(inner), TaintType::NestedMutable(top, inner_taint)) => Type {
                expr: TypeExpr::Ref(Box::new(self.typify_taint(*inner, inner_taint.as_ref()))),
                annotation: top.expect_constant(),
            },
            (TypeExpr::Tuple(child_types), TaintType::Tuple(top, child_taints)) => Type {
                expr: TypeExpr::Tuple(
                    child_types.iter().zip(child_taints.iter()).map(|(child_type, child_taint)| self.typify_taint(child_type.clone(), child_taint)).collect()
                ),
                annotation: top.expect_constant(),
            },
            (tp, taint) => panic!("Unexpected type {:?} with taint {:?}", tp, taint),
        }
    }

    fn pure_taint_for_type(&self, typ: Type<Empty>) -> Type<ConstantTaint> {
        match typ.expr {
            TypeExpr::Field => Type {
                expr: TypeExpr::Field,
                annotation: ConstantTaint::Pure,
            },
            TypeExpr::U(size) => Type {
                expr: TypeExpr::U(size),
                annotation: ConstantTaint::Pure,
            },
            TypeExpr::Array(inner, size) => Type {
                expr: TypeExpr::Array(Box::new(self.pure_taint_for_type(*inner)), size),
                annotation: ConstantTaint::Pure,
            },
            TypeExpr::Slice(inner) => Type {
                expr: TypeExpr::Slice(Box::new(self.pure_taint_for_type(*inner))),
                annotation: ConstantTaint::Pure,
            },
            TypeExpr::Ref(inner) => Type {
                expr: TypeExpr::Ref(Box::new(self.pure_taint_for_type(*inner))),
                annotation: ConstantTaint::Pure,
            },
            TypeExpr::WitnessRef => Type {
                expr: TypeExpr::WitnessRef,
                annotation: ConstantTaint::Pure,
            },
            TypeExpr::Tuple(_elements) => {todo!("Tuples not supported yet")}
        }
    }
}
