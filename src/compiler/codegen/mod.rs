use std::collections::HashMap;

use crate::{
    compiler::{
        analysis::types::{FunctionTypeInfo, TypeInfo},
        flow_analysis::{CFG, FlowAnalysis},
        ir::r#type::{Type, TypeExpr},
        ssa::{
            self, BinaryArithOpKind, Block, BlockId, CmpKind, Const, DMatrix, Endianness, Function, FunctionId, GlobalDef, LookupTarget, MemOp, Radix, SSA, Terminator, TupleIdx, ValueId
        },
        taint_analysis::ConstantTaint,
    },
    vm::{self, bytecode},
};

struct FrameLayouter {
    next_free: usize,
    variables: HashMap<ValueId, usize>,
}

impl FrameLayouter {
    fn new() -> Self {
        Self {
            next_free: 2,
            variables: HashMap::new(),
        }
    }

    fn get_value(&mut self, value: ValueId) -> bytecode::FramePosition {
        bytecode::FramePosition(self.variables[&value])
    }

    fn alloc_value(&mut self, value: ValueId, tp: &Type<ConstantTaint>) -> bytecode::FramePosition {
        self.variables.insert(value, self.next_free);
        self.variables.insert(value, self.next_free);
        let r = self.next_free;
        self.next_free += self.type_size(&tp);
        bytecode::FramePosition(r)
    }

    fn alloc_u64(&mut self, value: ValueId, size: usize) -> bytecode::FramePosition {
        assert!(size <= 64);
        self.variables.insert(value, self.next_free);
        let r = self.next_free;
        self.next_free += 1;
        bytecode::FramePosition(r)
    }

    fn alloc_field(&mut self, value: ValueId) -> bytecode::FramePosition {
        self.variables.insert(value, self.next_free);
        let r = self.next_free;
        self.next_free += bytecode::LIMBS;
        bytecode::FramePosition(r)
    }

    fn alloc_ptr(&mut self, value: ValueId) -> bytecode::FramePosition {
        self.variables.insert(value, self.next_free);
        let r = self.next_free;
        self.next_free += 1;
        bytecode::FramePosition(r)
    }

    fn type_size(&self, tp: &Type<ConstantTaint>) -> usize {
        match tp.expr {
            TypeExpr::Field => bytecode::LIMBS,
            TypeExpr::U(bits) => {
                assert!(bits <= 64);
                1
            }
            TypeExpr::Array(_, _) => 1, // Ptr
            TypeExpr::Slice(_) => 1,    // Ptr
            TypeExpr::WitnessRef => 1,  // Ptr
            TypeExpr::Tuple(_) => 1, // Ptr
            _ => todo!(),
        }
    }

    // This method needs to ensure contiguous storage!
    fn alloc_many_contiguous(
        &mut self,
        values: Vec<(ValueId, &Type<ConstantTaint>)>,
    ) -> bytecode::FramePosition {
        let r = self.next_free;
        for (value, tp) in values {
            self.alloc_value(value, tp);
        }
        bytecode::FramePosition(r)
    }
}

struct EmitterState {
    code: Vec<bytecode::OpCode>,
    block_entrances: HashMap<BlockId, usize>,
    block_exits: HashMap<BlockId, usize>,
}

impl EmitterState {
    fn new() -> Self {
        Self {
            code: Vec::new(),
            block_entrances: HashMap::new(),
            block_exits: HashMap::new(),
        }
    }

    fn push_op(&mut self, op: bytecode::OpCode) {
        self.code.push(op);
    }

    fn enter_block(&mut self, block: BlockId) {
        self.block_entrances.insert(block, self.code.len());
    }

    fn exit_block(&mut self, block: BlockId) {
        self.block_exits.insert(block, self.code.len());
    }
}

struct GlobalFrameLayouter {
    offsets: Vec<usize>,
    sizes: Vec<usize>,
    total_size: usize,
}

impl GlobalFrameLayouter {
    fn new(ssa: &SSA<ConstantTaint>) -> Self {
        let globals = ssa.get_globals();
        let mut offsets = Vec::new();
        let mut sizes = Vec::new();
        let mut next_free = 0usize;
        for global in globals.iter() {
            let size = match global {
                ssa::GlobalDef::Const(ssa::Const::Field(_)) => bytecode::LIMBS,
                ssa::GlobalDef::Const(ssa::Const::U(bits, _)) => {
                    assert!(*bits <= 64);
                    1
                }
                ssa::GlobalDef::Const(ssa::Const::WitnessRef(_)) => 1,
                ssa::GlobalDef::Const(ssa::Const::FnPtr(_)) => panic!("FnPtr globals not supported in codegen"),
                ssa::GlobalDef::Array(_, _) => 1, // Ptr (BoxedValue)
            };
            offsets.push(next_free);
            sizes.push(size);
            next_free += size;
        }
        GlobalFrameLayouter {
            offsets,
            sizes,
            total_size: next_free,
        }
    }

    fn get_offset(&self, global: usize) -> usize {
        self.offsets[global]
    }

    fn get_size(&self, global: usize) -> usize {
        self.sizes[global]
    }
}

pub struct CodeGen {}

impl CodeGen {
    pub fn new() -> Self {
        Self {}
    }

    pub fn run(
        &self,
        ssa: &SSA<ConstantTaint>,
        cfg: &FlowAnalysis,
        type_info: &TypeInfo<ConstantTaint>,
    ) -> bytecode::Program {
        let global_layouter = GlobalFrameLayouter::new(ssa);

        let function = ssa.get_main();
        let function = self.run_function(
            function,
            cfg.get_function_cfg(ssa.get_main_id()),
            type_info.get_function(ssa.get_main_id()),
            &global_layouter,
        );

        let mut functions = vec![function];

        let mut function_ids = HashMap::new();
        function_ids.insert(ssa.get_main_id(), 0);

        let mut cur_fn_begin = functions[0].code.len();

        for (function_id, function) in ssa.iter_functions() {
            if *function_id == ssa.get_main_id() {
                continue;
            }
            let function = self.run_function(
                function,
                cfg.get_function_cfg(*function_id),
                type_info.get_function(*function_id),
                &global_layouter,
            );
            function_ids.insert(*function_id, cur_fn_begin);
            cur_fn_begin += function.code.len();
            functions.push(function);
        }

        let mut cur_fun_off = 0;
        for function in functions.iter_mut() {
            for op in function.code.iter_mut() {
                match op {
                    bytecode::OpCode::Call { func, .. } => {
                        func.0 = *function_ids.get(&FunctionId(func.0 as u64)).unwrap() as isize;
                    }
                    bytecode::OpCode::Jmp { target } => {
                        target.0 += cur_fun_off as isize;
                    }
                    bytecode::OpCode::JmpIf { if_t, if_f, .. } => {
                        if_t.0 += cur_fun_off as isize;
                        if_f.0 += cur_fun_off as isize;
                    }
                    _ => {}
                }
            }
            cur_fun_off += function.code.len();
        }

        bytecode::Program {
            functions: functions,
            global_frame_size: global_layouter.total_size,
        }
    }

    fn run_function(
        &self,
        function: &Function<ConstantTaint>,
        cfg: &CFG,
        type_info: &FunctionTypeInfo<ConstantTaint>,
        global_layouter: &GlobalFrameLayouter,
    ) -> bytecode::Function {
        let mut layouter = FrameLayouter::new();
        let entry = function.get_entry();
        let mut emitter = EmitterState::new();

        // Entry block params need to be allocated at the beginning of the frame (after return address and return data pointer)
        for (param, tp) in entry.get_parameters() {
            layouter.alloc_value(*param, tp);
        }

        // Consts get dumped in the first block
        for (val, constant) in function.iter_consts() {
            match constant {
                Const::U(u, v) => emitter.push_op(bytecode::OpCode::MovConst {
                    res: layouter.alloc_u64(*val, *u),
                    val: *v as u64,
                }),
                Const::Field(v) => {
                    let start = layouter.alloc_field(*val);
                    for i in 0..bytecode::LIMBS {
                        // pushing in montgomery form
                        emitter.push_op(bytecode::OpCode::MovConst {
                            res: start.offset(i as isize),
                            val: v.0.0[i],
                        })
                    }
                }
                Const::WitnessRef(v) => {
                    emitter.push_op(bytecode::OpCode::WitnessRefAlloc {
                        res: layouter.alloc_ptr(*val),
                        data: *v,
                    });
                }
                Const::FnPtr(_) => panic!("FnPtr constants not supported in codegen"),
            }
        }

        self.run_block_body(
            function,
            function.get_entry_id(),
            entry,
            type_info,
            &mut layouter,
            &mut emitter,
            global_layouter,
        );

        for block_id in cfg.get_domination_pre_order() {
            if block_id == function.get_entry_id() {
                continue;
            }
            let block = function.get_block(block_id);
            for (param, tp) in block.get_parameters() {
                layouter.alloc_value(*param, tp);
            }
            self.run_block_body(
                function,
                block_id,
                block,
                type_info,
                &mut layouter,
                &mut emitter,
                global_layouter,
            );
        }

        for (block_id, block) in function.get_blocks() {
            let mut block_exit_start: usize = emitter.block_exits[&block_id];
            match block.get_terminator().unwrap() {
                Terminator::Jmp(tgt, args) => {
                    let params = function.get_block(*tgt).get_parameters();
                    for (arg, (param, tp)) in args.iter().zip(params) {
                        emitter.code[block_exit_start] = bytecode::OpCode::MovFrame {
                            size: layouter.type_size(tp),
                            target: layouter.get_value(*param),
                            source: layouter.get_value(*arg),
                        };
                        block_exit_start += 1;
                    }
                    emitter.code[block_exit_start] = bytecode::OpCode::Jmp {
                        target: bytecode::JumpTarget(
                            *emitter.block_entrances.get(&tgt).unwrap() as isize
                        ),
                    };
                }
                Terminator::JmpIf(cond, if_t, if_f) => {
                    emitter.code[block_exit_start] = bytecode::OpCode::JmpIf {
                        cond: layouter.get_value(*cond),
                        if_t: bytecode::JumpTarget(
                            *emitter.block_entrances.get(&if_t).unwrap() as isize
                        ),
                        if_f: bytecode::JumpTarget(
                            *emitter.block_entrances.get(&if_f).unwrap() as isize
                        ),
                    };
                }
                Terminator::Return(_) => {
                    // Nothing to do, returns are correct right away
                }
            }
        }

        bytecode::Function {
            name: function.get_name().to_string(),
            frame_size: layouter.next_free,
            code: emitter.code,
        }
    }

    fn run_block_body(
        &self,
        _function: &Function<ConstantTaint>,
        block_id: BlockId,
        block: &Block<ConstantTaint>,
        type_info: &FunctionTypeInfo<ConstantTaint>,
        layouter: &mut FrameLayouter,
        emitter: &mut EmitterState,
        global_layouter: &GlobalFrameLayouter,
    ) {
        emitter.enter_block(block_id);
        for instruction in block.get_instructions() {
            match instruction {
                ssa::OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::Add,
                    result: val,
                    lhs: op1,
                    rhs: op2,
                } => match &type_info.get_value_type(*val).expr {
                    TypeExpr::Field => {
                        let result = layouter.alloc_field(*val);
                        emitter.push_op(bytecode::OpCode::AddField {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    TypeExpr::U(bits) => {
                        let result = layouter.alloc_u64(*val, *bits);
                        emitter.push_op(bytecode::OpCode::AddU64 {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    TypeExpr::WitnessRef => {
                        let result = layouter.alloc_ptr(*val);
                        emitter.push_op(bytecode::OpCode::AddBoxed {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    t => panic!("Unsupported type for addition: {:?}", t),
                },
                ssa::OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::Sub,
                    result: val,
                    lhs: op1,
                    rhs: op2,
                } => match &type_info.get_value_type(*val).expr {
                    TypeExpr::Field => {
                        let result = layouter.alloc_field(*val);
                        emitter.push_op(bytecode::OpCode::SubField {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    TypeExpr::U(bits) => {
                        let result = layouter.alloc_u64(*val, *bits);
                        emitter.push_op(bytecode::OpCode::SubU64 {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    t => panic!("Unsupported type for subtraction: {:?}", t),
                },
                ssa::OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::Div,
                    result: val,
                    lhs: op1,
                    rhs: op2,
                } => match &type_info.get_value_type(*val).expr {
                    TypeExpr::Field => {
                        let result = layouter.alloc_field(*val);
                        emitter.push_op(bytecode::OpCode::DivField {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    TypeExpr::U(bits) => {
                        let result = layouter.alloc_u64(*val, *bits);
                        emitter.push_op(bytecode::OpCode::DivU64 {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    t => panic!("Unsupported type for division: {:?}", t),
                },
                ssa::OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::Mul,
                    result: val,
                    lhs: op1,
                    rhs: op2,
                } => match &type_info.get_value_type(*val).expr {
                    TypeExpr::Field => {
                        let result = layouter.alloc_field(*val);
                        emitter.push_op(bytecode::OpCode::MulField {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    TypeExpr::U(bits) => {
                        let result = layouter.alloc_u64(*val, *bits);
                        emitter.push_op(bytecode::OpCode::MulU64 {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    t => panic!("Unsupported type for multiplication: {:?}", t),
                },
                ssa::OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::And,
                    result: val,
                    lhs: op1,
                    rhs: op2,
                } => {
                    match &type_info.get_value_type(*val).expr {
                        TypeExpr::Field => {
                            panic!("Unsupported: field and");
                            // let result = layouter.alloc_field(*val);
                            // emitter.push_op(bytecode::OpCode::MulF(
                            //     result,
                            //     layouter.get_value(*op1),
                            //     layouter.get_value(*op2),
                            // ));
                        }
                        TypeExpr::U(bits) => {
                            let result = layouter.alloc_u64(*val, *bits);
                            emitter.push_op(bytecode::OpCode::AndU64 {
                                res: result,
                                a: layouter.get_value(*op1),
                                b: layouter.get_value(*op2),
                            });
                        }
                        t => panic!("Unsupported type for multiplication: {:?}", t),
                    }
                }
                ssa::OpCode::Cmp {
                    kind: CmpKind::Lt,
                    result: val,
                    lhs: op1,
                    rhs: op2,
                } => match &type_info.get_value_type(*val).expr {
                    TypeExpr::U(bits) => {
                        let result = layouter.alloc_u64(*val, *bits);
                        emitter.push_op(bytecode::OpCode::LtU64 {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    t => panic!("Unsupported type for comparison: {:?}", t),
                },
                ssa::OpCode::Cmp {
                    kind: CmpKind::Eq,
                    result: val,
                    lhs: op1,
                    rhs: op2,
                } => match &type_info.get_value_type(*val).expr {
                    TypeExpr::U(bits) => {
                        let result = layouter.alloc_u64(*val, *bits);
                        emitter.push_op(bytecode::OpCode::EqU64 {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    t => panic!("Unsupported type for comparison: {:?}", t),
                },
                ssa::OpCode::Truncate {
                    result: val,
                    value: op,
                    to_bits,
                    from_bits,
                } => {
                    let result = layouter.alloc_u64(*val, 64);
                    assert!(*to_bits <= 64);
                    let in_type = type_info.get_value_type(*op);
                    if in_type.is_field() {
                        emitter.push_op(bytecode::OpCode::TruncateFToU {
                            res: result,
                            a: layouter.get_value(*op),
                            to_bits: *to_bits as u64,
                        });
                    } else {
                        assert!(*from_bits <= 64);
                        emitter.push_op(bytecode::OpCode::TruncateU64 {
                            res: result,
                            a: layouter.get_value(*op),
                            to_bits: *to_bits as u64,
                        });
                    }
                }
                ssa::OpCode::Cast {
                    result: r,
                    value: v,
                    target: tgt,
                } => {
                    if *tgt == ssa::CastTarget::Nop {
                        let pos = layouter.variables[v];
                        layouter.variables.insert(*r, pos);
                        continue;
                    }
                    let result = layouter.alloc_value(*r, &type_info.get_value_type(*r));
                    let l_type = type_info.get_value_type(*v);
                    let r_type = type_info.get_value_type(*r);
                    match (&l_type.expr, &r_type.expr) {
                        (TypeExpr::U(_), TypeExpr::U(_)) => {
                            emitter.push_op(bytecode::OpCode::MovFrame {
                                target: result,
                                source: layouter.get_value(*v),
                                size: layouter.type_size(&l_type),
                            })
                        }
                        (TypeExpr::Field, TypeExpr::U(_)) => {
                            emitter.push_op(bytecode::OpCode::CastFieldToU64 {
                                res: result,
                                a: layouter.get_value(*v),
                            });
                        }
                        (TypeExpr::U(_), TypeExpr::Field) => {
                            emitter.push_op(bytecode::OpCode::CastU64ToField {
                                res: result,
                                a: layouter.get_value(*v),
                            });
                        }
                        _ => panic!("Unsupported cast: {:?} -> {:?}", l_type, r_type),
                    }
                    // TODO: Implement this, it _will_ break
                }
                ssa::OpCode::Not {
                    result: r,
                    value: v,
                } => {
                    let result = layouter.alloc_value(*r, &type_info.get_value_type(*r));
                    emitter.push_op(bytecode::OpCode::NotU64 {
                        res: result,
                        a: layouter.get_value(*v),
                    });
                }
                ssa::OpCode::Constrain { a, b, c } => {
                    let a_type = type_info.get_value_type(*a);
                    let b_type = type_info.get_value_type(*b);
                    let c_type = type_info.get_value_type(*c);
                    if !a_type.is_field() || !b_type.is_field() || !c_type.is_field() {
                        panic!(
                            "Unsupported type for constrain: {:?}, {:?}, {:?}",
                            a_type, b_type, c_type
                        );
                    }
                    emitter.push_op(bytecode::OpCode::R1C {
                        a: layouter.get_value(*a),
                        b: layouter.get_value(*b),
                        c: layouter.get_value(*c),
                    });
                }
                ssa::OpCode::WriteWitness {
                    result: None,
                    value: v,
                    witness_annotation: _,
                } => {
                    
                    emitter.push_op(bytecode::OpCode::WriteWitness {
                        val: layouter.get_value(*v),
                    });
                }
                ssa::OpCode::ArrayGet {
                    result: r,
                    array: arr,
                    index: idx,
                } => {
                    let res = layouter.alloc_value(*r, &type_info.get_value_type(*r));
                    emitter.push_op(bytecode::OpCode::ArrayGet {
                        res,
                        array: layouter.get_value(*arr),
                        index: layouter.get_value(*idx),
                        stride: layouter
                            .type_size(&type_info.get_value_type(*arr).get_array_element()),
                    });
                }
                ssa::OpCode::TupleProj {
                    result: r,
                    tuple: t,
                    idx,
                } => {
                    if let TupleIdx::Static(i) = idx {
                        let res = layouter.alloc_value(*r, &type_info.get_value_type(*r));
                        emitter.push_op(bytecode::OpCode::TupleProj {
                            res,
                            tuple: layouter.get_value(*t),
                            index: *i as u64,
                            child_sizes: type_info.get_value_type(*t).get_tuple_elements().iter().map(
                                |elem_type| layouter.type_size(elem_type)    
                            ).collect(),
                        });
                    } else {
                        panic!("Dynamic tuple indexing should not appear here");
                    }
                }
                ssa::OpCode::ArraySet {
                    result: r,
                    array: arr,
                    index: idx,
                    value: val,
                } => {
                    let res = layouter.alloc_value(*r, &type_info.get_value_type(*r));
                    emitter.push_op(bytecode::OpCode::ArraySet {
                        res,
                        array: layouter.get_value(*arr),
                        index: layouter.get_value(*idx),
                        source: layouter.get_value(*val),
                        stride: layouter
                            .type_size(&type_info.get_value_type(*arr).get_array_element()),
                    });
                }
                ssa::OpCode::SlicePush {
                    result: _r,
                    slice: _sl,
                    values: _vals,
                    dir: _,
                } => {
                    panic!("SlicePush bytecode opcode not yet implemented");
                }
                ssa::OpCode::SliceLen {
                    result: _r,
                    slice: _sl,
                } => {
                    panic!("SliceLen bytecode opcode not yet implemented");
                }
                ssa::OpCode::MkSeq {
                    result: r,
                    elems: vals,
                    seq_type: _,
                    elem_type: eltype,
                } => {
                    let res = layouter.alloc_value(*r, &type_info.get_value_type(*r));
                    let args = vals
                        .iter()
                        .map(|a| layouter.get_value(*a))
                        .collect::<Vec<_>>();
                    let is_ptr = eltype.is_heap_allocated();
                    let stride = layouter.type_size(eltype);
                    emitter.push_op(bytecode::OpCode::ArrayAlloc {
                        res,
                        stride: layouter.type_size(eltype),
                        meta: vm::array::BoxedLayout::array(args.len() * stride, is_ptr),
                        items: args,
                    });
                }
                ssa::OpCode::MkTuple {
                    result,
                    elems,
                    element_types
                } => {
                    assert!(
                        element_types.len() <= 14,
                        "Struct has {} fields, but maximum is 14",
                        element_types.len()
                    );
                    let res = layouter.alloc_value(*result, &type_info.get_value_type(*result));
                    let fields = elems
                        .iter()
                        .map(|a| layouter.get_value(*a))
                        .collect::<Vec<_>>();
                    let field_sizes: Vec<usize> = element_types.iter().map(|elem_type| {
                        let size = layouter.type_size(elem_type);
                        assert!(
                            size <= 8,
                            "Struct field has {} bits, but maximum is 512",
                            size * 64
                        );
                        size
                    }).collect();
                    let reference_counting = element_types.iter().map(
                        |elem_type| elem_type.is_heap_allocated()
                    ).collect();
                    emitter.push_op(bytecode::OpCode::TupleAlloc {
                        res,
                        meta: vm::array::BoxedLayout::new_struct(
                            field_sizes,
                            reference_counting,
                        ),
                        fields,
                    });
                }
                ssa::OpCode::Call {
                    results: r,
                    function: ssa::CallTarget::Static(fnid),
                    args: params,
                } => {
                    let r = layouter.alloc_many_contiguous(
                        r.iter()
                            .map(|a| (*a, type_info.get_value_type(*a)))
                            .collect(),
                    );
                    let args = params
                        .iter()
                        .map(|a| {
                            (
                                layouter.type_size(&type_info.get_value_type(*a)),
                                layouter.get_value(*a),
                            )
                        })
                        .collect::<Vec<_>>();
                    emitter.push_op(bytecode::OpCode::Call {
                        func: bytecode::JumpTarget(fnid.0 as isize),
                        args: args,
                        ret: r,
                    });
                }
                ssa::OpCode::Call {
                    function: ssa::CallTarget::Dynamic(_),
                    ..
                } => {
                    panic!("Dynamic call targets are not supported in codegen")
                }
                ssa::OpCode::MemOp {
                    kind: MemOp::Drop,
                    value: r,
                } => {
                    // assert!(type_info.get_value_type(*r).is_array_or_slice());
                    emitter.push_op(bytecode::OpCode::DecRc {
                        array: layouter.get_value(*r),
                    });
                }
                ssa::OpCode::MemOp {
                    kind: MemOp::Bump(size),
                    value: r,
                } => {
                    // assert!(type_info.get_value_type(*r).is_array_or_slice());
                    emitter.push_op(bytecode::OpCode::IncRc {
                        array: layouter.get_value(*r),
                        amount: *size as u64,
                    });
                }
                ssa::OpCode::AssertEq { lhs: _, rhs: _ } => {
                    // TODO: Implement this
                }
                ssa::OpCode::ToBits {
                    result: r,
                    value,
                    endianness: Endianness::Little,
                    count,
                } => {
                    emitter.push_op(bytecode::OpCode::ToBitsLe {
                        res: layouter.alloc_value(*r, &type_info.get_value_type(*r)),
                        val: layouter.get_value(*value),
                        count: *count as u64,
                    });
                }
                ssa::OpCode::ToRadix {
                    result: r,
                    value: v,
                    radix: Radix::Bytes,
                    endianness: Endianness::Big,
                    count: c,
                } => {
                    assert!(
                        type_info.get_value_type(*v).is_field(),
                        "TODO: Implement toRadix for U-values"
                    );
                    assert!(*c <= 8, "TODO: Implement toRadix for > 8 bytes");
                    emitter.push_op(bytecode::OpCode::ToBytesBeLt8 {
                        val: layouter.get_value(*v),
                        count: *c as u64,
                        res: layouter.alloc_value(*r, &type_info.get_value_type(*r)),
                    })
                }
                ssa::OpCode::ToRadix {
                    result: _,
                    value: _,
                    radix: _,
                    endianness: _,
                    count: _,
                } => {
                    panic!("ToRadix not yet implemented");
                }
                ssa::OpCode::NextDCoeff { result: out } => {
                    let v = layouter.alloc_field(*out);
                    emitter.push_op(bytecode::OpCode::NextDCoeff { v });
                }
                ssa::OpCode::BumpD {
                    matrix: m,
                    variable: var,
                    sensitivity: coeff,
                } => {
                    let v = layouter.get_value(*var);
                    let coeff = layouter.get_value(*coeff);
                    emitter.push_op(match m {
                        DMatrix::A => bytecode::OpCode::BumpDa { v, coeff },
                        DMatrix::B => bytecode::OpCode::BumpDb { v, coeff },
                        DMatrix::C => bytecode::OpCode::BumpDc { v, coeff },
                    });
                }
                ssa::OpCode::FreshWitness {
                    result: r,
                    result_type: tp,
                } => {
                    assert!(matches!(tp.expr, TypeExpr::WitnessRef));
                    emitter.push_op(bytecode::OpCode::FreshWitness {
                        res: layouter.alloc_ptr(*r),
                    });
                }
                ssa::OpCode::PureToWitnessRef {
                    result: r,
                    value: v,
                    result_annotation: _,
                } => {
                    emitter.push_op(bytecode::OpCode::PureToWitnessRef {
                        res: layouter.alloc_ptr(*r),
                        v: layouter.get_value(*v),
                    });
                }
                ssa::OpCode::UnboxField {
                    result: r,
                    value: v,
                } => {
                    emitter.push_op(bytecode::OpCode::UnboxField {
                        res: layouter.alloc_field(*r),
                        v: layouter.get_value(*v),
                    });
                }
                ssa::OpCode::MulConst {
                    result: r,
                    const_val: c,
                    var: v,
                } => {
                    emitter.push_op(bytecode::OpCode::MulConst {
                        res: layouter.alloc_ptr(*r),
                        coeff: layouter.get_value(*c),
                        v: layouter.get_value(*v),
                    });
                }
                ssa::OpCode::Rangecheck {
                    value: val,
                    max_bits,
                } => {
                    emitter.push_op(bytecode::OpCode::Rangecheck {
                        val: layouter.get_value(*val),
                        max_bits: *max_bits,
                    });
                }
                ssa::OpCode::Lookup {
                    target: LookupTarget::Rangecheck(8),
                    keys,
                    results,
                } => {
                    assert!(keys.len() == 1);
                    assert!(results.len() == 0);
                    assert!(type_info.get_value_type(keys[0]).is_field());
                    emitter.push_op(bytecode::OpCode::Rngchk8Field {
                        val: layouter.get_value(keys[0]),
                    });
                }
                ssa::OpCode::DLookup {
                    target: LookupTarget::Rangecheck(8),
                    keys,
                    results,
                } => {
                    assert!(keys.len() == 1);
                    assert!(results.len() == 0);
                    assert!(type_info.get_value_type(keys[0]).is_witness_ref());
                    emitter.push_op(bytecode::OpCode::Drngchk8Field {
                        val: layouter.get_value(keys[0]),
                    });
                }
                ssa::OpCode::Todo { payload, .. } => {
                    panic!("Todo opcode encountered in Codegen: {}", payload);
                }
                ssa::OpCode::InitGlobal { global, value } => {
                    emitter.push_op(bytecode::OpCode::InitGlobal {
                        src: layouter.get_value(*value),
                        global_offset: global_layouter.get_offset(*global),
                        size: global_layouter.get_size(*global),
                    });
                }
                ssa::OpCode::DropGlobal { global } => {
                    emitter.push_op(bytecode::OpCode::DropGlobal {
                        global_offset: global_layouter.get_offset(*global),
                    });
                }
                ssa::OpCode::ReadGlobal { result: r, offset, result_type: _ } => {
                    let global_idx = *offset as usize;
                    let res = layouter.alloc_value(*r, &type_info.get_value_type(*r));
                    emitter.push_op(bytecode::OpCode::ReadGlobal {
                        res,
                        global_offset: global_layouter.get_offset(global_idx),
                        size: global_layouter.get_size(global_idx),
                    });
                }
                other => panic!("Unsupported instruction: {:?}", other),
            }
        }
        emitter.exit_block(block_id);
        match block.get_terminator().unwrap() {
            Terminator::Jmp(_, params) => {
                emitter.push_op(bytecode::OpCode::Nop {});
                for _ in 0..params.len() {
                    emitter.push_op(bytecode::OpCode::Nop {});
                }
            }
            Terminator::JmpIf(_, _, _) => {
                emitter.push_op(bytecode::OpCode::Nop {});
            }
            Terminator::Return(params) => {
                let mut offset = 0;
                for param in params {
                    let size = layouter.type_size(&type_info.get_value_type(*param));
                    emitter.push_op(bytecode::OpCode::WritePtr {
                        ptr: bytecode::FramePosition::return_data_ptr(),
                        offset,
                        src: layouter.get_value(*param),
                        size,
                    });
                    offset += size as isize;
                }
                emitter.push_op(bytecode::OpCode::Ret {});
            }
        }
    }
}
