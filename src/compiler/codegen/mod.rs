use std::collections::HashMap;

use crate::{
    compiler::{
        analysis::types::{FunctionTypeInfo, TypeInfo},
        flow_analysis::{CFG, FlowAnalysis},
        ir::r#type::{Type, TypeExpr},
        ssa::{
            self, BinaryArithOpKind, BlockId, CmpKind, DMatrix, Endianness, FunctionId, HLBlock,
            HLFunction, HLSSA, LookupTarget, MemOp, Radix, Terminator, ValueId,
        },
    },
    vm::{self, bytecode},
};

/// Returns (stride, elem_kind) for an array element type in a lookup opcode.
fn lookup_elem_kind(elem_type: &Type) -> (usize, usize) {
    match &elem_type.expr {
        TypeExpr::Field => (bytecode::LIMBS, bytecode::ELEM_FIELD),
        TypeExpr::U(bits) | TypeExpr::I(bits) => {
            assert!(
                *bits <= 64,
                "Array lookup unsupported for {elem_type} (>64 bits)"
            );
            (1, bytecode::ELEM_WORD)
        }
        TypeExpr::WitnessOf(inner) => {
            let inner_kind = lookup_elem_kind(inner);
            assert!(
                inner_kind.1 != bytecode::ELEM_WITNESS,
                "Nested WitnessOf in array lookup element type: {elem_type}"
            );
            (1, bytecode::ELEM_WITNESS)
        }
        _ => panic!("Unsupported array element type in lookup: {elem_type}"),
    }
}

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

    fn alloc_value(&mut self, value: ValueId, tp: &Type) -> bytecode::FramePosition {
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

    fn alloc_temp_field(&mut self) -> bytecode::FramePosition {
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

    fn type_size(&self, tp: &Type) -> usize {
        match tp.expr {
            TypeExpr::Field => bytecode::LIMBS,
            TypeExpr::U(bits) | TypeExpr::I(bits) => {
                assert!(bits <= 64);
                1
            }
            TypeExpr::Array(_, _) => 1,  // Ptr
            TypeExpr::Slice(_) => 1,     // Ptr
            TypeExpr::WitnessOf(_) => 1, // Ptr
            TypeExpr::Tuple(_) => 1,     // Ptr
            TypeExpr::Ref(_) => 1,       // Ptr
            _ => todo!(),
        }
    }

    fn alloc_scratch(&mut self, count: usize) -> bytecode::FramePosition {
        let r = self.next_free;
        self.next_free += count;
        bytecode::FramePosition(r)
    }

    // This method needs to ensure contiguous storage!
    fn alloc_many_contiguous(&mut self, values: Vec<(ValueId, &Type)>) -> bytecode::FramePosition {
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
    fn new(ssa: &HLSSA) -> Self {
        let global_types = ssa.get_global_types();
        let mut offsets = Vec::new();
        let mut sizes = Vec::new();
        let mut next_free = 0usize;
        for typ in global_types.iter() {
            let size = Self::type_frame_size(typ);
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

    fn type_frame_size(typ: &Type) -> usize {
        use crate::compiler::ir::r#type::TypeExpr;
        match &typ.expr {
            TypeExpr::Field => bytecode::LIMBS,
            TypeExpr::U(bits) | TypeExpr::I(bits) => {
                assert!(*bits <= 64);
                1
            }
            // Heap-allocated types are pointers (1 word)
            _ => 1,
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

    pub fn run(&self, ssa: &HLSSA, cfg: &FlowAnalysis, type_info: &TypeInfo) -> bytecode::Program {
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
            functions,
            global_frame_size: global_layouter.total_size,
        }
    }

    fn run_function(
        &self,
        function: &HLFunction,
        cfg: &CFG,
        type_info: &FunctionTypeInfo,
        global_layouter: &GlobalFrameLayouter,
    ) -> bytecode::Function {
        let mut layouter = FrameLayouter::new();
        let entry = function.get_entry();
        let mut emitter = EmitterState::new();

        // Entry block params need to be allocated at the beginning of the frame (after return address and return data pointer)
        for (param, tp) in entry.get_parameters() {
            layouter.alloc_value(*param, tp);
        }

        self.run_block_body(
            function,
            function.get_entry_id(),
            entry,
            type_info,
            cfg,
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
                cfg,
                &mut layouter,
                &mut emitter,
                global_layouter,
            );
        }

        // Reserve scratch for loop back-edge.
        let max_loop_scratch = function
            .get_blocks()
            .filter(|(block_id, _)| cfg.is_loop_entry(**block_id))
            .map(|(_, block)| {
                block
                    .get_parameters()
                    .map(|(_, tp)| layouter.type_size(tp))
                    .sum::<usize>()
            })
            .max()
            .unwrap_or(0);
        let scratch_base = layouter.alloc_scratch(max_loop_scratch);

        for (block_id, block) in function.get_blocks() {
            let mut exit_instruction_cursor: usize = emitter.block_exits[&block_id];
            match block.get_terminator().unwrap() {
                Terminator::Jmp(tgt, args) => {
                    if cfg.dominates(*tgt, *block_id) {
                        // Back-edge: copy through scratch to avoid clobbering
                        let mut scratch_frame_offset = 0isize;
                        for (arg, (_param, tp)) in
                            args.iter().zip(function.get_block(*tgt).get_parameters())
                        {
                            let size = layouter.type_size(tp);
                            emitter.code[exit_instruction_cursor] = bytecode::OpCode::MovFrame {
                                size,
                                target: scratch_base.offset(scratch_frame_offset),
                                source: layouter.get_value(*arg),
                            };
                            exit_instruction_cursor += 1;
                            scratch_frame_offset += size as isize;
                        }
                        let mut scratch_frame_offset = 0isize;
                        for (_arg, (param, tp)) in
                            args.iter().zip(function.get_block(*tgt).get_parameters())
                        {
                            let size = layouter.type_size(tp);
                            emitter.code[exit_instruction_cursor] = bytecode::OpCode::MovFrame {
                                size,
                                target: layouter.get_value(*param),
                                source: scratch_base.offset(scratch_frame_offset),
                            };
                            exit_instruction_cursor += 1;
                            scratch_frame_offset += size as isize;
                        }
                    } else {
                        for (arg, (param, tp)) in
                            args.iter().zip(function.get_block(*tgt).get_parameters())
                        {
                            let size = layouter.type_size(tp);
                            emitter.code[exit_instruction_cursor] = bytecode::OpCode::MovFrame {
                                size,
                                target: layouter.get_value(*param),
                                source: layouter.get_value(*arg),
                            };
                            exit_instruction_cursor += 1;
                        }
                    }
                    emitter.code[exit_instruction_cursor] = bytecode::OpCode::Jmp {
                        target: bytecode::JumpTarget(
                            *emitter.block_entrances.get(&tgt).unwrap() as isize
                        ),
                    };
                }
                Terminator::JmpIf(cond, if_t, if_f) => {
                    emitter.code[exit_instruction_cursor] = bytecode::OpCode::JmpIf {
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
        _function: &HLFunction,
        block_id: BlockId,
        block: &HLBlock,
        type_info: &FunctionTypeInfo,
        cfg: &CFG,
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
                    TypeExpr::U(bits) | TypeExpr::I(bits) => {
                        let result = layouter.alloc_u64(*val, *bits);
                        emitter.push_op(bytecode::OpCode::AddInt {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                            bits: *bits as u64,
                        });
                    }
                    TypeExpr::WitnessOf(_) => {
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
                    TypeExpr::U(bits) | TypeExpr::I(bits) => {
                        let result = layouter.alloc_u64(*val, *bits);
                        emitter.push_op(bytecode::OpCode::SubInt {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                            bits: *bits as u64,
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
                    TypeExpr::U(_bits) => {
                        let result = layouter.alloc_u64(*val, *_bits);
                        emitter.push_op(bytecode::OpCode::DivU64 {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    TypeExpr::I(_) => panic!("Signed div not yet implemented"),
                    t => panic!("Unsupported type for division: {:?}", t),
                },
                ssa::OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::Mod,
                    result: val,
                    lhs: op1,
                    rhs: op2,
                } => match &type_info.get_value_type(*val).expr {
                    TypeExpr::Field => {
                        panic!("Modulo is not defined on field elements")
                    }
                    TypeExpr::U(_bits) => {
                        let result = layouter.alloc_u64(*val, *_bits);
                        emitter.push_op(bytecode::OpCode::ModU64 {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    TypeExpr::I(_) => panic!("Signed mod not yet implemented"),
                    t => panic!("Unsupported type for modulo: {:?}", t),
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
                    TypeExpr::U(bits) | TypeExpr::I(bits) => {
                        let result = layouter.alloc_u64(*val, *bits);
                        emitter.push_op(bytecode::OpCode::MulInt {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                            bits: *bits as u64,
                        });
                    }
                    t => panic!("Unsupported type for multiplication: {:?}", t),
                },
                ssa::OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::And,
                    result: val,
                    lhs: op1,
                    rhs: op2,
                } => match &type_info.get_value_type(*val).expr {
                    TypeExpr::Field => {
                        panic!("Unsupported: field and");
                    }
                    TypeExpr::U(bits) | TypeExpr::I(bits) => {
                        let result = layouter.alloc_u64(*val, *bits);
                        emitter.push_op(bytecode::OpCode::AndU64 {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    t => panic!("Unsupported type for bitwise and: {:?}", t),
                },
                ssa::OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::Or,
                    result: val,
                    lhs: op1,
                    rhs: op2,
                } => match &type_info.get_value_type(*val).expr {
                    TypeExpr::U(bits) | TypeExpr::I(bits) => {
                        let result = layouter.alloc_u64(*val, *bits);
                        emitter.push_op(bytecode::OpCode::OrU64 {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    t => panic!("Unsupported type for bitwise or: {:?}", t),
                },
                ssa::OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::Xor,
                    result: val,
                    lhs: op1,
                    rhs: op2,
                } => match &type_info.get_value_type(*val).expr {
                    TypeExpr::U(bits) | TypeExpr::I(bits) => {
                        let result = layouter.alloc_u64(*val, *bits);
                        emitter.push_op(bytecode::OpCode::XorU64 {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    t => panic!("Unsupported type for bitwise xor: {:?}", t),
                },
                ssa::OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::Shl,
                    result: val,
                    lhs: op1,
                    rhs: op2,
                } => match &type_info.get_value_type(*val).expr {
                    TypeExpr::U(bits) | TypeExpr::I(bits) => {
                        let result = layouter.alloc_u64(*val, *bits);
                        emitter.push_op(bytecode::OpCode::ShlU64 {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                            bits: *bits as u64,
                        });
                    }
                    t => panic!("Unsupported type for shift left: {:?}", t),
                },
                ssa::OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::Shr,
                    result: val,
                    lhs: op1,
                    rhs: op2,
                } => match &type_info.get_value_type(*val).expr {
                    TypeExpr::U(bits) | TypeExpr::I(bits) => {
                        let result = layouter.alloc_u64(*val, *bits);
                        emitter.push_op(bytecode::OpCode::UshrU64 {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    t => panic!("Unsupported type for shift right: {:?}", t),
                },
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
                    TypeExpr::I(_) => panic!("Signed Lt not yet implemented"),
                    t => panic!("Unsupported type for comparison: {:?}", t),
                },
                ssa::OpCode::Cmp {
                    kind: CmpKind::Eq,
                    result: val,
                    lhs: op1,
                    rhs: op2,
                } => match &type_info.get_value_type(*val).expr {
                    TypeExpr::U(bits) | TypeExpr::I(bits) => {
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
                    assert!(*to_bits <= 64);
                    let in_type = type_info.get_value_type(*op);
                    if in_type.is_field() {
                        // TruncateFToU writes a full Field (4 limbs), so allocate Field-sized output
                        let result = layouter.alloc_field(*val);
                        emitter.push_op(bytecode::OpCode::TruncateFToU {
                            res: result,
                            a: layouter.get_value(*op),
                            to_bits: *to_bits as u64,
                        });
                    } else {
                        let result = layouter.alloc_u64(*val, 64);
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
                    let l_type = type_info.get_value_type(*v);
                    let r_type = type_info.get_value_type(*r);
                    if matches!(tgt, ssa::CastTarget::WitnessOf) {
                        // PureToWitnessRef reads a Field (4 u64s) from the frame.
                        // If the source is not Field-sized, cast to Field first.
                        let field_pos = if l_type.expr != TypeExpr::Field {
                            let tmp = layouter.alloc_temp_field();
                            emitter.push_op(bytecode::OpCode::CastU64ToField {
                                res: tmp,
                                a: layouter.get_value(*v),
                            });
                            tmp
                        } else {
                            layouter.get_value(*v)
                        };
                        emitter.push_op(bytecode::OpCode::PureToWitnessRef {
                            res: layouter.alloc_ptr(*r),
                            v: field_pos,
                        });
                        continue;
                    }
                    let is_nop =
                        matches!(tgt, ssa::CastTarget::Nop | ssa::CastTarget::ArrayToSlice)
                            || l_type.expr == r_type.expr;
                    if is_nop {
                        let pos = layouter.variables[v];
                        layouter.variables.insert(*r, pos);
                        continue;
                    }
                    let result = layouter.alloc_value(*r, &r_type);
                    match (&l_type.expr, &r_type.expr) {
                        (TypeExpr::U(_) | TypeExpr::I(_), TypeExpr::U(_) | TypeExpr::I(_)) => {
                            emitter.push_op(bytecode::OpCode::MovFrame {
                                target: result,
                                source: layouter.get_value(*v),
                                size: layouter.type_size(&l_type),
                            })
                        }
                        (TypeExpr::Field, TypeExpr::U(_) | TypeExpr::I(_)) => {
                            emitter.push_op(bytecode::OpCode::CastFieldToU64 {
                                res: result,
                                a: layouter.get_value(*v),
                            });
                        }
                        (TypeExpr::U(_) | TypeExpr::I(_), TypeExpr::Field) => {
                            emitter.push_op(bytecode::OpCode::CastU64ToField {
                                res: result,
                                a: layouter.get_value(*v),
                            });
                        }
                        _ => panic!("Unsupported cast: {:?} -> {:?}", l_type, r_type),
                    }
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
                    ..
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
                    let res = layouter.alloc_value(*r, &type_info.get_value_type(*r));
                    emitter.push_op(bytecode::OpCode::TupleProj {
                        res,
                        tuple: layouter.get_value(*t),
                        index: *idx as u64,
                        child_sizes: type_info
                            .get_value_type(*t)
                            .get_tuple_elements()
                            .iter()
                            .map(|elem_type| layouter.type_size(elem_type))
                            .collect(),
                    });
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
                    result: r,
                    slice: sl,
                } => {
                    let res = layouter.alloc_value(*r, &type_info.get_value_type(*r));
                    let slice_type = type_info.get_value_type(*sl);
                    let elem_type = slice_type.get_array_element();
                    let stride = layouter.type_size(&elem_type);
                    emitter.push_op(bytecode::OpCode::SliceLen {
                        res,
                        array: layouter.get_value(*sl),
                        stride,
                    });
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
                    element_types,
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
                    let field_sizes: Vec<usize> = element_types
                        .iter()
                        .map(|elem_type| {
                            let size = layouter.type_size(elem_type);
                            assert!(
                                size <= 8,
                                "Struct field has {} bits, but maximum is 512",
                                size * 64
                            );
                            size
                        })
                        .collect();
                    let reference_counting = element_types
                        .iter()
                        .map(|elem_type| elem_type.is_heap_allocated())
                        .collect();
                    emitter.push_op(bytecode::OpCode::TupleAlloc {
                        res,
                        meta: vm::array::BoxedLayout::new_struct(field_sizes, reference_counting),
                        fields,
                    });
                }
                ssa::OpCode::Call {
                    results: r,
                    function: ssa::CallTarget::Static(fnid),
                    args: params,
                    unconstrained: _,
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
                        args,
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
                ssa::OpCode::AssertEq { lhs, rhs } => {
                    let lhs_type = type_info.get_value_type(*lhs);
                    match &lhs_type.expr {
                        TypeExpr::Field => {
                            emitter.push_op(bytecode::OpCode::AssertEqField {
                                a: layouter.get_value(*lhs),
                                b: layouter.get_value(*rhs),
                            });
                        }
                        TypeExpr::U(_) | TypeExpr::I(_) => {
                            emitter.push_op(bytecode::OpCode::AssertEqU64 {
                                a: layouter.get_value(*lhs),
                                b: layouter.get_value(*rhs),
                            });
                        }
                        t => panic!("Unsupported type for AssertEq in vm: {:?}", t),
                    }
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
                    assert!(*c <= 32, "ToRadix byte count must be <= 32");
                    emitter.push_op(bytecode::OpCode::ToBytesBe {
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
                    result_type: _,
                } => {
                    emitter.push_op(bytecode::OpCode::FreshWitness {
                        res: layouter.alloc_ptr(*r),
                    });
                }
                ssa::OpCode::MulConst {
                    result: r,
                    const_val: c,
                    var: v,
                } => {
                    // MulConst reads coeff as Field (4 u64s). Cast if needed.
                    let c_type = type_info.get_value_type(*c);
                    let coeff_pos = if c_type.expr != TypeExpr::Field {
                        let tmp = layouter.alloc_temp_field();
                        emitter.push_op(bytecode::OpCode::CastU64ToField {
                            res: tmp,
                            a: layouter.get_value(*c),
                        });
                        tmp
                    } else {
                        layouter.get_value(*c)
                    };
                    emitter.push_op(bytecode::OpCode::MulConst {
                        res: layouter.alloc_ptr(*r),
                        coeff: coeff_pos,
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
                    flag,
                } => {
                    assert!(keys.len() == 1);
                    assert!(results.len() == 0);
                    assert!(type_info.get_value_type(keys[0]).is_field());
                    emitter.push_op(bytecode::OpCode::Rngchk8Field {
                        val: layouter.get_value(keys[0]),
                        flag: layouter.get_value(*flag),
                    });
                }
                ssa::OpCode::Lookup {
                    target: LookupTarget::Array(arr),
                    keys,
                    results,
                    flag,
                } => {
                    assert!(keys.len() == 1);
                    assert!(results.len() == 1);
                    let arr_type = type_info.get_value_type(*arr);
                    let elem_type = arr_type.get_array_element();
                    let (stride, elem_kind) = lookup_elem_kind(&elem_type);
                    emitter.push_op(bytecode::OpCode::ArrayLookupField {
                        array: layouter.get_value(*arr),
                        index: layouter.get_value(keys[0]),
                        result: layouter.get_value(results[0]),
                        flag: layouter.get_value(*flag),
                        stride,
                        elem_kind,
                    });
                }
                ssa::OpCode::DLookup {
                    target: LookupTarget::Rangecheck(8),
                    keys,
                    results,
                    flag,
                } => {
                    assert!(keys.len() == 1);
                    assert!(results.len() == 0);
                    assert!(type_info.get_value_type(keys[0]).is_witness_of());
                    emitter.push_op(bytecode::OpCode::Drngchk8Field {
                        val: layouter.get_value(keys[0]),
                        flag: layouter.get_value(*flag),
                    });
                }
                ssa::OpCode::DLookup {
                    target: LookupTarget::Array(arr),
                    keys,
                    results,
                    flag,
                } => {
                    assert!(keys.len() == 1);
                    assert!(results.len() == 1);
                    let arr_type = type_info.get_value_type(*arr);
                    let elem_type = arr_type.get_array_element();
                    let (stride, elem_kind) = lookup_elem_kind(&elem_type);
                    emitter.push_op(bytecode::OpCode::DarrayLookupField {
                        array: layouter.get_value(*arr),
                        index: layouter.get_value(keys[0]),
                        result: layouter.get_value(results[0]),
                        flag: layouter.get_value(*flag),
                        stride,
                        elem_kind,
                    });
                }
                ssa::OpCode::Lookup {
                    target: LookupTarget::Spread(bits),
                    keys,
                    results,
                    flag,
                } => {
                    assert!(keys.len() == 1);
                    assert!(results.len() == 1);
                    emitter.push_op(bytecode::OpCode::SpreadLookupField {
                        val: layouter.get_value(keys[0]),
                        result: layouter.get_value(results[0]),
                        flag: layouter.get_value(*flag),
                        bits: *bits as usize,
                    });
                }
                ssa::OpCode::DLookup {
                    target: LookupTarget::Spread(bits),
                    keys,
                    results,
                    flag,
                } => {
                    assert!(keys.len() == 1);
                    assert!(results.len() == 1);
                    emitter.push_op(bytecode::OpCode::DspreadLookupField {
                        val: layouter.get_value(keys[0]),
                        result: layouter.get_value(results[0]),
                        flag: layouter.get_value(*flag),
                        bits: *bits as usize,
                    });
                }
                ssa::OpCode::Spread { result, value } => {
                    let value_type = type_info.get_value_type(*value);
                    let value_bits = match &value_type.expr {
                        TypeExpr::U(bits) | TypeExpr::I(bits) => bits,
                        _ => panic!("Unsupported spread input type in codegen: {value_type}"),
                    };
                    if *value_bits > 32 {
                        todo!("Spread bytecode lowering for integer widths > 32 bits");
                    }
                    let result_type = type_info.get_value_type(*result);
                    let res = match result_type.strip_witness().expr {
                        TypeExpr::U(bits) | TypeExpr::I(bits) => layouter.alloc_u64(*result, bits),
                        TypeExpr::Field => layouter.alloc_field(*result),
                        _ => panic!("Unsupported spread result type: {result_type}"),
                    };
                    emitter.push_op(bytecode::OpCode::SpreadU32 {
                        res,
                        val: layouter.get_value(*value),
                    });
                }
                ssa::OpCode::Unspread {
                    result_odd,
                    result_even,
                    value,
                } => {
                    let odd_type = type_info.get_value_type(*result_odd);
                    let even_type = type_info.get_value_type(*result_even);
                    let res_and = match odd_type.strip_witness().expr {
                        TypeExpr::U(bits) | TypeExpr::I(bits) => {
                            layouter.alloc_u64(*result_odd, bits)
                        }
                        TypeExpr::Field => layouter.alloc_field(*result_odd),
                        _ => panic!("Unsupported unspread odd result type: {odd_type}"),
                    };
                    let res_xor = match even_type.strip_witness().expr {
                        TypeExpr::U(bits) | TypeExpr::I(bits) => {
                            layouter.alloc_u64(*result_even, bits)
                        }
                        TypeExpr::Field => layouter.alloc_field(*result_even),
                        _ => panic!("Unsupported unspread even result type: {even_type}"),
                    };
                    emitter.push_op(bytecode::OpCode::UnspreadU64 {
                        res_and,
                        res_xor,
                        val: layouter.get_value(*value),
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
                ssa::OpCode::ReadGlobal {
                    result: r,
                    offset,
                    result_type: _,
                } => {
                    let global_idx = *offset as usize;
                    let res = layouter.alloc_value(*r, &type_info.get_value_type(*r));
                    emitter.push_op(bytecode::OpCode::ReadGlobal {
                        res,
                        global_offset: global_layouter.get_offset(global_idx),
                        size: global_layouter.get_size(global_idx),
                    });
                }
                ssa::OpCode::Const { result, value } => match value {
                    ssa::ConstValue::U(size, val) | ssa::ConstValue::I(size, val) => {
                        emitter.push_op(bytecode::OpCode::MovConst {
                            res: layouter.alloc_u64(*result, *size),
                            val: *val as u64,
                        });
                    }
                    ssa::ConstValue::Field(val) => {
                        let start = layouter.alloc_field(*result);
                        for i in 0..bytecode::LIMBS {
                            emitter.push_op(bytecode::OpCode::MovConst {
                                res: start.offset(i as isize),
                                val: val.0.0[i],
                            })
                        }
                    }
                    ssa::ConstValue::FnPtr(_) => {
                        panic!("FnPtr constants not supported in codegen");
                    }
                },
                ssa::OpCode::Alloc { result, elem_type } => {
                    let res = layouter.alloc_ptr(*result);
                    let elem_size = layouter.type_size(elem_type);
                    let elem_rc = elem_type.is_heap_allocated();
                    let meta = vm::array::BoxedLayout::ref_cell(elem_size, elem_rc);
                    emitter.push_op(bytecode::OpCode::RefAlloc { res, meta });
                }
                ssa::OpCode::Store { ptr, value } => {
                    let ptr_type = type_info.get_value_type(*ptr);
                    let elem_type = ptr_type.get_pointed();
                    let stride = layouter.type_size(&elem_type);
                    let elem_rc = if elem_type.is_heap_allocated() {
                        1usize
                    } else {
                        0usize
                    };
                    emitter.push_op(bytecode::OpCode::RefStore {
                        cell: layouter.get_value(*ptr),
                        source: layouter.get_value(*value),
                        stride,
                        elem_rc,
                    });
                }
                ssa::OpCode::Load { result, ptr } => {
                    let ptr_type = type_info.get_value_type(*ptr);
                    let elem_type = ptr_type.get_pointed();
                    let stride = layouter.type_size(&elem_type);
                    let res = layouter.alloc_value(*result, &elem_type);
                    emitter.push_op(bytecode::OpCode::RefLoad {
                        res,
                        cell: layouter.get_value(*ptr),
                        stride,
                    });
                }
                other => panic!("Unsupported instruction: {:?}", other),
            }
        }
        emitter.exit_block(block_id);
        match block.get_terminator().unwrap() {
            Terminator::Jmp(tgt, params) => {
                emitter.push_op(bytecode::OpCode::Nop {});
                // Back-edges (jumps to loop headers) need scratch copies to avoid
                // clobbering parameters
                let nop_count = if cfg.dominates(*tgt, block_id) {
                    2 * params.len()
                } else {
                    params.len()
                };
                for _ in 0..nop_count {
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
