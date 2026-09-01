//! The code generation path from HLSSA to Mavros VM bytecode.

pub mod layout;

use crate::{
    collections::HashMap,
    compiler::{
        analysis::{
            flow_analysis::{CFG, FlowAnalysis},
            types::{FunctionTypeInfo, TypeInfo},
        },
        codegen::{
            CodeGenOptions,
            bytecode::layout::{
                ConstantPoolInterner, FrameLayouter, GlobalFrameLayouter, StructLayoutInterner,
                for_each_constant_word,
            },
        },
        ssa::{
            BlockId, FunctionId, Instruction, SourceLocation, Terminator, ValueId,
            hlssa::{
                self, BinaryArithOpKind, CmpKind, DMatrix, Endianness, HLBlock, HLFunction, HLSSA,
                HLSSAConstantsSnapshot, LookupTarget, MAX_SUPPORTED_SIGNED_BITS, Radix, RefCountOp,
                Type, TypeExpr,
            },
        },
        util::ice_non_elided_tuple,
    },
    vm::{self, bytecode},
};

fn vm_source_location(location: &SourceLocation) -> bytecode::SourceLocation {
    bytecode::SourceLocation::new(
        location.file.to_string(),
        location.start.line,
        location.start.column,
    )
}

/// Materialize every constant `ValueId` referenced by `function` into the function's frame at
/// entry.
///
/// Multi-cell constants are interned into the program-global constant pool (`pool`) and loaded with
/// a single `MovConstPool`, so one copy is shared across every function that references them.
/// Single-cell scalars are spilled inline as a `MovConst`, which is already as compact as a pool
/// load.
fn materialize_constants(
    function: &HLFunction,
    constants: &HLSSAConstantsSnapshot,
    pool: &mut ConstantPoolInterner,
    layouter: &mut FrameLayouter,
    emitter: &mut EmitterState,
) {
    let mut referenced: crate::collections::HashSet<ValueId> =
        crate::collections::HashSet::default();
    for (_, block) in function.get_blocks() {
        for instr in block.get_instructions() {
            for vid in instr.get_inputs() {
                if constants.contains_key(vid) {
                    referenced.insert(*vid);
                }
            }
        }
        if let Some(term) = block.get_terminator() {
            match term {
                Terminator::Jmp(_, args) | Terminator::Return(args) => {
                    for vid in args {
                        if constants.contains_key(vid) {
                            referenced.insert(*vid);
                        }
                    }
                }
                Terminator::JmpIf(cond, _, _) => {
                    if constants.contains_key(cond) {
                        referenced.insert(*cond);
                    }
                }
            }
        }
    }

    // Sort for determinism: HashSet iteration order is non-deterministic but the emitted bytecode
    // must be stable across runs.
    let mut referenced: Vec<ValueId> = referenced.into_iter().collect();
    referenced.sort_by_key(|v| v.0);

    for vid in referenced {
        let constant = constants.get(&vid).expect("vid is in constants").as_ref();
        let cells = constant_cell_count(constant);
        let res = match constant {
            hlssa::Constant::Int(size, _) => layouter.alloc_int(vid, *size),
            hlssa::Constant::Field(_) => layouter.alloc_field(vid),
            hlssa::Constant::Blob(_) => layouter.alloc_long_data(vid, cells),
            hlssa::Constant::FnPtr(_) => panic!("FnPtr constants not supported in codegen"),
        };

        // Multi-cell constants (fields, u128s, blobs) are interned into the program-global constant
        // pool and loaded with a single `MovConstPool`, sharing one copy across every function that
        // references them. Single-cell scalars stay inline: a `MovConst` is already as small as a
        // pool load, so pooling them would only add indirection.
        if cells >= 2 {
            let pool_offset = pool.intern(vid, constant);
            emitter.push_op(bytecode::OpCode::MovConstPool {
                res,
                pool_offset,
                size: cells,
            });
        } else {
            spill_constant_to_frame(constant, res, emitter);
        }
    }
}

/// The number of `u64` cells `value` occupies once materialized — i.e. the number of words
/// [`for_each_constant_word`] emits.
///
/// This is both the frame-slot size and the `mov_const_pool` memcpy size, so it is derived from the
/// same visitor to keep them in lockstep.
fn constant_cell_count(value: &hlssa::Constant) -> usize {
    let mut count = 0usize;
    for_each_constant_word(value, &mut |_| count += 1);
    count
}

/// Spill `value` into `res` inline, one `MovConst` per word, in the layout defined by
/// [`for_each_constant_word`] (the same layout the constant pool uses).
fn spill_constant_to_frame(
    value: &hlssa::Constant,
    res: bytecode::FramePosition,
    emitter: &mut EmitterState,
) {
    let mut offset = 0isize;
    for_each_constant_word(value, &mut |word| {
        emitter.push_op(bytecode::OpCode::MovConst {
            res: res.offset(offset),
            val: word,
        });
        offset += 1;
    });
}

// CODE GENERATOR
// ================================================================================================

/// The code generator that lowers HLSSA to Mavros bytecode.
pub struct CodeGen {
    options: CodeGenOptions,
}

impl CodeGen {
    pub fn new(options: CodeGenOptions) -> Self {
        Self { options }
    }

    pub fn run(&self, ssa: &HLSSA, cfg: &FlowAnalysis, type_info: &TypeInfo) -> bytecode::Program {
        let global_layouter = GlobalFrameLayouter::new(ssa);
        let struct_interner = StructLayoutInterner::new();
        let mut const_pool = ConstantPoolInterner::new();
        let constants = ssa.const_snapshot();

        // Entry points are emitted first, in entry-table order; the remaining functions follow.
        let entry_ids: Vec<FunctionId> = ssa.get_entry_points().to_vec();
        let function_order: Vec<FunctionId> = entry_ids
            .iter()
            .copied()
            .chain(ssa.get_function_ids().filter(|id| !entry_ids.contains(id)))
            .collect();

        let mut functions = Vec::new();
        let mut function_ids = HashMap::default();
        let mut cur_fn_begin = 0;

        for function_id in function_order {
            let function = self.run_function(
                ssa.get_function(function_id),
                cfg.get_function_cfg(function_id),
                type_info.get_function(function_id),
                &global_layouter,
                &constants,
                &mut const_pool,
            );
            function_ids.insert(function_id, cur_fn_begin);
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

        let witgen_entry = *entry_ids
            .get(bytecode::ENTRY_WITGEN)
            .expect("SSA has no witgen entry point");
        let entry_blob_field_count =
            match ssa.get_function(witgen_entry).get_param_types().as_slice() {
                [
                    Type {
                        expr: TypeExpr::Blob(_, len),
                    },
                ] => *len,
                // A program with no inputs and no declared return gets a zero-length blob.
                // Since nothing reads it, the blob gets DCE-d away.
                [] => 0,
                params => panic!(
                    "ICE: witgen entry must take a single Blob<Field; N> parameter, got {params:?}"
                ),
            };

        bytecode::Program {
            functions,
            entry_points: (0..entry_ids.len()).collect(),
            entry_blob_field_count,
            global_frame_size: global_layouter.total_size,
            struct_layouts: struct_interner.into_table(),
            constant_pool: const_pool.into_pool(),
        }
    }

    fn run_function(
        &self,
        function: &HLFunction,
        cfg: &CFG,
        type_info: &FunctionTypeInfo,
        global_layouter: &GlobalFrameLayouter,
        constants: &HLSSAConstantsSnapshot,
        pool: &mut ConstantPoolInterner,
    ) -> bytecode::Function {
        let mut layouter = FrameLayouter::new();
        let entry = function.get_entry();
        let fallback_location = function
            .get_entry()
            .first_location()
            .cloned()
            .unwrap_or_else(|| SourceLocation::synthetic(function.get_name()));
        let mut emitter = EmitterState::new(vm_source_location(&fallback_location));

        // Entry block params need to be allocated at the beginning of the frame (after return
        // address and return data pointer)
        for (param, tp) in entry.get_parameters() {
            layouter.alloc_value(*param, tp);
        }

        materialize_constants(function, constants, pool, &mut layouter, &mut emitter);

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
            source_locations: emitter.source_locations,
        }
    }

    fn run_block_body(
        &self,
        function: &HLFunction,
        block_id: BlockId,
        block: &HLBlock,
        type_info: &FunctionTypeInfo,
        cfg: &CFG,
        layouter: &mut FrameLayouter,
        emitter: &mut EmitterState,
        global_layouter: &GlobalFrameLayouter,
    ) {
        let block_location = block
            .first_location()
            .or_else(|| function.get_entry().first_location())
            .cloned()
            .unwrap_or_else(|| SourceLocation::synthetic(function.get_name()));
        emitter.set_source_location(vm_source_location(&block_location));
        emitter.enter_block(block_id);

        // Two arm shapes appear below.
        //
        // An operation whose two signed forms emit the **same** opcode lists both variants in the
        // pattern — `UAdd | SAdd`, `UMul | SMul`, `UShl | SShl` — so that adding a
        // `BinaryArithOpKind` is a compile error here rather than a silent fall-through to the
        // catch-all panic.
        //
        // One whose forms emit **different** opcodes binds the kind instead
        // (`kind @ (UDiv | SDiv)`) and matches on `kind.is_signed()` alongside the operand type,
        // because then the opcode choice is a joint fact about both the reading and the width.
        for (instruction, source_location) in block.get_instructions_with_source_locations() {
            emitter.set_source_location(vm_source_location(source_location));

            match instruction {
                hlssa::OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::UAdd | BinaryArithOpKind::SAdd,
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
                    TypeExpr::Int(bits) if *bits <= 64 => {
                        let result = layouter.alloc_int(*val, *bits);
                        emitter.push_op(bytecode::OpCode::AddInt {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                            bits: *bits as u64,
                        });
                    }
                    TypeExpr::Int(128) => {
                        let result = layouter.alloc_int(*val, 128);
                        emitter.push_op(bytecode::OpCode::AddInt128 {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
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
                hlssa::OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::USub | BinaryArithOpKind::SSub,
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
                    TypeExpr::Int(bits) if *bits <= 64 => {
                        let result = layouter.alloc_int(*val, *bits);
                        emitter.push_op(bytecode::OpCode::SubInt {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                            bits: *bits as u64,
                        });
                    }
                    TypeExpr::Int(128) => {
                        let result = layouter.alloc_int(*val, 128);
                        emitter.push_op(bytecode::OpCode::SubInt128 {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    t => panic!("Unsupported type for subtraction: {:?}", t),
                },
                hlssa::OpCode::BinaryArithOp {
                    kind: kind @ (BinaryArithOpKind::UDiv | BinaryArithOpKind::SDiv),
                    result: val,
                    lhs: op1,
                    rhs: op2,
                } => match (kind.is_signed(), &type_info.get_value_type(*val).expr) {
                    (false, TypeExpr::Field) => {
                        let result = layouter.alloc_field(*val);
                        emitter.push_op(bytecode::OpCode::DivField {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    (false, TypeExpr::Int(bits)) if *bits <= 64 => {
                        let result = layouter.alloc_int(*val, *bits);
                        emitter.push_op(bytecode::OpCode::UdivInt {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    (true, TypeExpr::Int(bits)) if *bits <= MAX_SUPPORTED_SIGNED_BITS => {
                        let result = layouter.alloc_int(*val, *bits);
                        emitter.push_op(bytecode::OpCode::SdivInt {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                            bits: *bits as u64,
                        });
                    }
                    (false, TypeExpr::Int(128)) => {
                        let result = layouter.alloc_int(*val, 128);
                        emitter.push_op(bytecode::OpCode::UdivInt128 {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    (signed, t) => panic!(
                        "Unsupported type for {} division: {:?}",
                        if signed { "signed" } else { "unsigned" },
                        t
                    ),
                },
                hlssa::OpCode::BinaryArithOp {
                    kind: kind @ (BinaryArithOpKind::URem | BinaryArithOpKind::SRem),
                    result: val,
                    lhs: op1,
                    rhs: op2,
                } => match (kind.is_signed(), &type_info.get_value_type(*val).expr) {
                    (_, TypeExpr::Field) => {
                        panic!("Modulo is not defined on field elements")
                    }
                    (false, TypeExpr::Int(bits)) if *bits <= 64 => {
                        let result = layouter.alloc_int(*val, *bits);
                        emitter.push_op(bytecode::OpCode::UremInt {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    (true, TypeExpr::Int(bits)) if *bits <= MAX_SUPPORTED_SIGNED_BITS => {
                        let result = layouter.alloc_int(*val, *bits);
                        emitter.push_op(bytecode::OpCode::SremInt {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                            bits: *bits as u64,
                        });
                    }
                    (false, TypeExpr::Int(128)) => {
                        let result = layouter.alloc_int(*val, 128);
                        emitter.push_op(bytecode::OpCode::UremInt128 {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    (signed, t) => panic!(
                        "Unsupported type for {} modulo: {:?}",
                        if signed { "signed" } else { "unsigned" },
                        t
                    ),
                },
                hlssa::OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::UMul | BinaryArithOpKind::SMul,
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
                    TypeExpr::Int(bits) if *bits <= 64 => {
                        let result = layouter.alloc_int(*val, *bits);
                        emitter.push_op(bytecode::OpCode::MulInt {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                            bits: *bits as u64,
                        });
                    }
                    TypeExpr::Int(128) => {
                        let result = layouter.alloc_int(*val, 128);
                        emitter.push_op(bytecode::OpCode::MulInt128 {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    t => panic!("Unsupported type for multiplication: {:?}", t),
                },
                hlssa::OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::And,
                    result: val,
                    lhs: op1,
                    rhs: op2,
                } => match &type_info.get_value_type(*val).expr {
                    TypeExpr::Field => {
                        panic!("Unsupported: field and");
                    }
                    TypeExpr::Int(bits) if *bits <= 64 => {
                        let result = layouter.alloc_int(*val, *bits);
                        emitter.push_op(bytecode::OpCode::AndInt {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    TypeExpr::Int(128) => {
                        let result = layouter.alloc_int(*val, 128);
                        emitter.push_op(bytecode::OpCode::AndInt128 {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    t => panic!("Unsupported type for bitwise and: {:?}", t),
                },
                hlssa::OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::Or,
                    result: val,
                    lhs: op1,
                    rhs: op2,
                } => match &type_info.get_value_type(*val).expr {
                    TypeExpr::Int(bits) if *bits <= 64 => {
                        let result = layouter.alloc_int(*val, *bits);
                        emitter.push_op(bytecode::OpCode::OrInt {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    TypeExpr::Int(128) => {
                        let result = layouter.alloc_int(*val, 128);
                        emitter.push_op(bytecode::OpCode::OrInt128 {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    t => panic!("Unsupported type for bitwise or: {:?}", t),
                },
                hlssa::OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::Xor,
                    result: val,
                    lhs: op1,
                    rhs: op2,
                } => match &type_info.get_value_type(*val).expr {
                    TypeExpr::Int(bits) if *bits <= 64 => {
                        let result = layouter.alloc_int(*val, *bits);
                        emitter.push_op(bytecode::OpCode::XorInt {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    TypeExpr::Int(128) => {
                        let result = layouter.alloc_int(*val, 128);
                        emitter.push_op(bytecode::OpCode::XorInt128 {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    t => panic!("Unsupported type for bitwise xor: {:?}", t),
                },
                hlssa::OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::UShl | BinaryArithOpKind::SShl,
                    result: val,
                    lhs: op1,
                    rhs: op2,
                } => match &type_info.get_value_type(*val).expr {
                    TypeExpr::Int(bits) if *bits <= 64 => {
                        let result = layouter.alloc_int(*val, *bits);
                        emitter.push_op(bytecode::OpCode::ShlInt {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                            bits: *bits as u64,
                        });
                    }
                    TypeExpr::Int(128) => {
                        let result = layouter.alloc_int(*val, 128);
                        emitter.push_op(bytecode::OpCode::ShlInt128 {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    t => panic!("Unsupported type for shift left: {:?}", t),
                },
                hlssa::OpCode::BinaryArithOp {
                    kind: kind @ (BinaryArithOpKind::UShr | BinaryArithOpKind::SShr),
                    result: val,
                    lhs: op1,
                    rhs: op2,
                } => match (kind.is_signed(), &type_info.get_value_type(*val).expr) {
                    (false, TypeExpr::Int(bits)) if *bits <= 64 => {
                        let result = layouter.alloc_int(*val, *bits);
                        // Zero-fill. Needs `bits` only to mask the shift amount to `bits - 1`, the
                        // way LLVM does; the result of a logical shift cannot exceed the width, so
                        // nothing is re-masked.
                        emitter.push_op(bytecode::OpCode::UshrInt {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                            bits: *bits as u64,
                        });
                    }

                    // Sign-fill, matching Noir and `IntArithOp::AShr` on the LLVM side. Needs
                    // `bits` both to mask the amount and because the sign lives at `bits - 1`,
                    // not at 63.
                    (true, TypeExpr::Int(bits)) if *bits <= MAX_SUPPORTED_SIGNED_BITS => {
                        let result = layouter.alloc_int(*val, *bits);
                        emitter.push_op(bytecode::OpCode::AshrInt {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                            bits: *bits as u64,
                        });
                    }
                    (false, TypeExpr::Int(128)) => {
                        let result = layouter.alloc_int(*val, 128);
                        emitter.push_op(bytecode::OpCode::UshrInt128 {
                            res: result,
                            a: layouter.get_value(*op1),
                            b: layouter.get_value(*op2),
                        });
                    }
                    (signed, t) => panic!(
                        "Unsupported type for {} shift right: {:?}",
                        if signed { "signed" } else { "unsigned" },
                        t
                    ),
                },
                hlssa::OpCode::Cmp {
                    kind: kind @ (CmpKind::ULt | CmpKind::SLt),
                    result: val,
                    lhs: op1,
                    rhs: op2,
                } => {
                    let result_bits = match &type_info.get_value_type(*val).expr {
                        TypeExpr::Int(bits) => *bits,
                        t => panic!("Unsupported result type for comparison: {:?}", t),
                    };
                    let result = layouter.alloc_int(*val, result_bits);
                    let lhs_type = type_info.get_value_type(*op1);
                    let rhs_type = type_info.get_value_type(*op2);
                    match (kind.is_signed(), &lhs_type.expr, &rhs_type.expr) {
                        (true, TypeExpr::Int(lhs_bits), TypeExpr::Int(rhs_bits))
                            if *lhs_bits == *rhs_bits && *lhs_bits <= MAX_SUPPORTED_SIGNED_BITS =>
                        {
                            emitter.push_op(bytecode::OpCode::SltInt {
                                res: result,
                                a: layouter.get_value(*op1),
                                b: layouter.get_value(*op2),
                                bits: *lhs_bits as u64,
                            });
                        }
                        (false, TypeExpr::Int(lhs_bits), TypeExpr::Int(rhs_bits))
                            if *lhs_bits == *rhs_bits && *lhs_bits <= 64 =>
                        {
                            emitter.push_op(bytecode::OpCode::UltInt {
                                res: result,
                                a: layouter.get_value(*op1),
                                b: layouter.get_value(*op2),
                            });
                        }
                        (false, TypeExpr::Int(128), TypeExpr::Int(128)) => {
                            emitter.push_op(bytecode::OpCode::UltInt128 {
                                res: result,
                                a: layouter.get_value(*op1),
                                b: layouter.get_value(*op2),
                            });
                        }
                        (false, TypeExpr::Field, TypeExpr::Field) => {
                            emitter.push_op(bytecode::OpCode::LtField {
                                res: result,
                                a: layouter.get_value(*op1),
                                b: layouter.get_value(*op2),
                            })
                        }
                        _ => panic!(
                            "unsupported args for `{}`: {} {}",
                            kind.symbol(),
                            lhs_type,
                            rhs_type
                        ),
                    }
                }
                hlssa::OpCode::Cmp {
                    kind: CmpKind::Eq,
                    result: val,
                    lhs: op1,
                    rhs: op2,
                } => {
                    let result_bits = match &type_info.get_value_type(*val).expr {
                        TypeExpr::Int(bits) => *bits,
                        t => panic!("Unsupported result type for comparison: {:?}", t),
                    };
                    let result = layouter.alloc_int(*val, result_bits);
                    let lhs_type = type_info.get_value_type(*op1);
                    let rhs_type = type_info.get_value_type(*op2);
                    match (&lhs_type.expr, &rhs_type.expr) {
                        // Equality is a comparison of raw patterns, so one arm serves both readings
                        (TypeExpr::Int(lhs_bits), TypeExpr::Int(rhs_bits))
                            if *lhs_bits == *rhs_bits && *lhs_bits <= 64 =>
                        {
                            emitter.push_op(bytecode::OpCode::EqInt {
                                res: result,
                                a: layouter.get_value(*op1),
                                b: layouter.get_value(*op2),
                            });
                        }
                        (TypeExpr::Int(128), TypeExpr::Int(128)) => {
                            emitter.push_op(bytecode::OpCode::EqInt128 {
                                res: result,
                                a: layouter.get_value(*op1),
                                b: layouter.get_value(*op2),
                            });
                        }
                        (TypeExpr::Field, TypeExpr::Field) => {
                            emitter.push_op(bytecode::OpCode::EqField {
                                res: result,
                                a: layouter.get_value(*op1),
                                b: layouter.get_value(*op2),
                            });
                        }
                        _ => panic!("unsupported args {} {}", lhs_type, rhs_type),
                    }
                }
                hlssa::OpCode::Cast {
                    result: r,
                    value: v,
                    target: tgt,
                } => {
                    let l_type = type_info.get_value_type(*v);
                    let r_type = type_info.get_value_type(*r);
                    if matches!(tgt, hlssa::CastTarget::Map(_) | hlssa::CastTarget::ValueOf) {
                        panic!(
                            "ICE: {} cast should have been lowered before bytecode codegen",
                            tgt
                        );
                    }
                    if matches!(tgt, hlssa::CastTarget::WitnessOf) {
                        // PureToWitnessRef reads a Field (4 u64s) from the frame.
                        // If the source is not Field-sized, cast to Field first.
                        let field_pos = if l_type.expr != TypeExpr::Field {
                            let tmp = layouter.alloc_temp_field();
                            match &l_type.expr {
                                TypeExpr::Int(bits) if *bits <= 64 => {
                                    emitter.push_op(bytecode::OpCode::CastIntToField {
                                        res: tmp,
                                        a: layouter.get_value(*v),
                                    });
                                }
                                TypeExpr::Int(128) => {
                                    emitter.push_op(bytecode::OpCode::CastInt128ToField {
                                        res: tmp,
                                        a: layouter.get_value(*v),
                                    });
                                }
                                t => panic!("Unsupported witness cast source: {:?}", t),
                            }
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
                    let is_nop = matches!(
                        tgt,
                        hlssa::CastTarget::Nop | hlssa::CastTarget::ArrayToSlice
                    ) || l_type.expr == r_type.expr
                        || (l_type.is_witness_of() && r_type.is_witness_of());
                    if is_nop {
                        let pos = layouter.variables[v];
                        layouter.variables.insert(*r, pos);
                        continue;
                    }
                    let result = layouter.alloc_value(*r, &r_type);
                    match (&l_type.expr, &r_type.expr) {
                        (TypeExpr::Int(source_bits), TypeExpr::Int(target_bits))
                            if *source_bits <= 64 && *target_bits <= 64 =>
                        {
                            let source_cells = source_bits.div_ceil(64);
                            let target_cells = target_bits.div_ceil(64);
                            let copied_cells = source_cells.min(target_cells);
                            emitter.push_op(bytecode::OpCode::MovFrame {
                                target: result,
                                source: layouter.get_value(*v),
                                size: copied_cells,
                            });
                            if target_cells > source_cells {
                                for cell in source_cells..target_cells {
                                    emitter.push_op(bytecode::OpCode::MovConst {
                                        res: result.offset(cell as isize),
                                        val: 0,
                                    });
                                }
                            }
                            if target_bits < source_bits {
                                emitter.push_op(bytecode::OpCode::TruncateInt {
                                    res: result,
                                    a: result,
                                    to_bits: *target_bits as u64,
                                });
                            }
                        }
                        (TypeExpr::Int(128), TypeExpr::Int(target_bits)) if *target_bits <= 64 => {
                            emitter.push_op(bytecode::OpCode::MovFrame {
                                target: result,
                                source: layouter.get_value(*v),
                                size: 1,
                            });
                            if *target_bits < 64 {
                                emitter.push_op(bytecode::OpCode::TruncateInt {
                                    res: result,
                                    a: result,
                                    to_bits: *target_bits as u64,
                                });
                            }
                        }
                        (TypeExpr::Int(source_bits), TypeExpr::Int(128)) if *source_bits <= 64 => {
                            emitter.push_op(bytecode::OpCode::MovFrame {
                                target: result,
                                source: layouter.get_value(*v),
                                size: 1,
                            });
                            emitter.push_op(bytecode::OpCode::MovConst {
                                res: result.offset(1),
                                val: 0,
                            });
                        }
                        (TypeExpr::Field, TypeExpr::Int(bits)) if *bits <= 64 => {
                            emitter.push_op(bytecode::OpCode::CastFieldToInt {
                                res: result,
                                a: layouter.get_value(*v),
                            });
                            if *bits < 64 {
                                emitter.push_op(bytecode::OpCode::TruncateInt {
                                    res: result,
                                    a: result,
                                    to_bits: *bits as u64,
                                });
                            }
                        }
                        (TypeExpr::Field, TypeExpr::Int(128)) => {
                            emitter.push_op(bytecode::OpCode::CastFieldToInt128 {
                                res: result,
                                a: layouter.get_value(*v),
                            });
                        }
                        (TypeExpr::Int(bits), TypeExpr::Field) if *bits <= 64 => {
                            emitter.push_op(bytecode::OpCode::CastIntToField {
                                res: result,
                                a: layouter.get_value(*v),
                            });
                        }
                        (TypeExpr::Int(128), TypeExpr::Field) => {
                            emitter.push_op(bytecode::OpCode::CastInt128ToField {
                                res: result,
                                a: layouter.get_value(*v),
                            });
                        }
                        _ => panic!("unsupported args {} {}", l_type, r_type),
                    }
                }

                // Unreachable on any program that gets here: `LowerWitnessBitwiseOps::lower_not`
                // rewrites every `Not`, pure or witness, into `(2^bits - 1) - value` during
                // `spill_witness`. Kept because lowering a pure `Not` straight to `NotInt` is the
                // obvious way to make it one instruction instead of a field subtraction.
                hlssa::OpCode::Not {
                    result: r,
                    value: v,
                } => {
                    let result_type = type_info.get_value_type(*r);
                    match &result_type.expr {
                        TypeExpr::Int(bits) if *bits <= 64 => {
                            let result = layouter.alloc_value(*r, result_type);
                            emitter.push_op(bytecode::OpCode::NotInt {
                                res: result,
                                a: layouter.get_value(*v),
                                bits: *bits as u64,
                            });
                        }
                        TypeExpr::Int(128) => {
                            let result = layouter.alloc_value(*r, result_type);
                            emitter.push_op(bytecode::OpCode::NotInt128 {
                                res: result,
                                a: layouter.get_value(*v),
                            });
                        }
                        t => panic!("Unsupported type for not: {:?}", t),
                    }
                }
                hlssa::OpCode::Constrain { a, b, c } => {
                    let a_type = type_info.get_value_type(*a);
                    let b_type = type_info.get_value_type(*b);
                    let c_type = type_info.get_value_type(*c);
                    if !a_type.is_field() || !b_type.is_field() || !c_type.is_field() {
                        panic!(
                            "Unsupported type for constrain: {:?}, {:?}, {:?}",
                            a_type, b_type, c_type
                        );
                    }
                    if self.options.check_constraints {
                        emit_assert_r1c(layouter, emitter, *a, *b, *c);
                    }
                    emitter.push_op(bytecode::OpCode::R1C {
                        a: layouter.get_value(*a),
                        b: layouter.get_value(*b),
                        c: layouter.get_value(*c),
                    });
                }
                hlssa::OpCode::WriteWitness {
                    result: None,
                    value: v,
                    ..
                } => {
                    emitter.push_op(bytecode::OpCode::WriteWitness {
                        val: layouter.get_value(*v),
                    });
                }
                hlssa::OpCode::ArrayGet {
                    result: r,
                    array: arr,
                    index: idx,
                } => {
                    let res = layouter.alloc_value(*r, &type_info.get_value_type(*r));
                    let arr_type = type_info.get_value_type(*arr);
                    match &arr_type.expr {
                        // Blobs live inline in the frame, not behind a boxed
                        // pointer, so element reads are frame-relative.
                        TypeExpr::Blob(elem, len) => {
                            emitter.push_op(bytecode::OpCode::BlobGet {
                                res,
                                source: layouter.get_value(*arr),
                                index: layouter.get_value(*idx),
                                stride: layouter.type_size(elem),
                                len: *len,
                            });
                        }
                        _ => {
                            emitter.push_op(bytecode::OpCode::ArrayGet {
                                res,
                                array: layouter.get_value(*arr),
                                index: layouter.get_value(*idx),
                                stride: layouter.type_size(&arr_type.get_array_element()),
                            });
                        }
                    }
                }
                hlssa::OpCode::TupleProj { .. } | hlssa::OpCode::TupleRefProj { .. } => {
                    ice_non_elided_tuple()
                }
                hlssa::OpCode::ArraySet {
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
                hlssa::OpCode::SlicePush {
                    result: r,
                    slice: sl,
                    values: vals,
                    dir,
                } => {
                    let res = layouter.alloc_value(*r, &type_info.get_value_type(*r));
                    let slice_type = type_info.get_value_type(*sl);
                    let elem_type = slice_type.get_array_element();
                    let stride = layouter.type_size(&elem_type);
                    let value_positions = vals
                        .iter()
                        .map(|v| layouter.get_value(*v))
                        .collect::<Vec<_>>();
                    let is_push_front = matches!(dir, hlssa::SliceOpDir::Front) as usize;
                    emitter.push_op(bytecode::OpCode::SlicePush {
                        res,
                        slice: layouter.get_value(*sl),
                        stride,
                        is_push_front,
                        values: value_positions,
                    });
                }
                hlssa::OpCode::SliceLen {
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
                hlssa::OpCode::MkSeq {
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
                hlssa::OpCode::MkSeqOfBlob {
                    result: r,
                    element_type: eltype,
                    blob,
                } => {
                    assert!(
                        !eltype.is_heap_allocated(),
                        "MkSeqOfBlob only supports scalar element types"
                    );
                    let res = layouter.alloc_value(*r, &type_info.get_value_type(*r));
                    let stride = layouter.type_size(eltype);
                    let len = match &type_info.get_value_type(*r).expr {
                        TypeExpr::Array(_, len) => *len,
                        other => panic!("MkSeqOfBlob result must be an array, got {:?}", other),
                    };
                    let blob_start = layouter.get_value(*blob);
                    emitter.push_op(bytecode::OpCode::ArrayAllocFromFrame {
                        res,
                        stride,
                        meta: vm::array::BoxedLayout::array(len * stride, false),
                        count: len,
                        source: blob_start,
                    });
                }
                hlssa::OpCode::MkRepeated {
                    result: r,
                    element,
                    seq_type: _,
                    count,
                    elem_type: eltype,
                } => {
                    let res = layouter.alloc_value(*r, &type_info.get_value_type(*r));
                    let item = layouter.get_value(*element);
                    let is_ptr = eltype.is_heap_allocated();
                    let stride = layouter.type_size(eltype);
                    emitter.push_op(bytecode::OpCode::ArrayAllocRepeated {
                        res,
                        stride,
                        meta: vm::array::BoxedLayout::array(*count * stride, is_ptr),
                        count: *count,
                        item,
                    });
                }
                hlssa::OpCode::MkTuple { .. } => ice_non_elided_tuple(),
                hlssa::OpCode::Call {
                    results: r,
                    function: hlssa::CallTarget::Static(fnid),
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
                hlssa::OpCode::Call {
                    function: hlssa::CallTarget::Dynamic(_),
                    ..
                } => {
                    panic!("Dynamic call targets are not supported in codegen")
                }
                hlssa::OpCode::MemOp {
                    kind: RefCountOp::Drop,
                    value: r,
                } => {
                    emitter.push_op(bytecode::OpCode::DecRc {
                        array: layouter.get_value(*r),
                    });
                }
                hlssa::OpCode::MemOp {
                    kind: RefCountOp::Bump(size),
                    value: r,
                } => {
                    emitter.push_op(bytecode::OpCode::IncRc {
                        array: layouter.get_value(*r),
                        amount: *size as u64,
                    });
                }
                hlssa::OpCode::AssertCmp { kind, lhs, rhs } => match kind {
                    hlssa::CmpKind::Eq => {
                        let lhs_type = type_info.get_value_type(*lhs);
                        let rhs_type = type_info.get_value_type(*rhs);
                        match (&lhs_type.expr, &rhs_type.expr) {
                            (TypeExpr::Field, TypeExpr::Field) => {
                                emitter.push_op(bytecode::OpCode::AssertEqField {
                                    a: layouter.get_value(*lhs),
                                    b: layouter.get_value(*rhs),
                                });
                            }
                            // As for `Cmp`/`Eq` above: one arm, raw patterns.
                            (TypeExpr::Int(lhs_bits), TypeExpr::Int(rhs_bits))
                                if *lhs_bits == *rhs_bits && *lhs_bits <= 64 =>
                            {
                                emitter.push_op(bytecode::OpCode::AssertEqInt {
                                    a: layouter.get_value(*lhs),
                                    b: layouter.get_value(*rhs),
                                });
                            }
                            (TypeExpr::Int(128), TypeExpr::Int(128)) => {
                                emitter.push_op(bytecode::OpCode::AssertEqInt128 {
                                    a: layouter.get_value(*lhs),
                                    b: layouter.get_value(*rhs),
                                });
                            }
                            _ => panic!("unsupported args {} {}", lhs_type, rhs_type),
                        }
                    }
                    kind @ (hlssa::CmpKind::ULt | hlssa::CmpKind::SLt) => {
                        let lhs_type = type_info.get_value_type(*lhs);
                        let rhs_type = type_info.get_value_type(*rhs);
                        let cmp_result = layouter.alloc_scratch(1);
                        match (kind.is_signed(), &lhs_type.expr, &rhs_type.expr) {
                            (true, TypeExpr::Int(lhs_bits), TypeExpr::Int(rhs_bits))
                                if *lhs_bits == *rhs_bits
                                    && *lhs_bits <= MAX_SUPPORTED_SIGNED_BITS =>
                            {
                                emitter.push_op(bytecode::OpCode::SltInt {
                                    res: cmp_result,
                                    a: layouter.get_value(*lhs),
                                    b: layouter.get_value(*rhs),
                                    bits: *lhs_bits as u64,
                                });
                            }
                            (false, TypeExpr::Int(lhs_bits), TypeExpr::Int(rhs_bits))
                                if *lhs_bits == *rhs_bits && *lhs_bits <= 64 =>
                            {
                                emitter.push_op(bytecode::OpCode::UltInt {
                                    res: cmp_result,
                                    a: layouter.get_value(*lhs),
                                    b: layouter.get_value(*rhs),
                                });
                            }
                            (false, TypeExpr::Int(128), TypeExpr::Int(128)) => {
                                emitter.push_op(bytecode::OpCode::UltInt128 {
                                    res: cmp_result,
                                    a: layouter.get_value(*lhs),
                                    b: layouter.get_value(*rhs),
                                });
                            }
                            (false, TypeExpr::Field, TypeExpr::Field) => {
                                emitter.push_op(bytecode::OpCode::LtField {
                                    res: cmp_result,
                                    a: layouter.get_value(*lhs),
                                    b: layouter.get_value(*rhs),
                                });
                            }
                            _ => panic!(
                                "unsupported args for `{}`: {} {}",
                                kind.symbol(),
                                lhs_type,
                                rhs_type
                            ),
                        }
                        let one = layouter.alloc_scratch(1);
                        emitter.push_op(bytecode::OpCode::MovConst { res: one, val: 1 });
                        emitter.push_op(bytecode::OpCode::AssertEqInt {
                            a: cmp_result,
                            b: one,
                        });
                    }
                },
                hlssa::OpCode::Assert { value } => {
                    let one = layouter.alloc_scratch(1);
                    emitter.push_op(bytecode::OpCode::MovConst { res: one, val: 1 });
                    emitter.push_op(bytecode::OpCode::AssertEqInt {
                        a: layouter.get_value(*value),
                        b: one,
                    });
                }
                hlssa::OpCode::AssertR1C { a, b, c } => {
                    emit_assert_r1c(layouter, emitter, *a, *b, *c);
                }
                hlssa::OpCode::ToBits {
                    result: r,
                    value,
                    endianness,
                    count,
                } => {
                    let res = layouter.alloc_value(*r, &type_info.get_value_type(*r));
                    let val = layouter.get_value(*value);
                    let count = *count as u64;
                    let op = match endianness {
                        Endianness::Big => bytecode::OpCode::ToBitsBe { res, val, count },
                        Endianness::Little => bytecode::OpCode::ToBitsLe { res, val, count },
                    };
                    emitter.push_op(op);
                }
                hlssa::OpCode::ToRadix {
                    result: r,
                    value: v,
                    radix: Radix::Bytes,
                    endianness,
                    count: c,
                } => {
                    assert!(
                        type_info.get_value_type(*v).is_field(),
                        "TODO: Implement toRadix for U-values"
                    );
                    assert!(*c <= 32, "ToRadix byte count must be <= 32");
                    let res = layouter.alloc_value(*r, &type_info.get_value_type(*r));
                    match endianness {
                        Endianness::Big => emitter.push_op(bytecode::OpCode::ToBytesBe {
                            val: layouter.get_value(*v),
                            count: *c as u64,
                            res,
                        }),
                        Endianness::Little => emitter.push_op(bytecode::OpCode::ToBytesLe {
                            val: layouter.get_value(*v),
                            count: *c as u64,
                            res,
                        }),
                    }
                }
                hlssa::OpCode::ToRadix {
                    result: _,
                    value: v,
                    radix,
                    endianness,
                    count,
                } => {
                    panic!(
                        "ToRadix not yet implemented: radix={:?} endianness={:?} count={} value_type={:?}",
                        radix,
                        endianness,
                        count,
                        type_info.get_value_type(*v),
                    );
                }
                hlssa::OpCode::NextDCoeff { result: out } => {
                    let v = layouter.alloc_field(*out);
                    emitter.push_op(bytecode::OpCode::NextDCoeff { v });
                }
                hlssa::OpCode::BumpD {
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
                hlssa::OpCode::FreshWitness {
                    result: r,
                    result_type: _,
                } => {
                    emitter.push_op(bytecode::OpCode::FreshWitness {
                        res: layouter.alloc_ptr(*r),
                    });
                }
                hlssa::OpCode::MulConst {
                    result: r,
                    const_val: c,
                    var: v,
                } => {
                    // MulConst reads coeff as Field (4 u64s). Cast if needed.
                    let c_type = type_info.get_value_type(*c);
                    let coeff_pos = match &c_type.expr {
                        TypeExpr::Field => layouter.get_value(*c),
                        TypeExpr::Int(bits) if *bits <= 64 => {
                            let tmp = layouter.alloc_temp_field();
                            emitter.push_op(bytecode::OpCode::CastIntToField {
                                res: tmp,
                                a: layouter.get_value(*c),
                            });
                            tmp
                        }
                        TypeExpr::Int(128) => {
                            let tmp = layouter.alloc_temp_field();
                            emitter.push_op(bytecode::OpCode::CastInt128ToField {
                                res: tmp,
                                a: layouter.get_value(*c),
                            });
                            tmp
                        }
                        t => panic!("Unsupported MulConst coefficient type: {:?}", t),
                    };
                    emitter.push_op(bytecode::OpCode::MulConst {
                        res: layouter.alloc_ptr(*r),
                        coeff: coeff_pos,
                        v: layouter.get_value(*v),
                    });
                }
                hlssa::OpCode::Rangecheck {
                    value: val,
                    max_bits,
                } => {
                    emitter.push_op(bytecode::OpCode::Rangecheck {
                        val: layouter.get_value(*val),
                        max_bits: *max_bits,
                    });
                }
                hlssa::OpCode::Lookup {
                    target: LookupTarget::Rangecheck(bits),
                    args,
                    flag,
                } => {
                    assert!(args.len() == 1);
                    assert!(type_info.get_value_type(args[0]).is_field());
                    emitter.push_op(bytecode::OpCode::RngchkField {
                        val: layouter.get_value(args[0]),
                        flag: layouter.get_value(*flag),
                        bits: *bits as usize,
                    });
                }
                hlssa::OpCode::Lookup {
                    target: LookupTarget::Array(arr),
                    args,
                    flag,
                } => {
                    assert!(args.len() == 2);
                    let arr_type = type_info.get_value_type(*arr);
                    let elem_type = arr_type.get_array_element();
                    let (stride, elem_kind) = lookup_elem_kind(&elem_type);
                    emitter.push_op(bytecode::OpCode::ArrayLookupField {
                        array: layouter.get_value(*arr),
                        index: layouter.get_value(args[0]),
                        result: layouter.get_value(args[1]),
                        flag: layouter.get_value(*flag),
                        stride,
                        elem_kind,
                    });
                }
                hlssa::OpCode::DLookup {
                    target: LookupTarget::Rangecheck(bits),
                    args,
                    flag,
                } => {
                    assert!(args.len() == 1);
                    assert!(type_info.get_value_type(args[0]).is_witness_of());
                    emitter.push_op(bytecode::OpCode::DrngchkField {
                        val: layouter.get_value(args[0]),
                        flag: layouter.get_value(*flag),
                        bits: *bits as usize,
                    });
                }
                hlssa::OpCode::DLookup {
                    target: LookupTarget::Array(arr),
                    args,
                    flag,
                } => {
                    assert!(args.len() == 2);
                    let arr_type = type_info.get_value_type(*arr);
                    let elem_type = arr_type.get_array_element();
                    let (stride, elem_kind) = lookup_elem_kind(&elem_type);
                    emitter.push_op(bytecode::OpCode::DarrayLookupField {
                        array: layouter.get_value(*arr),
                        index: layouter.get_value(args[0]),
                        result: layouter.get_value(args[1]),
                        flag: layouter.get_value(*flag),
                        stride,
                        elem_kind,
                    });
                }
                hlssa::OpCode::Lookup {
                    target: LookupTarget::Spread(bits),
                    args,
                    flag,
                } => {
                    assert!(args.len() == 2);
                    emitter.push_op(bytecode::OpCode::SpreadLookupField {
                        val: layouter.get_value(args[0]),
                        result: layouter.get_value(args[1]),
                        flag: layouter.get_value(*flag),
                        bits: *bits as usize,
                    });
                }
                hlssa::OpCode::DLookup {
                    target: LookupTarget::Spread(bits),
                    args,
                    flag,
                } => {
                    assert!(args.len() == 2);
                    emitter.push_op(bytecode::OpCode::DspreadLookupField {
                        val: layouter.get_value(args[0]),
                        result: layouter.get_value(args[1]),
                        flag: layouter.get_value(*flag),
                        bits: *bits as usize,
                    });
                }
                hlssa::OpCode::Lookup {
                    target: LookupTarget::Pow2(size),
                    args,
                    flag,
                } => {
                    assert!(args.len() == 2);
                    emitter.push_op(bytecode::OpCode::Pow2LookupField {
                        amount: layouter.get_value(args[0]),
                        factor: layouter.get_value(args[1]),
                        flag: layouter.get_value(*flag),
                        size: *size as usize,
                    });
                }
                hlssa::OpCode::DLookup {
                    target: LookupTarget::Pow2(size),
                    args,
                    flag,
                } => {
                    assert!(args.len() == 2);
                    emitter.push_op(bytecode::OpCode::Dpow2LookupField {
                        amount: layouter.get_value(args[0]),
                        factor: layouter.get_value(args[1]),
                        flag: layouter.get_value(*flag),
                        size: *size as usize,
                    });
                }
                hlssa::OpCode::Spread { result, value, .. } => {
                    let value_type = type_info.get_value_type(*value);
                    // `Spread` interleaves a bit pattern with zeros; there is no signed form of it,
                    // and the width it actually supports is the `> 32` bound just below.
                    let value_bits = match value_type.strip_witness().expr {
                        TypeExpr::Int(bits) => bits,
                        t => panic!("Unsupported spread value type: {:?}", t),
                    };
                    if value_bits > 32 {
                        todo!("Spread bytecode lowering for integer widths > 32 bits");
                    }
                    let result_type = type_info.get_value_type(*result);
                    let res = match result_type.strip_witness().expr {
                        TypeExpr::Int(bits) => layouter.alloc_int(*result, bits),
                        TypeExpr::Field => layouter.alloc_field(*result),
                        _ => panic!("Unsupported spread result type: {result_type}"),
                    };
                    emitter.push_op(bytecode::OpCode::SpreadU32 {
                        res,
                        val: layouter.get_value(*value),
                    });
                }
                hlssa::OpCode::Unspread {
                    result_odd,
                    result_even,
                    value,
                    ..
                } => {
                    let odd_type = type_info.get_value_type(*result_odd);
                    let even_type = type_info.get_value_type(*result_even);
                    let res_and = match odd_type.strip_witness().expr {
                        TypeExpr::Int(bits) => layouter.alloc_int(*result_odd, bits),
                        TypeExpr::Field => layouter.alloc_field(*result_odd),
                        _ => panic!("Unsupported unspread odd result type: {odd_type}"),
                    };
                    let res_xor = match even_type.strip_witness().expr {
                        TypeExpr::Int(bits) => layouter.alloc_int(*result_even, bits),
                        TypeExpr::Field => layouter.alloc_field(*result_even),
                        _ => panic!("Unsupported unspread even result type: {even_type}"),
                    };
                    emitter.push_op(bytecode::OpCode::UnspreadU64 {
                        res_and,
                        res_xor,
                        val: layouter.get_value(*value),
                    });
                }
                hlssa::OpCode::Todo { payload, .. } => {
                    panic!("Todo opcode encountered in Codegen: {}", payload);
                }
                hlssa::OpCode::InitGlobal { global, value } => {
                    emitter.push_op(bytecode::OpCode::InitGlobal {
                        src: layouter.get_value(*value),
                        global_offset: global_layouter.get_offset(*global),
                        size: global_layouter.get_size(*global),
                    });
                }
                hlssa::OpCode::DropGlobal { global } => {
                    emitter.push_op(bytecode::OpCode::DropGlobal {
                        global_offset: global_layouter.get_offset(*global),
                    });
                }
                hlssa::OpCode::ReadGlobal {
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
                hlssa::OpCode::Alloc { result, value } => {
                    let elem_type = type_info.get_value_type(*value);
                    let res = layouter.alloc_ptr(*result);
                    let elem_size = layouter.type_size(elem_type);
                    let elem_rc = elem_type.is_heap_allocated();
                    let meta = vm::array::BoxedLayout::ref_cell(elem_size, elem_rc);
                    emitter.push_op(bytecode::OpCode::RefAlloc { res, meta });
                    emitter.push_op(bytecode::OpCode::RefStore {
                        cell: layouter.get_value(*result),
                        source: layouter.get_value(*value),
                        stride: elem_size,
                        elem_rc: 0,
                    });
                }
                hlssa::OpCode::Store { ptr, value } => {
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
                hlssa::OpCode::Load { result, ptr } => {
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

// EMITTER STATE
// ================================================================================================

struct EmitterState {
    code: Vec<bytecode::OpCode>,
    source_locations: Vec<bytecode::SourceLocation>,
    current_source_location: bytecode::SourceLocation,
    block_entrances: HashMap<BlockId, usize>,
    block_exits: HashMap<BlockId, usize>,
}

impl EmitterState {
    fn new(current_source_location: bytecode::SourceLocation) -> Self {
        Self {
            code: Vec::new(),
            source_locations: Vec::new(),
            current_source_location,
            block_entrances: HashMap::default(),
            block_exits: HashMap::default(),
        }
    }

    fn push_op(&mut self, op: bytecode::OpCode) {
        self.code.push(op);
        self.source_locations
            .push(self.current_source_location.clone());
    }

    fn set_source_location(&mut self, source_location: bytecode::SourceLocation) {
        self.current_source_location = source_location;
    }

    fn enter_block(&mut self, block: BlockId) {
        self.block_entrances.insert(block, self.code.len());
    }

    fn exit_block(&mut self, block: BlockId) {
        self.block_exits.insert(block, self.code.len());
    }
}

fn emit_assert_r1c(
    layouter: &mut FrameLayouter,
    emitter: &mut EmitterState,
    a: ValueId,
    b: ValueId,
    c: ValueId,
) {
    let product = layouter.alloc_temp_field();
    emitter.push_op(bytecode::OpCode::MulField {
        res: product,
        a: layouter.get_value(a),
        b: layouter.get_value(b),
    });
    emitter.push_op(bytecode::OpCode::AssertEqField {
        a: product,
        b: layouter.get_value(c),
    });
}

// UTILITY FUNCTIONS
// ================================================================================================

/// Returns (stride, elem_kind) for an array element type in a lookup opcode.
fn lookup_elem_kind(elem_type: &Type) -> (usize, usize) {
    match &elem_type.expr {
        // FIELD-ASSUMPTION: L3-felt-limbs
        TypeExpr::Field => (bytecode::FELT_LIMBS, bytecode::ELEM_FIELD),
        // One `ELEM_WORD` per element -- a property of the lookup table's layout, not of how
        // anything reads the element.
        TypeExpr::Int(bits) if *bits <= 64 => (1, bytecode::ELEM_WORD),
        TypeExpr::Int(128) => (2, bytecode::ELEM_U128),
        TypeExpr::Int(_) => panic!("Array lookup unsupported for {elem_type}"),
        TypeExpr::WitnessOf(inner) => {
            let inner_kind = lookup_elem_kind(inner);
            assert!(
                inner_kind.1 != bytecode::ELEM_WITNESS,
                "Nested WitnessOf in array lookup element type: {elem_type}"
            );
            (1, bytecode::ELEM_WITNESS)
        }
        TypeExpr::Array(inner, _) | TypeExpr::Slice(inner) => lookup_elem_kind(inner),
        _ => panic!("Unsupported array element type in lookup: {elem_type}"),
    }
}
