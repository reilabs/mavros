use crate::compiler::ir::r#type::{SSAType, Type};
use itertools::Itertools;
use std::{collections::HashMap, fmt::Display, vec};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct ValueId(pub u64);
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct BlockId(pub u64);
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct FunctionId(pub u64);

pub trait SsaAnnotator {
    fn annotate_value(&self, _function_id: FunctionId, _value_id: ValueId) -> String {
        "".to_string()
    }
    fn annotate_function(&self, _function_id: FunctionId) -> String {
        "".to_string()
    }
    fn annotate_block(&self, _function_id: FunctionId, _block_id: BlockId) -> String {
        "".to_string()
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct DefaultSsaAnnotator;
impl SsaAnnotator for DefaultSsaAnnotator {}

pub trait Instruction: Clone + std::fmt::Debug + 'static {
    fn get_inputs(&self) -> impl Iterator<Item = &ValueId>;
    fn get_results(&self) -> impl Iterator<Item = &ValueId>;
    fn get_inputs_mut(&mut self) -> impl Iterator<Item = &mut ValueId>;
    fn get_operands_mut(&mut self) -> impl Iterator<Item = &mut ValueId>;

    /// Static call targets for building call graphs.
    fn get_static_call_targets(&self) -> Vec<FunctionId>;

    /// Display an instruction. Takes closures for function name resolution
    /// and value annotation (so the trait doesn't depend on SSA).
    fn display_instruction(
        &self,
        func_name: &dyn Fn(FunctionId) -> String,
        annotate_value: &dyn Fn(ValueId) -> String,
    ) -> String;
}

#[derive(Clone)]
pub struct SSA<Op: Instruction, Ty: SSAType> {
    functions: HashMap<FunctionId, Function<Op, Ty>>,
    /// Type of each global slot (indexed by slot number)
    global_types: Vec<Ty>,
    /// Function that initializes all globals (emits InitGlobal opcodes)
    globals_init_fn: Option<FunctionId>,
    /// Function that drops all globals (emits DropGlobal opcodes)
    globals_deinit_fn: Option<FunctionId>,
    main_id: FunctionId,
    next_function_id: u64,
}

pub type HLSSA = SSA<OpCode, Type>;
pub type HLFunction = Function<OpCode, Type>;
pub type HLBlock = Block<OpCode, Type>;

impl HLSSA {
    pub fn new() -> Self {
        Self::with_main("main".to_string())
    }
}

impl<Op: Instruction, Ty: SSAType> SSA<Op, Ty> {
    pub fn with_main(name: String) -> Self {
        let main_function = Function::<Op, Ty>::empty(name);
        let main_id = FunctionId(0);
        let mut functions = HashMap::new();
        functions.insert(main_id, main_function);
        SSA {
            functions,
            global_types: Vec::new(),
            globals_init_fn: None,
            globals_deinit_fn: None,
            main_id,
            next_function_id: 1,
        }
    }
}

impl<Op: Instruction, Ty: SSAType> SSA<Op, Ty> {
    pub fn prepare_rebuild(self) -> (SSA<Op, Ty>, HashMap<FunctionId, Function<Op, Ty>>, Vec<Ty>) {
        (
            SSA {
                functions: HashMap::new(),
                global_types: Vec::new(),
                globals_init_fn: self.globals_init_fn,
                globals_deinit_fn: self.globals_deinit_fn,
                main_id: self.main_id,
                next_function_id: self.next_function_id,
            },
            self.functions,
            self.global_types,
        )
    }

    pub fn insert_function(&mut self, function: Function<Op, Ty>) -> FunctionId {
        let new_id = FunctionId(self.next_function_id);
        self.next_function_id += 1;
        self.functions.insert(new_id, function);
        new_id
    }

    pub fn set_entry_point(&mut self, id: FunctionId) {
        self.main_id = id;
    }

    pub fn get_main_id(&self) -> FunctionId {
        self.main_id
    }

    pub fn get_main_mut(&mut self) -> &mut Function<Op, Ty> {
        self.functions
            .get_mut(&self.main_id)
            .expect("Main function should exist")
    }

    pub fn get_main(&self) -> &Function<Op, Ty> {
        self.functions
            .get(&self.main_id)
            .expect("Main function should exist")
    }

    pub fn get_function(&self, id: FunctionId) -> &Function<Op, Ty> {
        self.functions.get(&id).expect("Function should exist")
    }

    pub fn get_function_mut(&mut self, id: FunctionId) -> &mut Function<Op, Ty> {
        self.functions.get_mut(&id).expect("Function should exist")
    }

    pub fn take_function(&mut self, id: FunctionId) -> Function<Op, Ty> {
        self.functions.remove(&id).expect("Function should exist")
    }

    pub fn put_function(&mut self, id: FunctionId, function: Function<Op, Ty>) {
        self.functions.insert(id, function);
    }

    pub fn add_function(&mut self, name: String) -> FunctionId {
        let new_id = FunctionId(self.next_function_id);
        self.next_function_id += 1;
        let function = Function::<Op, Ty>::empty(name);
        self.functions.insert(new_id, function);
        new_id
    }

    pub fn iter_functions(&self) -> impl Iterator<Item = (&FunctionId, &Function<Op, Ty>)> {
        self.functions.iter()
    }

    pub fn iter_functions_mut(
        &mut self,
    ) -> impl Iterator<Item = (&FunctionId, &mut Function<Op, Ty>)> {
        self.functions.iter_mut()
    }

    pub fn get_function_ids(&self) -> impl Iterator<Item = FunctionId> {
        self.functions.keys().copied()
    }

    pub fn set_global_types(&mut self, types: Vec<Ty>) {
        self.global_types = types;
    }

    pub fn get_global_types(&self) -> &[Ty] {
        &self.global_types
    }

    pub fn num_globals(&self) -> usize {
        self.global_types.len()
    }

    pub fn set_globals_init_fn(&mut self, id: FunctionId) {
        self.globals_init_fn = Some(id);
    }

    pub fn set_globals_deinit_fn(&mut self, id: FunctionId) {
        self.globals_deinit_fn = Some(id);
    }

    pub fn get_globals_init_fn(&self) -> Option<FunctionId> {
        self.globals_init_fn
    }

    pub fn get_globals_deinit_fn(&self) -> Option<FunctionId> {
        self.globals_deinit_fn
    }
}

impl<Op: Instruction, Ty: SSAType> SSA<Op, Ty> {
    pub fn to_string(&self, value_annotator: &dyn SsaAnnotator) -> String {
        let func_name = |id: FunctionId| self.get_function(id).get_name().to_string();
        self.functions
            .iter()
            .sorted_by_key(|(fn_id, _)| fn_id.0)
            .map(|(fn_id, func)| func.to_string(&func_name, *fn_id, value_annotator))
            .join("\n\n")
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum CallTarget {
    Static(FunctionId),
    Dynamic(ValueId),
}

#[derive(Clone)]
pub struct Function<Op: Instruction, Ty: SSAType> {
    entry_block: BlockId,
    blocks: HashMap<BlockId, Block<Op, Ty>>,
    name: String,
    returns: Vec<Ty>,
    next_block: u64,
    next_value: u64,
}

impl<Op: Instruction, Ty: SSAType> Function<Op, Ty> {
    pub fn to_string(
        &self,
        func_name: &dyn Fn(FunctionId) -> String,
        id: FunctionId,
        value_annotator: &dyn SsaAnnotator,
    ) -> String {
        let header = format!(
            "fn {}@{}(block {}) -> {} [{}] {{",
            self.name,
            id.0,
            self.entry_block.0,
            self.returns.iter().map(|t| format!("{}", t)).join(", "),
            value_annotator.annotate_function(id)
        );
        let blocks = self
            .blocks
            .iter()
            .sorted_by_key(|(bid, _)| bid.0)
            .map(|(bid, block)| block.to_string(func_name, id, *bid, value_annotator))
            .join("\n");
        let footer = "}".to_string();
        format!("{}\n{}\n{}", header, blocks, footer)
    }
}

impl<Op: Instruction, Ty: SSAType> Function<Op, Ty> {
    pub fn empty(name: String) -> Self {
        let entry = Block::empty();
        let entry_id = BlockId(0);
        let mut blocks = HashMap::new();
        blocks.insert(entry_id, entry);
        Function {
            entry_block: BlockId(0),
            blocks,
            name,
            next_block: 1,
            returns: Vec::new(),
            next_value: 0,
        }
    }

    pub fn prepare_rebuild(self) -> (Function<Op, Ty>, HashMap<BlockId, Block<Op, Ty>>, Vec<Ty>) {
        (
            Function {
                entry_block: self.entry_block,
                blocks: HashMap::new(),
                next_block: self.next_block,
                name: self.name,
                returns: vec![],
                next_value: self.next_value,
            },
            self.blocks,
            self.returns,
        )
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    pub fn get_var_num_bound(&self) -> usize {
        return self.next_value as usize;
    }

    pub fn get_entry_mut(&mut self) -> &mut Block<Op, Ty> {
        self.blocks
            .get_mut(&self.entry_block)
            .expect("Entry block should exist")
    }

    pub fn get_entry(&self) -> &Block<Op, Ty> {
        self.blocks
            .get(&self.entry_block)
            .expect("Entry block should exist")
    }

    pub fn get_entry_id(&self) -> BlockId {
        self.entry_block
    }

    pub fn get_block(&self, id: BlockId) -> &Block<Op, Ty> {
        self.blocks.get(&id).expect("Block should exist")
    }

    pub fn get_block_mut(&mut self, id: BlockId) -> &mut Block<Op, Ty> {
        self.blocks.get_mut(&id).expect("Block should exist")
    }

    pub fn take_block(&mut self, id: BlockId) -> Block<Op, Ty> {
        self.blocks.remove(&id).expect("Block should exist")
    }

    pub fn put_block(&mut self, id: BlockId, block: Block<Op, Ty>) {
        self.blocks.insert(id, block);
    }

    pub fn add_block(&mut self) -> BlockId {
        let (id, _) = self.add_block_mut();
        id
    }

    pub fn add_block_mut(&mut self) -> (BlockId, &mut Block<Op, Ty>) {
        let new_id = BlockId(self.next_block);
        self.next_block += 1;
        self.blocks.insert(new_id, Block::empty());
        (new_id, self.blocks.get_mut(&new_id).unwrap())
    }

    pub fn block_is_terminated(&self, block_id: BlockId) -> bool {
        self.blocks
            .get(&block_id)
            .unwrap()
            .get_terminator()
            .is_some()
    }

    pub fn next_virtual_block(&mut self) -> (BlockId, Block<Op, Ty>) {
        let new_id = BlockId(self.next_block);
        self.next_block += 1;
        let block = Block::empty();
        (new_id, block)
    }

    pub fn add_return_type(&mut self, typ: Ty) {
        self.returns.push(typ);
    }

    pub fn get_param_types(&self) -> Vec<Ty> {
        self.get_entry()
            .parameters
            .iter()
            .map(|(_, typ)| typ.clone())
            .collect()
    }

    pub fn iter_returns_mut(&mut self) -> impl Iterator<Item = &mut Ty> {
        self.returns.iter_mut()
    }

    pub fn get_returns(&self) -> &[Ty] {
        &self.returns
    }

    pub fn get_blocks(&self) -> impl Iterator<Item = (&BlockId, &Block<Op, Ty>)> {
        self.blocks.iter()
    }

    pub fn add_parameter(&mut self, block_id: BlockId, typ: Ty) -> ValueId {
        let value_id = ValueId(self.next_value);
        self.next_value += 1;
        self.blocks
            .get_mut(&block_id)
            .unwrap()
            .parameters
            .push((value_id, typ));
        value_id
    }

    pub fn get_blocks_mut(&mut self) -> impl Iterator<Item = (&BlockId, &mut Block<Op, Ty>)> {
        self.blocks.iter_mut()
    }

    pub fn take_blocks(&mut self) -> HashMap<BlockId, Block<Op, Ty>> {
        std::mem::take(&mut self.blocks)
    }

    pub fn put_blocks(&mut self, blocks: HashMap<BlockId, Block<Op, Ty>>) {
        self.blocks = blocks;
    }

    pub fn fresh_value(&mut self) -> ValueId {
        let value_id = ValueId(self.next_value);
        self.next_value += 1;
        value_id
    }

    pub fn take_returns(&mut self) -> Vec<Ty> {
        std::mem::take(&mut self.returns)
    }

    pub fn code_size(&self) -> usize {
        self.blocks
            .values()
            .map(|b| {
                b.instructions
                    .iter()
                    .map(|i| i.get_inputs().count() + 1)
                    .sum::<usize>()
            })
            .sum()
    }

    pub fn terminate_block_with_jmp_if(
        &mut self,
        block_id: BlockId,
        condition: ValueId,
        then_destination: BlockId,
        else_destination: BlockId,
    ) {
        self.blocks
            .get_mut(&block_id)
            .unwrap()
            .set_terminator(Terminator::JmpIf(
                condition,
                then_destination,
                else_destination,
            ));
    }

    pub fn terminate_block_with_return(&mut self, block_id: BlockId, return_values: Vec<ValueId>) {
        self.blocks
            .get_mut(&block_id)
            .unwrap()
            .set_terminator(Terminator::Return(return_values));
    }

    pub fn terminate_block_with_jmp(
        &mut self,
        block_id: BlockId,
        destination: BlockId,
        arguments: Vec<ValueId>,
    ) {
        self.blocks
            .get_mut(&block_id)
            .unwrap()
            .set_terminator(Terminator::Jmp(destination, arguments));
    }
}

#[derive(Clone)]
pub struct Block<Op: Instruction, Ty: SSAType> {
    parameters: Vec<(ValueId, Ty)>,
    instructions: Vec<Op>,
    terminator: Option<Terminator>,
}

impl<Op: Instruction, Ty: SSAType> Block<Op, Ty> {
    pub fn to_string(
        &self,
        func_name: &dyn Fn(FunctionId) -> String,
        func_id: FunctionId,
        id: BlockId,
        value_annotator: &dyn SsaAnnotator,
    ) -> String {
        let params = self
            .parameters
            .iter()
            .map(|v| {
                let annotation = value_annotator.annotate_value(func_id, v.0);
                let annotation = if annotation.is_empty() {
                    "".to_string()
                } else {
                    format!(" [{}]", annotation)
                };
                format!("v{} : {}{}", v.0.0, v.1.to_string(), annotation)
            })
            .join(", ");
        let annotate_value = |value: ValueId| -> String {
            let annotation = value_annotator.annotate_value(func_id, value);
            if annotation.is_empty() {
                "".to_string()
            } else {
                format!("[{}]", annotation)
            }
        };
        let instructions = self
            .instructions
            .iter()
            .map(|i| format!("    {}", i.display_instruction(func_name, &annotate_value)))
            .join("\n");
        let terminator = match &self.terminator {
            Some(t) => format!("    {}", t.to_string()),
            None => "".to_string(),
        };
        let block_annotation = value_annotator.annotate_block(func_id, id);
        let block_annotation = if block_annotation.is_empty() {
            "".to_string()
        } else {
            format!(" [{}]", block_annotation)
        };
        format!(
            "  block_{}({}){} {{\n{}\n{}\n  }}",
            id.0, params, block_annotation, instructions, terminator
        )
    }
}

impl<Op: Instruction, Ty: SSAType> Block<Op, Ty> {
    pub fn empty() -> Self {
        Block {
            parameters: Vec::new(),
            instructions: Vec::new(),
            terminator: None,
        }
    }

    pub fn take_instructions(&mut self) -> Vec<Op> {
        std::mem::take(&mut self.instructions)
    }

    pub fn put_instructions(&mut self, instructions: Vec<Op>) {
        self.instructions = instructions;
    }

    pub fn push_instruction(&mut self, instruction: Op) {
        self.instructions.push(instruction);
    }

    pub fn set_terminator(&mut self, terminator: Terminator) {
        self.terminator = Some(terminator);
    }

    pub fn get_parameters(&self) -> impl Iterator<Item = &(ValueId, Ty)> {
        self.parameters.iter()
    }

    pub fn get_parameters_mut(&mut self) -> impl Iterator<Item = &mut (ValueId, Ty)> {
        self.parameters.iter_mut()
    }

    pub fn take_parameters(&mut self) -> Vec<(ValueId, Ty)> {
        std::mem::take(&mut self.parameters)
    }

    pub fn put_parameters(&mut self, parameters: Vec<(ValueId, Ty)>) {
        self.parameters = parameters;
    }

    pub fn push_parameter(&mut self, value_id: ValueId, typ: Ty) {
        self.parameters.push((value_id, typ));
    }

    pub fn get_parameter_values(&self) -> impl Iterator<Item = &ValueId> {
        self.parameters.iter().map(|(id, _)| id)
    }

    pub fn get_instruction(&self, i: usize) -> &Op {
        &self.instructions[i]
    }

    pub fn get_instructions(&self) -> impl DoubleEndedIterator<Item = &Op> {
        self.instructions.iter()
    }

    pub fn get_instructions_mut(&mut self) -> impl Iterator<Item = &mut Op> {
        self.instructions.iter_mut()
    }

    pub fn get_terminator(&self) -> Option<&Terminator> {
        self.terminator.as_ref()
    }

    pub fn get_terminator_mut(&mut self) -> &mut Terminator {
        self.terminator.as_mut().unwrap()
    }

    pub fn take_terminator(&mut self) -> Option<Terminator> {
        std::mem::take(&mut self.terminator)
    }

    pub fn has_parameters(&self) -> bool {
        self.parameters.len() > 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryArithOpKind {
    Add,
    Mul,
    Div,
    Sub,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Mod,
}

#[derive(Debug, Clone, Copy)]
pub enum CmpKind {
    Lt,
    Eq,
}

#[derive(Debug, Clone, Copy)]
pub enum SeqType {
    Array(usize),
    Slice,
    Tuple,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CastTarget {
    Field,
    U(usize),
    I(usize),
    WitnessOf,
    Nop,
    ArrayToSlice,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endianness {
    Big,
    Little,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceOpDir {
    Front,
    Back,
}

impl Display for CastTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CastTarget::Field => write!(f, "Field"),
            CastTarget::U(size) => write!(f, "u{}", size),
            CastTarget::I(size) => write!(f, "i{}", size),
            CastTarget::WitnessOf => write!(f, "WitnessOf"),
            CastTarget::Nop => write!(f, "Nop"),
            CastTarget::ArrayToSlice => write!(f, "ArrayToSlice"),
        }
    }
}

impl Display for Endianness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Endianness::Big => write!(f, "big"),
            Endianness::Little => write!(f, "little"),
        }
    }
}

impl Display for SeqType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SeqType::Array(len) => write!(f, "Array[{}]", len),
            SeqType::Slice => write!(f, "Slice"),
            SeqType::Tuple => write!(f, "Tuple"),
        }
    }
}

impl SeqType {
    pub fn of(&self, t: Type) -> Type {
        match self {
            SeqType::Array(len) => t.array_of(*len),
            SeqType::Slice => t.slice_of(),
            SeqType::Tuple => panic!("Tuple type requires multiple element types"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum MemOp {
    Bump(usize),
    Drop,
}

#[derive(Debug, Clone, Copy)]
pub enum DMatrix {
    A,
    B,
    C,
}

#[derive(Debug, Clone, Copy)]
pub enum LookupTarget<V> {
    Rangecheck(u8),
    DynRangecheck(V),
    Array(V),
    Spread(u8),
}

#[derive(Debug, Clone, Copy)]
pub enum Radix<V> {
    Bytes,
    Dyn(V),
}

#[derive(Debug, Clone)]
pub enum OpCode {
    Cmp {
        kind: CmpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
    },
    BinaryArithOp {
        kind: BinaryArithOpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
    },
    Cast {
        result: ValueId,
        value: ValueId,
        target: CastTarget,
    },
    Truncate {
        result: ValueId,
        value: ValueId,
        to_bits: usize,
        from_bits: usize,
    },
    SExt {
        result: ValueId,
        value: ValueId,
        from_bits: usize,
        to_bits: usize,
    },
    Not {
        result: ValueId,
        value: ValueId,
    },
    MkSeq {
        result: ValueId,
        elems: Vec<ValueId>,
        seq_type: SeqType,
        elem_type: Type,
    },
    Alloc {
        result: ValueId,
        elem_type: Type,
    },
    Store {
        ptr: ValueId,
        value: ValueId,
    },
    Load {
        result: ValueId,
        ptr: ValueId,
    },
    AssertEq {
        lhs: ValueId,
        rhs: ValueId,
    },
    AssertR1C {
        a: ValueId,
        b: ValueId,
        c: ValueId,
    },
    Call {
        results: Vec<ValueId>,
        function: CallTarget,
        args: Vec<ValueId>,
        unconstrained: bool,
    },
    ArrayGet {
        result: ValueId,
        array: ValueId,
        index: ValueId,
    },
    ArraySet {
        result: ValueId,
        array: ValueId,
        index: ValueId,
        value: ValueId,
    },
    SlicePush {
        dir: SliceOpDir,
        result: ValueId,
        slice: ValueId,
        values: Vec<ValueId>,
    },
    SliceLen {
        result: ValueId,
        slice: ValueId,
    },
    Select {
        result: ValueId,
        cond: ValueId,
        if_t: ValueId,
        if_f: ValueId,
    },
    ToBits {
        result: ValueId,
        value: ValueId,
        endianness: Endianness,
        count: usize,
    },
    ToRadix {
        result: ValueId,
        value: ValueId,
        radix: Radix<ValueId>,
        endianness: Endianness,
        count: usize,
    },
    MemOp {
        kind: MemOp,
        value: ValueId,
    },
    ValueOf {
        result: ValueId,
        value: ValueId,
    },
    WriteWitness {
        result: Option<ValueId>,
        value: ValueId,
        pinned: bool,
    },
    FreshWitness {
        result: ValueId,
        result_type: Type,
    },
    NextDCoeff {
        result: ValueId,
    },
    BumpD {
        matrix: DMatrix,
        variable: ValueId,
        sensitivity: ValueId,
    },
    Constrain {
        a: ValueId,
        b: ValueId,
        c: ValueId,
    },
    Lookup {
        target: LookupTarget<ValueId>,
        keys: Vec<ValueId>,
        results: Vec<ValueId>,
        flag: ValueId,
    },
    DLookup {
        target: LookupTarget<ValueId>,
        keys: Vec<ValueId>,
        results: Vec<ValueId>,
        flag: ValueId,
    },
    MulConst {
        result: ValueId,
        const_val: ValueId,
        var: ValueId,
    },
    Rangecheck {
        value: ValueId,
        max_bits: usize,
    },
    ReadGlobal {
        result: ValueId,
        offset: u64,
        result_type: Type,
    },
    TupleProj {
        result: ValueId,
        tuple: ValueId,
        idx: usize,
    },
    MkTuple {
        result: ValueId,
        elems: Vec<ValueId>,
        element_types: Vec<Type>,
    },
    Todo {
        payload: String,
        results: Vec<ValueId>,
        result_types: Vec<Type>,
    },
    InitGlobal {
        global: usize,
        value: ValueId,
    },
    DropGlobal {
        global: usize,
    },
    Const {
        result: ValueId,
        value: ConstValue,
    },
    Spread {
        result: ValueId,
        value: ValueId,
    },
    Unspread {
        result_odd: ValueId,
        result_even: ValueId,
        value: ValueId,
    },
    Guard {
        condition: ValueId,
        inner: Box<OpCode>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConstValue {
    U(usize, u128),
    I(usize, u128),
    Field(ark_bn254::Fr),
    FnPtr(FunctionId),
}

impl Instruction for OpCode {
    fn get_static_call_targets(&self) -> Vec<FunctionId> {
        match self {
            OpCode::Call {
                function: CallTarget::Static(id),
                ..
            } => vec![*id],
            OpCode::Guard { inner, .. } => inner.get_static_call_targets(),
            _ => vec![],
        }
    }

    fn display_instruction(
        &self,
        func_name: &dyn Fn(FunctionId) -> String,
        annotate_value: &dyn Fn(ValueId) -> String,
    ) -> String {
        match self {
            OpCode::Cmp {
                kind,
                result,
                lhs,
                rhs,
            } => {
                let op_str = match kind {
                    CmpKind::Lt => "<",
                    CmpKind::Eq => "==",
                };
                format!(
                    "v{}{} = v{} {} v{}",
                    result.0,
                    annotate_value(*result),
                    lhs.0,
                    op_str,
                    rhs.0
                )
            }
            OpCode::BinaryArithOp {
                kind,
                result,
                lhs,
                rhs,
            } => {
                let op_str = match kind {
                    BinaryArithOpKind::Add => "+",
                    BinaryArithOpKind::Sub => "-",
                    BinaryArithOpKind::Mul => "*",
                    BinaryArithOpKind::Div => "/",
                    BinaryArithOpKind::And => "&",
                    BinaryArithOpKind::Or => "|",
                    BinaryArithOpKind::Xor => "^",
                    BinaryArithOpKind::Shl => "<<",
                    BinaryArithOpKind::Shr => ">>",
                    BinaryArithOpKind::Mod => "%",
                };
                format!(
                    "v{}{} = v{} {} v{}",
                    result.0,
                    annotate_value(*result),
                    lhs.0,
                    op_str,
                    rhs.0
                )
            }
            OpCode::Alloc {
                result,
                elem_type: typ,
            } => format!("v{}{} = alloc({})", result.0, annotate_value(*result), typ),
            OpCode::Store { ptr, value } => {
                format!("*v{}{} = v{}", ptr.0, annotate_value(*ptr), value.0)
            }
            OpCode::Load { result, ptr } => {
                format!("v{}{} = *v{}", result.0, annotate_value(*result), ptr.0)
            }
            OpCode::AssertEq { lhs, rhs } => format!("assert v{} == v{}", lhs.0, rhs.0),
            OpCode::AssertR1C {
                a: lhs,
                b: rhs,
                c: cond,
            } => {
                format!("assert v{} * v{} - v{} == 0", lhs.0, rhs.0, cond.0)
            }
            OpCode::Call {
                results: result,
                function,
                args,
                unconstrained,
            } => {
                let args_str = args.iter().map(|v| format!("v{}", v.0)).join(", ");
                let result_str = result
                    .iter()
                    .map(|v| format!("v{}{}", v.0, annotate_value(*v)))
                    .join(", ");
                let call_prefix = if *unconstrained {
                    "call_unconstrained"
                } else {
                    "call"
                };
                match function {
                    CallTarget::Static(fn_id) => {
                        format!(
                            "{} = {} {}@{}({})",
                            result_str,
                            call_prefix,
                            func_name(*fn_id),
                            fn_id.0,
                            args_str
                        )
                    }
                    CallTarget::Dynamic(fn_ptr) => {
                        format!(
                            "{} = {}_indirect v{}({})",
                            result_str, call_prefix, fn_ptr.0, args_str
                        )
                    }
                }
            }
            OpCode::ArrayGet {
                result,
                array,
                index,
            } => {
                format!(
                    "v{}{} = v{}[v{}]",
                    result.0,
                    annotate_value(*result),
                    array.0,
                    index.0
                )
            }
            OpCode::ArraySet {
                result,
                array,
                index,
                value: element,
            } => {
                format!(
                    "v{}{} = (v{}[v{}] = v{})",
                    result.0,
                    annotate_value(*result),
                    array.0,
                    index.0,
                    element.0
                )
            }
            OpCode::SlicePush {
                dir,
                result,
                slice,
                values,
            } => {
                let dir_str = match dir {
                    SliceOpDir::Front => "front",
                    SliceOpDir::Back => "back",
                };
                let values_str = values.iter().map(|v| format!("v{}", v.0)).join(", ");
                format!(
                    "v{}{} = slice_push_{}(v{}, [{}])",
                    result.0,
                    annotate_value(*result),
                    dir_str,
                    slice.0,
                    values_str
                )
            }
            OpCode::SliceLen { result, slice } => {
                format!(
                    "v{}{} = slice_len(v{})",
                    result.0,
                    annotate_value(*result),
                    slice.0
                )
            }
            OpCode::Select {
                result,
                cond,
                if_t: then,
                if_f: otherwise,
            } => {
                format!(
                    "v{}{} = v{} ? v{} : v{}",
                    result.0,
                    annotate_value(*result),
                    cond.0,
                    then.0,
                    otherwise.0
                )
            }
            OpCode::WriteWitness {
                result,
                value,
                pinned,
            } => {
                let r_str = if let Some(result) = result {
                    format!("v{}{} = ", result.0, annotate_value(*result))
                } else {
                    "".to_string()
                };
                let pinned_str = if *pinned { " [pinned]" } else { "" };
                format!("{}write_witness(v{}){}", r_str, value.0, pinned_str)
            }
            OpCode::FreshWitness {
                result,
                result_type: typ,
            } => {
                format!(
                    "v{}{} = fresh_witness(): {}",
                    result.0,
                    annotate_value(*result),
                    typ
                )
            }
            OpCode::Constrain { a, b, c } => {
                format!("constrain_r1c(v{} * v{} - v{} == 0)", a.0, b.0, c.0)
            }
            OpCode::Lookup {
                target,
                keys,
                results,
                flag,
            } => {
                let keys_str = keys.iter().map(|v| format!("v{}", v.0)).join(", ");
                let results_str = results.iter().map(|v| format!("v{}", v.0)).join(", ");
                let target_str = match target {
                    LookupTarget::Rangecheck(n) => format!("rngchk({})", n),
                    LookupTarget::DynRangecheck(v) => format!("rngchk(_ < v{})", v.0),
                    LookupTarget::Array(arr) => format!("v{}", arr.0),
                    LookupTarget::Spread(n) => format!("spread({})", n),
                };
                format!(
                    "constrain_lookup({}, ({}) => ({}), flag=v{})",
                    target_str, keys_str, results_str, flag.0
                )
            }
            OpCode::NextDCoeff { result } => {
                format!("v{}{} = next_d_coeff()", result.0, annotate_value(*result))
            }
            OpCode::BumpD {
                matrix,
                variable: result,
                sensitivity: value,
            } => {
                let matrix_str = match matrix {
                    DMatrix::A => "A",
                    DMatrix::B => "B",
                    DMatrix::C => "C",
                };
                format!("∂{} / ∂v{} += v{}", matrix_str, result.0, value.0)
            }
            OpCode::DLookup {
                target,
                keys,
                results,
                flag,
            } => {
                let keys_str = keys.iter().map(|v| format!("v{}", v.0)).join(", ");
                let results_str = results.iter().map(|v| format!("v{}", v.0)).join(", ");
                let target_str = match target {
                    LookupTarget::Rangecheck(n) => format!("rngchk({})", n),
                    LookupTarget::DynRangecheck(v) => format!("rngchk(_ < v{})", v.0),
                    LookupTarget::Array(arr) => format!("v{}", arr.0),
                    LookupTarget::Spread(n) => format!("spread({})", n),
                };
                format!(
                    "∂lookup({}, ({}) => ({}), flag=v{})",
                    target_str, keys_str, results_str, flag.0
                )
            }
            OpCode::MkSeq {
                result,
                elems: values,
                seq_type,
                elem_type: typ,
            } => {
                let values_str = values.iter().map(|v| format!("v{}", v.0)).join(", ");
                format!(
                    "v{}{} = [{}] : {} of {}",
                    result.0,
                    annotate_value(*result),
                    values_str,
                    seq_type,
                    typ
                )
            }
            OpCode::Cast {
                result,
                value,
                target,
            } => {
                format!(
                    "v{}{} = cast v{} to {}",
                    result.0,
                    annotate_value(*result),
                    value.0,
                    target
                )
            }
            OpCode::Truncate {
                result,
                value,
                to_bits: out_bits,
                from_bits: in_bits,
            } => {
                format!(
                    "v{}{} = truncate v{} from {} bits to {} bits",
                    result.0,
                    annotate_value(*result),
                    value.0,
                    in_bits,
                    out_bits
                )
            }
            OpCode::SExt {
                result,
                value,
                from_bits: in_bits,
                to_bits: out_bits,
            } => {
                format!(
                    "v{}{} = sext v{} from {} bits to {} bits",
                    result.0,
                    annotate_value(*result),
                    value.0,
                    in_bits,
                    out_bits
                )
            }
            OpCode::Not { result, value } => {
                format!("v{}{} = ~v{}", result.0, annotate_value(*result), value.0)
            }
            OpCode::ValueOf { result, value } => {
                format!(
                    "v{}{} = value_of v{}",
                    result.0,
                    annotate_value(*result),
                    value.0
                )
            }
            OpCode::ToBits {
                result,
                value,
                endianness,
                count: output_size,
            } => {
                format!(
                    "v{}{} = to_bits v{} (endianness: {}, size: {})",
                    result.0,
                    annotate_value(*result),
                    value.0,
                    endianness,
                    output_size
                )
            }
            OpCode::ToRadix {
                result,
                value,
                radix,
                endianness,
                count: output_size,
            } => {
                let radix_str = match radix {
                    Radix::Bytes => "bytes".to_string(),
                    Radix::Dyn(radix) => format!("v{}", radix.0),
                };
                format!(
                    "v{}{} = to_radix v{} {} (endianness: {}, size: {})",
                    result.0,
                    annotate_value(*result),
                    value.0,
                    radix_str,
                    endianness,
                    output_size
                )
            }
            OpCode::MemOp { kind, value } => {
                let name = match kind {
                    MemOp::Bump(n) => format!("inc_rc[+{}]", n),
                    MemOp::Drop => "drop".to_string(),
                };
                format!("{}(v{})", name, value.0)
            }
            OpCode::MulConst {
                result,
                const_val: constant,
                var,
            } => {
                format!(
                    "v{}{} = mul_const(v{}, v{})",
                    result.0,
                    annotate_value(*result),
                    constant.0,
                    var.0
                )
            }
            OpCode::Rangecheck {
                value: val,
                max_bits,
            } => {
                format!("rangecheck(v{}, {})", val.0, max_bits)
            }
            OpCode::ReadGlobal {
                result,
                offset: index,
                result_type: typ,
            } => {
                format!(
                    "v{}{} = read_global(g{}, {})",
                    result.0,
                    annotate_value(*result),
                    index,
                    typ
                )
            }
            OpCode::TupleProj { result, tuple, idx } => {
                format!(
                    "v{}{} = v{}.{}",
                    result.0,
                    annotate_value(*result),
                    tuple.0,
                    idx
                )
            }
            OpCode::MkTuple {
                result,
                elems,
                element_types: _,
            } => {
                let elems_str = elems.iter().map(|v| format!("v{}", v.0)).join(", ");
                format!("v{}{} = ({})", result.0, annotate_value(*result), elems_str)
            }
            OpCode::Todo {
                payload,
                results,
                result_types,
            } => {
                let results_str = results
                    .iter()
                    .zip(result_types.iter())
                    .map(|(r, tp)| format!("v{}: {}", r.0, tp))
                    .join(", ");
                format!("todo(\"{}\", [{}])", payload, results_str)
            }
            OpCode::InitGlobal { global, value } => {
                format!("init_global({}, v{})", global, value.0)
            }
            OpCode::DropGlobal { global } => {
                format!("drop_global({})", global)
            }
            OpCode::Const { result, value } => match value {
                ConstValue::U(size, val) => {
                    format!(
                        "v{}{} = u_const({}, {})",
                        result.0,
                        annotate_value(*result),
                        size,
                        val
                    )
                }
                ConstValue::I(size, val) => {
                    format!(
                        "v{}{} = i_const({}, {})",
                        result.0,
                        annotate_value(*result),
                        size,
                        val
                    )
                }
                ConstValue::Field(val) => {
                    format!(
                        "v{}{} = field_const({})",
                        result.0,
                        annotate_value(*result),
                        val
                    )
                }
                ConstValue::FnPtr(fn_id) => {
                    format!(
                        "v{}{} = fn_ptr_const({}@{})",
                        result.0,
                        annotate_value(*result),
                        func_name(*fn_id),
                        fn_id.0
                    )
                }
            },
            OpCode::Spread { result, value } => {
                format!(
                    "v{}{} = spread(v{})",
                    result.0,
                    annotate_value(*result),
                    value.0
                )
            }
            OpCode::Unspread {
                result_odd,
                result_even,
                value,
            } => {
                format!(
                    "v{}{}, v{}{} = unspread(v{})",
                    result_odd.0,
                    annotate_value(*result_odd),
                    result_even.0,
                    annotate_value(*result_even),
                    value.0
                )
            }
            OpCode::Guard { condition, inner } => {
                format!(
                    "guard(v{}) {{ {} }}",
                    condition.0,
                    inner.display_instruction(func_name, annotate_value)
                )
            }
        }
    }

    fn get_inputs(&self) -> impl Iterator<Item = &ValueId> {
        match self {
            Self::Alloc {
                result: _,
                elem_type: _,
            }
            | Self::FreshWitness {
                result: _,
                result_type: _,
            }
            | Self::NextDCoeff { result: _ }
            | Self::Const { .. } => vec![].into_iter(),
            Self::Cmp {
                kind: _,
                result: _,
                lhs: b,
                rhs: c,
            }
            | Self::BinaryArithOp {
                kind: _,
                result: _,
                lhs: b,
                rhs: c,
            }
            | Self::ArrayGet {
                result: _,
                array: b,
                index: c,
            } => vec![b, c].into_iter(),
            Self::Spread {
                result: _,
                value: v,
            } => vec![v].into_iter(),
            Self::Unspread {
                result_odd: _,
                result_even: _,
                value: v,
            } => vec![v].into_iter(),
            Self::ArraySet {
                result: _,
                array: b,
                index: c,
                value: d,
            } => vec![b, c, d].into_iter(),
            Self::SlicePush {
                dir: _,
                result: _,
                slice: b,
                values: c,
            } => {
                let mut ret_vec = vec![b];
                ret_vec.extend(c.iter());
                ret_vec.into_iter()
            }
            Self::SliceLen {
                result: _,
                slice: b,
            } => vec![b].into_iter(),
            Self::AssertEq { lhs: b, rhs: c }
            | Self::Store { ptr: b, value: c }
            | Self::BumpD {
                matrix: _,
                variable: b,
                sensitivity: c,
            }
            | Self::MulConst {
                result: _,
                const_val: b,
                var: c,
            } => vec![b, c].into_iter(),
            Self::Load { result: _, ptr: c }
            | Self::WriteWitness {
                result: _,
                value: c,
                pinned: _,
            }
            | Self::Cast {
                result: _,
                value: c,
                target: _,
            }
            | Self::Truncate {
                result: _,
                value: c,
                to_bits: _,
                from_bits: _,
            }
            | Self::SExt {
                result: _,
                value: c,
                from_bits: _,
                to_bits: _,
            } => vec![c].into_iter(),
            Self::Call {
                results: _,
                function,
                args: a,
                unconstrained: _,
            } => {
                let mut ret_vec = Vec::new();
                if let CallTarget::Dynamic(fn_ptr) = function {
                    ret_vec.push(fn_ptr);
                }
                ret_vec.extend(a.iter());
                ret_vec.into_iter()
            }
            Self::MkSeq {
                result: _,
                elems: inputs,
                seq_type: _,
                elem_type: _,
            } => inputs.iter().collect::<Vec<_>>().into_iter(),
            Self::Select {
                result: _,
                cond: b,
                if_t: c,
                if_f: d,
            }
            | Self::AssertR1C { a: b, b: c, c: d }
            | Self::Constrain { a: b, b: c, c: d } => vec![b, c, d].into_iter(),
            Self::Not {
                result: _,
                value: v,
            }
            | Self::ValueOf {
                result: _,
                value: v,
            } => vec![v].into_iter(),
            Self::ToBits {
                result: _,
                value: v,
                endianness: _,
                count: _,
            } => vec![v].into_iter(),
            Self::ToRadix {
                result: _,
                value: v,
                radix,
                endianness: _,
                count: _,
            } => {
                let mut ret_vec = vec![v];
                match radix {
                    Radix::Bytes => {}
                    Radix::Dyn(radix) => {
                        ret_vec.push(radix);
                    }
                }
                ret_vec.into_iter()
            }
            Self::MemOp { kind: _, value: v } => vec![v].into_iter(),
            Self::Rangecheck {
                value: val,
                max_bits: _,
            } => vec![val].into_iter(),
            Self::ReadGlobal {
                result: _,
                offset: _,
                result_type: _,
            } => vec![].into_iter(),
            Self::Lookup {
                target,
                keys,
                results,
                flag,
            }
            | Self::DLookup {
                target,
                keys,
                results,
                flag,
            } => {
                let mut ret_vec = vec![];
                match target {
                    LookupTarget::Rangecheck(_) | LookupTarget::Spread(_) => {}
                    LookupTarget::DynRangecheck(v) => {
                        ret_vec.push(v);
                    }
                    LookupTarget::Array(arr) => {
                        ret_vec.push(arr);
                    }
                }
                ret_vec.extend(keys);
                ret_vec.extend(results);
                ret_vec.push(flag);
                ret_vec.into_iter()
            }
            Self::TupleProj {
                result: _,
                tuple,
                idx: _,
            } => vec![tuple].into_iter(),
            OpCode::MkTuple {
                result: _,
                elems: e,
                element_types: _,
            } => e.iter().collect::<Vec<_>>().into_iter(),
            Self::Todo { .. } => vec![].into_iter(),
            Self::InitGlobal {
                global: _,
                value: v,
            } => vec![v].into_iter(),
            Self::DropGlobal { global: _ } => vec![].into_iter(),
            Self::Guard { condition, inner } => {
                let mut ret_vec = vec![condition];
                ret_vec.extend(inner.get_inputs());
                ret_vec.into_iter()
            }
        }
    }

    fn get_results(&self) -> impl Iterator<Item = &ValueId> {
        match self {
            Self::Alloc {
                result: r,
                elem_type: _,
            }
            | Self::FreshWitness {
                result: r,
                result_type: _,
            }
            | Self::Const { result: r, .. }
            | Self::Cmp {
                kind: _,
                result: r,
                lhs: _,
                rhs: _,
            }
            | Self::BinaryArithOp {
                kind: _,
                result: r,
                lhs: _,
                rhs: _,
            }
            | Self::ArrayGet {
                result: r,
                array: _,
                index: _,
            }
            | Self::ArraySet {
                result: r,
                array: _,
                index: _,
                value: _,
            }
            | Self::SlicePush {
                dir: _,
                result: r,
                slice: _,
                values: _,
            }
            | Self::SliceLen {
                result: r,
                slice: _,
            }
            | Self::Load { result: r, ptr: _ }
            | Self::MkSeq {
                result: r,
                elems: _,
                seq_type: _,
                elem_type: _,
            }
            | Self::Select {
                result: r,
                cond: _,
                if_t: _,
                if_f: _,
            }
            | Self::Cast {
                result: r,
                value: _,
                target: _,
            }
            | Self::Truncate {
                result: r,
                value: _,
                to_bits: _,
                from_bits: _,
            }
            | Self::SExt {
                result: r,
                value: _,
                from_bits: _,
                to_bits: _,
            }
            | Self::MulConst {
                result: r,
                const_val: _,
                var: _,
            }
            | Self::NextDCoeff { result: r }
            | Self::TupleProj {
                result: r,
                tuple: _,
                idx: _,
            }
            | Self::MkTuple {
                result: r,
                elems: _,
                element_types: _,
            } => vec![r].into_iter(),
            Self::WriteWitness {
                result: r,
                value: _,
                pinned: _,
            } => {
                let ret_vec = r.iter().collect::<Vec<_>>();
                ret_vec.into_iter()
            }
            Self::Call {
                results: r,
                function: _,
                args: _,
                unconstrained: _,
            } => r.iter().collect::<Vec<_>>().into_iter(),
            Self::Constrain { .. }
            | Self::BumpD {
                matrix: _,
                variable: _,
                sensitivity: _,
            }
            | Self::MemOp { kind: _, value: _ }
            | Self::Store { ptr: _, value: _ }
            | Self::AssertEq { lhs: _, rhs: _ }
            | Self::AssertR1C { a: _, b: _, c: _ }
            | Self::Rangecheck {
                value: _,
                max_bits: _,
            } => vec![].into_iter(),
            Self::Not {
                result: r,
                value: _,
            }
            | Self::ValueOf {
                result: r,
                value: _,
            }
            | Self::Spread {
                result: r,
                value: _,
            } => vec![r].into_iter(),
            Self::Unspread {
                result_odd,
                result_even,
                value: _,
            } => vec![result_odd, result_even].into_iter(),
            Self::ToBits {
                result: r,
                value: _,
                endianness: _,
                count: _,
            } => vec![r].into_iter(),
            Self::ToRadix {
                result: r,
                value: _,
                radix: _,
                endianness: _,
                count: _,
            } => vec![r].into_iter(),
            Self::ReadGlobal {
                result: r,
                offset: _,
                result_type: _,
            } => vec![r].into_iter(),
            Self::Lookup { .. } | Self::DLookup { .. } => vec![].into_iter(),
            Self::Todo { results, .. } => {
                let ret_vec: Vec<&ValueId> = results.iter().collect();
                ret_vec.into_iter()
            }
            Self::InitGlobal {
                global: _,
                value: _,
            } => vec![].into_iter(),
            Self::DropGlobal { global: _ } => vec![].into_iter(),
            Self::Guard { inner, .. } => inner.get_results().collect::<Vec<_>>().into_iter(),
        }
    }

    fn get_inputs_mut(&mut self) -> impl Iterator<Item = &mut ValueId> {
        match self {
            Self::Alloc {
                result: _,
                elem_type: _,
            }
            | Self::FreshWitness {
                result: _,
                result_type: _,
            }
            | Self::NextDCoeff { result: _ }
            | Self::Const { .. } => vec![].into_iter(),
            Self::Cmp {
                kind: _,
                result: _,
                lhs: b,
                rhs: c,
            }
            | Self::BinaryArithOp {
                kind: _,
                result: _,
                lhs: b,
                rhs: c,
            }
            | Self::ArrayGet {
                result: _,
                array: b,
                index: c,
            }
            | Self::MulConst {
                result: _,
                const_val: b,
                var: c,
            } => vec![b, c].into_iter(),
            Self::Spread {
                result: _,
                value: v,
            } => vec![v].into_iter(),
            Self::Unspread {
                result_odd: _,
                result_even: _,
                value: v,
            } => vec![v].into_iter(),
            Self::ArraySet {
                result: _,
                array: b,
                index: c,
                value: d,
            } => vec![b, c, d].into_iter(),
            Self::SlicePush {
                dir: _,
                result: _,
                slice: b,
                values: c,
            } => {
                let mut ret_vec = vec![b];
                let values_vec: Vec<&mut ValueId> = c.iter_mut().collect();
                ret_vec.extend(values_vec);
                ret_vec.into_iter()
            }
            Self::SliceLen {
                result: _,
                slice: b,
            } => vec![b].into_iter(),
            Self::AssertEq { lhs: b, rhs: c }
            | Self::Store { ptr: b, value: c }
            | Self::BumpD {
                matrix: _,
                variable: b,
                sensitivity: c,
            } => vec![b, c].into_iter(),
            Self::Load { result: _, ptr: c }
            | Self::WriteWitness {
                result: _,
                value: c,
                pinned: _,
            }
            | Self::Cast {
                result: _,
                value: c,
                target: _,
            }
            | Self::Truncate {
                result: _,
                value: c,
                to_bits: _,
                from_bits: _,
            }
            | Self::SExt {
                result: _,
                value: c,
                from_bits: _,
                to_bits: _,
            } => vec![c].into_iter(),
            Self::Call {
                results: _,
                function,
                args: a,
                unconstrained: _,
            } => {
                let mut ret_vec = Vec::new();
                if let CallTarget::Dynamic(fn_ptr) = function {
                    ret_vec.push(fn_ptr);
                }
                ret_vec.extend(a.iter_mut());
                ret_vec.into_iter()
            }
            Self::MkSeq {
                result: _,
                elems: inputs,
                seq_type: _,
                elem_type: _,
            } => inputs.iter_mut().collect::<Vec<_>>().into_iter(),
            Self::MkTuple {
                result: _,
                elems: inputs,
                element_types: _,
            } => inputs.iter_mut().collect::<Vec<_>>().into_iter(),
            Self::Select {
                result: _,
                cond: b,
                if_t: c,
                if_f: d,
            }
            | Self::AssertR1C { a: b, b: c, c: d }
            | Self::Constrain { a: b, b: c, c: d } => vec![b, c, d].into_iter(),
            Self::Not {
                result: _,
                value: v,
            }
            | Self::ValueOf {
                result: _,
                value: v,
            } => vec![v].into_iter(),
            Self::ToBits {
                result: _,
                value: v,
                endianness: _,
                count: _,
            } => vec![v].into_iter(),
            Self::ToRadix {
                result: _,
                value: v,
                radix,
                endianness: _,
                count: _,
            } => {
                let mut ret_vec = vec![v];
                match radix {
                    Radix::Bytes => {}
                    Radix::Dyn(radix) => {
                        ret_vec.push(radix);
                    }
                }
                ret_vec.into_iter()
            }
            Self::MemOp { kind: _, value: v } => vec![v].into_iter(),
            Self::Rangecheck {
                value: val,
                max_bits: _,
            } => vec![val].into_iter(),
            Self::ReadGlobal {
                result: _,
                offset: _,
                result_type: _,
            } => vec![].into_iter(),
            Self::Lookup {
                target,
                keys,
                results,
                flag,
            }
            | Self::DLookup {
                target,
                keys,
                results,
                flag,
            } => {
                let mut ret_vec = vec![];
                match target {
                    LookupTarget::Rangecheck(_) | LookupTarget::Spread(_) => {}
                    LookupTarget::DynRangecheck(v) => {
                        ret_vec.push(v);
                    }
                    LookupTarget::Array(arr) => {
                        ret_vec.push(arr);
                    }
                }
                ret_vec.extend(keys);
                ret_vec.extend(results);
                ret_vec.push(flag);
                ret_vec.into_iter()
            }
            Self::TupleProj {
                result: _,
                tuple,
                idx: _,
            } => vec![tuple].into_iter(),
            Self::Todo { .. } => vec![].into_iter(),
            Self::InitGlobal {
                global: _,
                value: v,
            } => vec![v].into_iter(),
            Self::DropGlobal { global: _ } => vec![].into_iter(),
            Self::Guard { condition, inner } => {
                let mut ret_vec = vec![condition];
                ret_vec.extend(inner.get_inputs_mut());
                ret_vec.into_iter()
            }
        }
    }

    fn get_operands_mut(&mut self) -> impl Iterator<Item = &mut ValueId> {
        match self {
            Self::Alloc {
                result: r,
                elem_type: _,
            }
            | Self::MemOp { kind: _, value: r }
            | Self::FreshWitness {
                result: r,
                result_type: _,
            }
            | Self::NextDCoeff { result: r }
            | Self::Const { result: r, .. } => vec![r].into_iter(),
            Self::Cmp {
                kind: _,
                result: a,
                lhs: b,
                rhs: c,
            }
            | Self::BinaryArithOp {
                kind: _,
                result: a,
                lhs: b,
                rhs: c,
            }
            | Self::ArrayGet {
                result: a,
                array: b,
                index: c,
            }
            | Self::MulConst {
                result: a,
                const_val: b,
                var: c,
            } => vec![a, b, c].into_iter(),
            Self::Cast {
                result: a,
                value: b,
                target: _,
            } => vec![a, b].into_iter(),
            Self::Truncate {
                result: a,
                value: b,
                to_bits: _,
                from_bits: _,
            } => vec![a, b].into_iter(),
            Self::SExt {
                result: a,
                value: b,
                from_bits: _,
                to_bits: _,
            } => vec![a, b].into_iter(),
            Self::ArraySet {
                result: a,
                array: b,
                index: c,
                value: d,
            } => vec![a, b, c, d].into_iter(),
            Self::SlicePush {
                dir: _,
                result: a,
                slice: b,
                values: c,
            } => {
                let mut ret_vec = vec![a, b];
                let values_vec = c.iter_mut().collect::<Vec<_>>();
                ret_vec.extend(values_vec);
                ret_vec.into_iter()
            }
            Self::SliceLen {
                result: a,
                slice: b,
            } => vec![a, b].into_iter(),
            Self::AssertR1C { a, b, c } | Self::Constrain { a, b, c } => vec![a, b, c].into_iter(),
            Self::Store { ptr: a, value: b }
            | Self::Load { result: a, ptr: b }
            | Self::AssertEq { lhs: a, rhs: b }
            | Self::BumpD {
                matrix: _,
                variable: a,
                sensitivity: b,
            } => vec![a, b].into_iter(),
            Self::WriteWitness {
                result: a,
                value: b,
                pinned: _,
            } => {
                let mut ret_vec = a.iter_mut().collect::<Vec<_>>();
                ret_vec.push(b);
                ret_vec.into_iter()
            }
            Self::Call {
                results: r,
                function,
                args: a,
                unconstrained: _,
            } => {
                let mut ret_vec = r.iter_mut().collect::<Vec<_>>();
                if let CallTarget::Dynamic(fn_ptr) = function {
                    ret_vec.push(fn_ptr);
                }
                let args_vec = a.iter_mut().collect::<Vec<_>>();
                ret_vec.extend(args_vec);
                ret_vec.into_iter()
            }
            Self::Lookup {
                target,
                keys,
                results,
                flag,
            }
            | Self::DLookup {
                target,
                keys,
                results,
                flag,
            } => {
                let mut ret_vec = vec![];
                match target {
                    LookupTarget::Rangecheck(_) | LookupTarget::Spread(_) => {}
                    LookupTarget::DynRangecheck(v) => {
                        ret_vec.push(v);
                    }
                    LookupTarget::Array(arr) => {
                        ret_vec.push(arr);
                    }
                }
                ret_vec.extend(keys);
                ret_vec.extend(results);
                ret_vec.push(flag);
                ret_vec.into_iter()
            }
            Self::MkSeq {
                result: r,
                elems: inputs,
                seq_type: _,
                elem_type: _,
            } => {
                let mut ret_vec = vec![r];
                ret_vec.extend(inputs);
                ret_vec.into_iter()
            }
            Self::Select {
                result: a,
                cond: b,
                if_t: c,
                if_f: d,
            } => vec![a, b, c, d].into_iter(),
            Self::Not {
                result: r,
                value: v,
            }
            | Self::ValueOf {
                result: r,
                value: v,
            }
            | Self::Spread {
                result: r,
                value: v,
            } => vec![r, v].into_iter(),
            Self::Unspread {
                result_odd: a,
                result_even: b,
                value: v,
            } => vec![a, b, v].into_iter(),
            Self::ToBits {
                result: r,
                value: v,
                endianness: _,
                count: _,
            } => vec![r, v].into_iter(),
            Self::ToRadix {
                result: r,
                value: v,
                radix,
                endianness: _,
                count: _,
            } => {
                let mut ret_vec = vec![r, v];
                match radix {
                    Radix::Bytes => {}
                    Radix::Dyn(radix) => {
                        ret_vec.push(radix);
                    }
                }
                ret_vec.into_iter()
            }
            Self::Rangecheck {
                value: val,
                max_bits: _,
            } => vec![val].into_iter(),
            Self::ReadGlobal {
                result: r,
                offset: _,
                result_type: _,
            } => vec![r].into_iter(),
            Self::TupleProj {
                result: r,
                tuple: t,
                idx: _,
            } => vec![r, t].into_iter(),
            OpCode::MkTuple {
                result: r,
                elems: e,
                element_types: _,
            } => {
                let mut ret_vec = vec![r];
                ret_vec.extend(e);
                ret_vec.into_iter()
            }
            Self::Todo { results, .. } => {
                let ret_vec: Vec<&mut ValueId> = results.iter_mut().collect();
                ret_vec.into_iter()
            }
            Self::InitGlobal {
                global: _,
                value: v,
            } => vec![v].into_iter(),
            Self::DropGlobal { global: _ } => vec![].into_iter(),
            Self::Guard { condition, inner } => {
                let mut ret_vec = vec![condition];
                ret_vec.extend(inner.get_operands_mut());
                ret_vec.into_iter()
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum Terminator {
    Jmp(BlockId, Vec<ValueId>),
    JmpIf(ValueId, BlockId, BlockId),
    Return(Vec<ValueId>),
}

impl Terminator {
    pub fn to_string(&self) -> String {
        match self {
            Terminator::Jmp(block_id, args) => {
                let args_str = args.iter().map(|v| format!("v{}", v.0)).join(", ");
                format!("jmp block_{}({})", block_id.0, args_str)
            }
            Terminator::JmpIf(cond, true_block, false_block) => {
                format!(
                    "jmp_if v{} to block_{}, else to block_{}",
                    cond.0, true_block.0, false_block.0
                )
            }
            Terminator::Return(values) => {
                let values_str = values.iter().map(|v| format!("v{}", v.0)).join(", ");
                format!("return {}", values_str)
            }
        }
    }
}

/// Compute spread of a byte/value: interleave zero bits between each bit.
/// E.g., spread(0b1011) = 0b01_00_01_01
pub fn spread_u64(v: u64) -> u64 {
    let mut result = 0u64;
    for i in 0..32 {
        if v & (1 << i) != 0 {
            result |= 1 << (2 * i);
        }
    }
    result
}

/// Extract even bits and odd bits from a spread sum.
/// Returns (odd_bits, even_bits).
pub fn unspread_u64(v: u64) -> (u64, u64) {
    let mut odd_val = 0u64;
    let mut even_val = 0u64;
    for i in 0..32 {
        if v & (1 << (2 * i)) != 0 {
            even_val |= 1 << i;
        }
        if v & (1 << (2 * i + 1)) != 0 {
            odd_val |= 1 << i;
        }
    }
    (odd_val, even_val)
}
