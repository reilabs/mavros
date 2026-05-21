//! The definition for the generic SSA structure used as the backend for both the HLSSA and LLSSA
//! variants used by the compiler.

pub mod builder;
pub mod hlssa;
pub mod hlssa_to_llssa;
pub mod id;
pub mod llssa;
pub mod traits;

use itertools::Itertools;
use std::{collections::HashMap, fmt::Debug, vec};

pub use id::{BlockId, FunctionId, ValueId};
pub use traits::{ConstantsDisplay, Instruction, SSAAnotator, SSAType};

// SSA
// ================================================================================================

/// The SSA structure used by the Mavros compiler, providing a generic IR that can be tailored with
/// custom instructions and types.
///
/// - `Op` is the type of instructions in the SSA, allowing customization of the instruction set
///   over which the IR is operating.
/// - `Ty` is the type system for the SSA, describing the valid types and their interactions.
/// - `Cn` is the type of the constant storage for the SSA, providing an arbitrary interface.
#[derive(Clone)]
pub struct SSA<Op: Instruction, Ty: SSAType, Cn: Clone + Debug> {
    /// A mapping from function identifiers to true functions contained in the SSA.
    functions: HashMap<FunctionId, Function<Op, Ty>>,

    /// The type of each global value, with the index in the vector corresponding to the global's identifier.
    global_types: Vec<Ty>,

    /// The function used to initialize the global values.
    globals_init_fn: Option<FunctionId>,

    /// The function used to de-initialize/drop the global values.
    globals_deinit_fn: Option<FunctionId>,

    /// The identifier of the main function.
    ///
    /// This may be the true program main as provided by Noir, or the identifier of the synthetic
    /// main created during the compilation process. While the exact details depend on the stage of
    /// the pipeline, this will always point to the entry point.
    main_id: FunctionId,

    /// A monotonic counter for function identifiers, used to ensure uniqueness.
    next_function_id: u64,

    /// A monotonic counter for `ValueId`s, globally unique within this SSA.
    next_value_id: u64,

    /// Side-table holding constant data for the SSA.
    constants: Cn,
}

impl<Op: Instruction, Ty: SSAType, Cn: Clone + Debug> SSA<Op, Ty, Cn> {
    pub fn with_main(name: String, constants: Cn) -> Self {
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
            next_value_id: 0,
            constants,
        }
    }
}

impl<Op: Instruction, Ty: SSAType, Cn: Clone + Debug> SSA<Op, Ty, Cn> {
    pub fn prepare_rebuild(
        self,
    ) -> (
        SSA<Op, Ty, Cn>,
        HashMap<FunctionId, Function<Op, Ty>>,
        Vec<Ty>,
    ) {
        (
            SSA {
                functions: HashMap::new(),
                global_types: Vec::new(),
                globals_init_fn: self.globals_init_fn,
                globals_deinit_fn: self.globals_deinit_fn,
                main_id: self.main_id,
                next_function_id: self.next_function_id,
                next_value_id: self.next_value_id,
                constants: self.constants,
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

    /// Allocate a fresh `ValueId` from the SSA-wide counter.
    pub fn fresh_value(&mut self) -> ValueId {
        let value_id = ValueId(self.next_value_id);
        self.next_value_id += 1;
        value_id
    }

    /// Upper bound on `ValueId`s issued so far. Useful for sizing dense per-value tables.
    pub fn value_num_bound(&self) -> usize {
        self.next_value_id as usize
    }

    /// Advance the value counter to `end` if it is currently lower. Used by passes that
    /// allocate `ValueId`s into a transient buffer (see e.g. the specializer) and want to
    /// reconcile the buffer's IDs back into the SSA.
    pub fn reserve_values_up_to(&mut self, end: u64) {
        if end > self.next_value_id {
            self.next_value_id = end;
        }
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

    pub fn const_storage(&self) -> &Cn {
        &self.constants
    }

    pub fn const_storage_mut(&mut self) -> &mut Cn {
        &mut self.constants
    }
}

impl<Op: Instruction, Ty: SSAType, C: Clone + Debug + ConstantsDisplay> SSA<Op, Ty, C> {
    pub fn to_string(&self, value_annotator: &dyn SSAAnotator) -> String {
        let func_name = |id: FunctionId| self.get_function(id).get_name().to_string();
        let consts = self.constants.display_constants(&func_name);
        let functions = self
            .functions
            .iter()
            .sorted_by_key(|(fn_id, _)| fn_id.0)
            .map(|(fn_id, func)| func.to_string(&func_name, *fn_id, value_annotator))
            .join("\n\n");
        if consts.is_empty() {
            functions
        } else {
            format!("{}\n\n{}", consts, functions)
        }
    }
}

// FUNCTION
// ================================================================================================

/// The generic SSA type representing a function object.
#[derive(Clone)]
pub struct Function<Op: Instruction, Ty: SSAType> {
    entry_block: BlockId,
    blocks: HashMap<BlockId, Block<Op, Ty>>,
    name: String,
    returns: Vec<Ty>,
    next_block: u64,
}

impl<Op: Instruction, Ty: SSAType> Function<Op, Ty> {
    pub fn to_string(
        &self,
        func_name: &dyn Fn(FunctionId) -> String,
        id: FunctionId,
        value_annotator: &dyn SSAAnotator,
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

    pub fn get_blocks_mut(&mut self) -> impl Iterator<Item = (&BlockId, &mut Block<Op, Ty>)> {
        self.blocks.iter_mut()
    }

    pub fn take_blocks(&mut self) -> HashMap<BlockId, Block<Op, Ty>> {
        std::mem::take(&mut self.blocks)
    }

    pub fn put_blocks(&mut self, blocks: HashMap<BlockId, Block<Op, Ty>>) {
        self.blocks = blocks;
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

// BLOCK
// ================================================================================================

/// The generic SSA type representing a block.
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
        value_annotator: &dyn SSAAnotator,
    ) -> String {
        let params = self
            .parameters
            .iter()
            .map(|v| {
                let annotation = value_annotator.annotate_value(func_id, v.0);
                let annotation = if annotation.is_empty() {
                    "".to_string()
                } else {
                    format!(" [{annotation}]")
                };
                format!("v{} : {}{annotation}", v.0.0, v.1)
            })
            .join(", ");
        let annotate_value = |value: ValueId| -> String {
            let annotation = value_annotator.annotate_value(func_id, value);
            if annotation.is_empty() {
                "".to_string()
            } else {
                format!("[{annotation}]")
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

// TERMINATOR
// ================================================================================================

/// The standard terminators for a block in the SSA.
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

// DEFAULT SSA ANNOTATOR
// ================================================================================================

/// The default annotator for the SSA that provides no annotation detail.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct DefaultSSAAnnotator;
impl SSAAnotator for DefaultSSAAnnotator {}
