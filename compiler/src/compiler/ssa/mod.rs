//! The definition for the generic SSA structure used as the backend for both the HLSSA and LLSSA
//! variants used by the compiler.

pub mod builder;
pub mod hlssa;
pub mod hlssa_to_llssa;
pub mod id;
pub mod llssa;
pub mod traits;

use bimap::BiHashMap;
use itertools::Itertools;
use std::{
    fmt::Debug,
    hash::Hash,
    sync::{
        Arc, RwLock,
        atomic::{AtomicU64, Ordering},
    },
    vec,
};

use crate::collections::HashMap;

pub use super::located::{Located, Location, SourceLocation, SourcePosition};
pub use id::{BlockId, FunctionId, ValueId};
pub use traits::{Instruction, SSAAnotator, SSAType};

// TYPE ALIASES
// ================================================================================================

/// Storage for constants inside the SSA, maintaining a 1:1 mapping between value and its
/// identifier.
///
/// Wrapped in a [`RwLock`] so constants can be interned and read through a shared `&SSA`, and the
/// values are `Arc`-shared so reads can hand back owned handles without keeping the lock held. The
/// lock is an implementation detail: all access goes through [`SSA`] methods that acquire and
/// release it internally, so callers never see a guard.
pub type SSAConstants<C> = RwLock<
    BiHashMap<
        ValueId,
        Arc<C>,
        crate::collections::FxBuildHasher,
        crate::collections::FxBuildHasher,
    >,
>;

/// An owned, lock-free snapshot of the constants (the `ValueId -> value` direction only), handed
/// out by [`SSA::const_snapshot`] so callers can iterate and look up constants without holding the
/// constants lock.
pub type SSAConstantsSnapshot<C> = HashMap<ValueId, Arc<C>>;

// SSA
// ================================================================================================

/// The SSA structure used by the Mavros compiler, providing a generic IR that can be tailored with
/// custom instructions and types.
///
/// - `Op` is the type of instructions in the SSA, allowing customization of the instruction set
///   over which the IR is operating.
/// - `Ty` is the type system for the SSA, describing the valid types and their interactions.
/// - `C` is the value type stored in the SSA's constants side-table, which maps each constant
///   `ValueId` to its value bidirectionally.
///
/// It is assumed that all `ValueId`s refer to unique values within the program. If one identifier
/// is used to refer to two different values, then miscompilation may occur. All sources of value
/// identifiers should either re-use identifiers known to be unique, or be issued (however
/// transitively) from [`SSA::fresh_value`].
pub struct SSA<Op: Instruction, Ty: SSAType, C: Clone + Debug + Eq + Hash> {
    /// A mapping from function identifiers to true functions contained in the SSA.
    functions: HashMap<FunctionId, Function<Op, Ty>>,

    /// The type of each global value, with the index in the vector corresponding to the global's identifier.
    global_types: Vec<Ty>,

    /// The function used to initialize the global values.
    globals_init_fn: Option<FunctionId>,

    /// The function used to de-initialize/drop the global values.
    globals_deinit_fn: Option<FunctionId>,

    /// The identifiers of the program's entry points. Each one serves as a reachability root and
    /// is emitted as a separately callable entry in the generated artifacts.
    entry_points: Vec<FunctionId>,

    /// A monotonic counter for function identifiers, used to ensure uniqueness.
    next_function_id: u64,

    /// A monotonic counter for `ValueId`s, globally unique within this SSA.
    ///
    /// Atomic so passes can mint fresh IDs from a shared `&SSA` (e.g. in the specializer) without
    /// needing exclusive access.
    next_value_id: AtomicU64,

    /// Bidirectional mapping between constant `ValueId`s and their values. The bijection is
    /// maintained by routing all insertions through [`SSA::add_const`], which interns by value.
    constants: SSAConstants<C>,
}

impl<Op: Instruction, Ty: SSAType, C: Clone + Debug + Eq + Hash> Clone for SSA<Op, Ty, C> {
    /// Take care when cloning the SSA as both original and clone will have the same state for the
    /// fresh variable allocation.
    fn clone(&self) -> Self {
        SSA {
            functions: self.functions.clone(),
            global_types: self.global_types.clone(),
            globals_init_fn: self.globals_init_fn,
            globals_deinit_fn: self.globals_deinit_fn,
            entry_points: self.entry_points.clone(),
            next_function_id: self.next_function_id,
            next_value_id: AtomicU64::new(self.next_value_id.load(Ordering::Relaxed)),
            constants: RwLock::new(self.constants.read().unwrap().clone()),
        }
    }
}

impl<Op: Instruction, Ty: SSAType, C: Clone + Debug + Eq + Hash> SSA<Op, Ty, C> {
    /// Constructs an empty SSA: no functions and no entry points. Functions are added with
    /// [`Self::add_function`] / [`Self::insert_function`] and registered as entry points with
    /// [`Self::add_entry_point`].
    pub fn empty() -> Self {
        SSA {
            functions: HashMap::default(),
            global_types: Vec::new(),
            globals_init_fn: None,
            globals_deinit_fn: None,
            entry_points: Vec::new(),
            next_function_id: 0,
            next_value_id: AtomicU64::new(0),
            constants: RwLock::new(BiHashMap::default()),
        }
    }

    pub fn with_main(name: String) -> Self {
        let mut ssa = Self::empty();
        let main_id = ssa.add_function(name);
        ssa.add_entry_point(main_id);
        ssa
    }
}

impl<Op: Instruction, Ty: SSAType, C: Clone + Debug + Eq + Hash> SSA<Op, Ty, C> {
    pub fn prepare_rebuild(
        self,
    ) -> (
        SSA<Op, Ty, C>,
        HashMap<FunctionId, Function<Op, Ty>>,
        Vec<Ty>,
    ) {
        (
            SSA {
                functions: HashMap::default(),
                global_types: Vec::new(),
                globals_init_fn: self.globals_init_fn,
                globals_deinit_fn: self.globals_deinit_fn,
                entry_points: self.entry_points.clone(),
                next_function_id: self.next_function_id,
                next_value_id: self.next_value_id,
                constants: self.constants,
            },
            self.functions,
            self.global_types,
        )
    }

    /// Inserts the provided `function` into the SSA.
    pub fn insert_function(&mut self, function: Function<Op, Ty>) -> FunctionId {
        let new_id = FunctionId(self.next_function_id);
        self.next_function_id += 1;
        self.functions.insert(new_id, function);
        new_id
    }

    /// Remove a function from the SSA, returning it if present.
    pub fn delete_function(&mut self, id: FunctionId) -> Option<Function<Op, Ty>> {
        self.functions.remove(&id)
    }

    /// Drop functions for which `f` returns `false`.
    pub fn retain_functions(&mut self, mut f: impl FnMut(FunctionId, &Function<Op, Ty>) -> bool) {
        self.functions.retain(|id, fun| f(*id, fun));
    }

    /// Replaces the program's sole entry point. Panics unless there is exactly one entry point —
    /// callers that work with a multi-entry program must go through [`Self::get_entry_points`].
    pub fn set_unique_entrypoint(&mut self, id: FunctionId) {
        assert_eq!(
            self.entry_points.len(),
            1,
            "set_unique_entrypoint called on an SSA with {} entry points",
            self.entry_points.len()
        );
        self.entry_points[0] = id;
    }

    /// Registers an additional entry point for the program.
    pub fn add_entry_point(&mut self, id: FunctionId) {
        assert!(
            !self.entry_points.contains(&id),
            "Function {id:?} is already an entry point"
        );
        self.entry_points.push(id);
    }

    /// All entry points of the program.
    pub fn get_entry_points(&self) -> &[FunctionId] {
        &self.entry_points
    }

    /// Whether `id` names one of the program's entry points.
    pub fn is_entry_point(&self, id: FunctionId) -> bool {
        self.entry_points.contains(&id)
    }

    /// The program's sole entry point. Panics unless there is exactly one entry point — callers
    /// that work with a multi-entry program must go through [`Self::get_entry_points`].
    pub fn get_unique_entrypoint_id(&self) -> FunctionId {
        assert_eq!(
            self.entry_points.len(),
            1,
            "get_unique_entrypoint_id called on an SSA with {} entry points",
            self.entry_points.len()
        );
        self.entry_points[0]
    }

    /// Gets a mutable reference to the program's sole entry point function. Panics unless there is
    /// exactly one entry point.
    pub fn get_unique_entrypoint_mut(&mut self) -> &mut Function<Op, Ty> {
        self.functions
            .get_mut(&self.get_unique_entrypoint_id())
            .expect("Entry point function should exist")
    }

    /// Gets a reference to the program's sole entry point function. Panics unless there is exactly
    /// one entry point.
    pub fn get_unique_entrypoint(&self) -> &Function<Op, Ty> {
        self.functions
            .get(&self.get_unique_entrypoint_id())
            .expect("Entry point function should exist")
    }

    pub fn get_function(&self, id: FunctionId) -> &Function<Op, Ty> {
        self.functions.get(&id).expect("Function should exist")
    }

    /// Gets a mutable reference to a function in the SSA or panics if no such function exists.
    pub fn get_function_mut(&mut self, id: FunctionId) -> &mut Function<Op, Ty> {
        self.functions.get_mut(&id).expect("Function should exist")
    }

    /// Removes the provided function from the SSA or panics if it does not exist.
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

    /// Creates a unique copy of the function identified by `function_id` under the returned
    /// function identifier.
    ///
    /// The copy has been inserted into the SSA, and while it contains the same block identifiers,
    /// all value identifiers barring those for constants are now unique to the new occurrences.
    pub fn duplicate_function(&mut self, function_id: FunctionId) -> FunctionId {
        self.duplicate_function_with_remap(function_id).0
    }

    /// Like [`Self::duplicate_function`], but also returns the `old ValueId -> new ValueId` remap.
    ///
    /// Block ids are preserved; only non-constant value ids are freshened. Callers that need to
    /// translate per-value analysis results from the original onto the clone use the remap.
    pub fn duplicate_function_with_remap(
        &mut self,
        function_id: FunctionId,
    ) -> (FunctionId, HashMap<ValueId, ValueId>) {
        let mut cloned = self
            .functions
            .get(&function_id)
            .expect("Function should exist")
            .clone();

        // Phase 1: collect every ValueId appearing anywhere in the function, then
        // allocate a fresh id for each unique non-constant one.
        let mut all_ids: Vec<ValueId> = Vec::new();
        for (_, block) in cloned.get_blocks() {
            for (v, _) in block.get_parameters() {
                all_ids.push(*v);
            }
            for instr in block.get_instructions() {
                all_ids.extend(instr.get_inputs().copied());
                all_ids.extend(instr.get_results().copied());
            }
            if let Some(term) = block.get_terminator() {
                match term {
                    Terminator::Jmp(_, args) => all_ids.extend(args.iter().copied()),
                    Terminator::JmpIf(cond, _, _) => all_ids.push(*cond),
                    Terminator::Return(vs) => all_ids.extend(vs.iter().copied()),
                }
            }
        }
        let mut remap: HashMap<ValueId, ValueId> = HashMap::default();
        for v in all_ids {
            if remap.contains_key(&v) || self.is_const(v) {
                continue;
            }
            let fresh = self.fresh_value();
            remap.insert(v, fresh);
        }

        // Phase 2: apply the remap to every ValueId reference in the cloned function.
        for (_, block) in cloned.get_blocks_mut() {
            for (v, _) in block.get_parameters_mut() {
                if let Some(&new_id) = remap.get(v) {
                    *v = new_id;
                }
            }
            for instr in block.get_instructions_mut() {
                for op in instr.get_operands_mut() {
                    if let Some(&new_id) = remap.get(op) {
                        *op = new_id;
                    }
                }
            }
            if block.get_terminator().is_some() {
                match block.get_terminator_mut() {
                    Terminator::Jmp(_, args) => {
                        for v in args.iter_mut() {
                            if let Some(&new_id) = remap.get(v) {
                                *v = new_id;
                            }
                        }
                    }
                    Terminator::JmpIf(cond, _, _) => {
                        if let Some(&new_id) = remap.get(cond) {
                            *cond = new_id;
                        }
                    }
                    Terminator::Return(vs) => {
                        for v in vs.iter_mut() {
                            if let Some(&new_id) = remap.get(v) {
                                *v = new_id;
                            }
                        }
                    }
                }
            }
        }

        (self.insert_function(cloned), remap)
    }

    /// Merges another SSA into this one, returning the `other FunctionId -> new FunctionId`
    /// remap.
    ///
    /// Every function of `other` is inserted under a fresh `FunctionId`, with all of its
    /// non-constant `ValueId`s freshened (the two SSAs may descend from a common ancestor, so
    /// their value identifiers can collide) and its constants re-interned by value into this
    /// SSA's constants table. Static call targets are rewritten to the remapped identifiers.
    ///
    /// Both SSAs must agree on the global value layout: globals are shared state addressed by
    /// offset, so diverging global types would silently corrupt the global frame.
    ///
    /// `other`'s entry points and globals init/deinit functions are NOT transferred; the caller
    /// decides which of the merged functions (if any) become entry points via
    /// [`SSA::add_entry_point`].
    pub fn merge(&mut self, other: Self) -> HashMap<FunctionId, FunctionId> {
        assert_eq!(
            self.global_types, other.global_types,
            "Cannot merge SSAs with diverging global types"
        );

        let mut other_fn_ids: Vec<FunctionId> = other.functions.keys().copied().collect();
        other_fn_ids.sort_by_key(|id| id.0);

        let mut fn_remap: HashMap<FunctionId, FunctionId> = HashMap::default();
        for old_id in &other_fn_ids {
            let new_id = FunctionId(self.next_function_id);
            self.next_function_id += 1;
            fn_remap.insert(*old_id, new_id);
        }

        // Re-intern all of `other`'s constants by value; identical values collapse onto this
        // SSA's existing ids. Sorted for deterministic fresh-id allocation.
        let mut value_remap: HashMap<ValueId, ValueId> = HashMap::default();
        let other_constants = other.const_snapshot();
        let mut const_ids: Vec<ValueId> = other_constants.keys().copied().collect();
        const_ids.sort_by_key(|v| v.0);
        for vid in const_ids {
            let new_id = self.add_const(other_constants[&vid].as_ref().clone());
            value_remap.insert(vid, new_id);
        }

        let mut other = other;
        for old_id in other_fn_ids {
            let mut function = other.take_function(old_id);

            // Freshen every non-constant ValueId. Walk blocks sorted by id so fresh-id
            // allocation (and thus the emitted program) is deterministic.
            let mut block_ids: Vec<BlockId> = function.blocks.keys().copied().collect();
            block_ids.sort_by_key(|b| b.0);
            for block_id in &block_ids {
                let block = function.get_block(*block_id);
                let mut ids: Vec<ValueId> = Vec::new();
                for (v, _) in block.get_parameters() {
                    ids.push(*v);
                }
                for instr in block.get_instructions() {
                    ids.extend(instr.get_inputs().copied());
                    ids.extend(instr.get_results().copied());
                }
                if let Some(term) = block.get_terminator() {
                    match term {
                        Terminator::Jmp(_, args) | Terminator::Return(args) => {
                            ids.extend(args.iter().copied())
                        }
                        Terminator::JmpIf(cond, _, _) => ids.push(*cond),
                    }
                }
                for v in ids {
                    if !value_remap.contains_key(&v) {
                        value_remap.insert(v, self.fresh_value());
                    }
                }
            }

            // Apply the value remap and rewrite call targets to the new function ids.
            for (_, block) in function.get_blocks_mut() {
                for (v, _) in block.get_parameters_mut() {
                    *v = value_remap[v];
                }
                for instr in block.get_instructions_mut() {
                    for op in instr.get_operands_mut() {
                        *op = value_remap[op];
                    }
                    instr.map_call_targets(&mut |callee| {
                        *fn_remap
                            .get(&callee)
                            .expect("Merged function calls a function missing from the merged SSA")
                    });
                }
                if block.get_terminator().is_some() {
                    match block.get_terminator_mut() {
                        Terminator::Jmp(_, args) | Terminator::Return(args) => {
                            for v in args.iter_mut() {
                                *v = value_remap[v];
                            }
                        }
                        Terminator::JmpIf(cond, _, _) => {
                            *cond = value_remap[cond];
                        }
                    }
                }
            }

            self.functions.insert(fn_remap[&old_id], function);
        }

        fn_remap
    }

    /// Allocate a fresh `ValueId` from the SSA-wide counter.
    ///
    /// Takes `&self` so passes that hold a shared `&HLSSA` (e.g. the specializer, while the
    /// symbolic executor borrows the SSA) can still mint ids. The counter is atomic.
    ///
    /// For minting a known number of values, see [`Self::fresh_values`].
    pub fn fresh_value(&self) -> ValueId {
        ValueId(self.next_value_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Allocate `n` fresh `ValueId`s from the SSA-wide counter.
    ///
    /// Takes `&self` so passes that hold a shared `&HLSSA` (e.g. the specializer, while the
    /// symbolic executor borrows the SSA) can still mint ids. The counter is atomic.
    ///
    /// For minting single values, see [`Self::fresh_value`].
    pub fn fresh_values(&self, n: usize) -> Vec<ValueId> {
        (0..n).map(|_| self.fresh_value()).collect()
    }

    /// Upper bound on `ValueId`s issued so far. Useful for sizing dense per-value tables.
    pub fn value_num_bound(&self) -> usize {
        self.next_value_id.load(Ordering::Relaxed) as usize
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

    /// Return a shared handle to the constant bound to `vid`, if any.
    ///
    /// The constants lock is acquired and released internally, so the caller never holds it.
    pub fn get_const(&self, vid: ValueId) -> Option<Arc<C>> {
        self.constants.read().unwrap().get_by_left(&vid).cloned()
    }

    /// Whether `vid` names a constant.
    pub fn is_const(&self, vid: ValueId) -> bool {
        self.constants.read().unwrap().contains_left(&vid)
    }

    /// Take an owned, lock-free snapshot of the constants (`ValueId -> value`).
    ///
    /// The values are `Arc`-shared, so cloning the table is cheap. The lock is released before
    /// returning, letting callers iterate and look up constants without holding it.
    pub fn const_snapshot(&self) -> SSAConstantsSnapshot<C> {
        self.constants
            .read()
            .unwrap()
            .iter()
            .map(|(vid, cv)| (*vid, cv.clone()))
            .collect()
    }

    /// Store a constant in the SSA.
    ///
    /// If `value` is already present in the constants table, returns its existing `ValueId`;
    /// otherwise allocates a fresh `ValueId`, inserts the pair, and returns it. This is the only
    /// safe way to introduce a constant. Takes `&self` so constants can be interned through a
    /// shared `&SSA`.
    pub fn add_const(&self, value: C) -> ValueId {
        // Fast path: re-interning an existing constant is by far the common case
        // (`materialize_constants` re-emits the whole table on every function entry). Resolve those
        // hits under a shared read lock so they don't serialize on the write lock.
        if let Some(&id) = self.constants.read().unwrap().get_by_right(&value) {
            return id;
        }

        // Slow path: a genuine miss. Take the write lock and re-check, since another writer may have
        // inserted between releasing the read lock and acquiring the write lock. Holding the write
        // lock across the re-check and the insert keeps them atomic; an intervening reader/writer
        // cannot observe a half-updated table.
        let mut constants = self.constants.write().unwrap();
        if let Some(&id) = constants.get_by_right(&value) {
            return id;
        }
        let id = self.fresh_value();
        constants.insert(id, Arc::new(value));
        id
    }

    /// Remove the constant bound to `vid` from the table, returning its value if present and
    /// panicking if not.
    pub fn remove_const_by_id(&self, vid: ValueId) -> Arc<C> {
        self.constants
            .write()
            .unwrap()
            .remove_by_left(&vid)
            .map(|(_, v)| v)
            .expect("Constant should exist")
    }

    /// Drop constants for which `f` returns `false`.
    pub fn retain_constants(&self, mut f: impl FnMut(&ValueId, &Arc<C>) -> bool) {
        self.constants.write().unwrap().retain(|vid, cv| f(vid, cv));
    }

    /// Visit every constant in place, without cloning the table.
    ///
    /// Holds the constants read lock for the duration of the walk and hands each
    /// `(&ValueId, &Arc<C>)` to `f`. Prefer this over [`SSA::const_snapshot`] when you only need a
    /// single pass and do not need to keep the constants around afterwards.
    ///
    /// `f` must not call any method that mutates the constants table (e.g. [`SSA::add_const`],
    /// [`SSA::retain_constants`]); doing so would deadlock on the read lock held here.
    pub fn for_each_const(&self, mut f: impl FnMut(&ValueId, &Arc<C>)) {
        for (vid, cv) in self.constants.read().unwrap().iter() {
            f(vid, cv);
        }
    }
}

impl<Op: Instruction, Ty: SSAType, C: Clone + Debug + Eq + Hash> SSA<Op, Ty, C> {
    pub fn to_string(&self, value_annotator: &dyn SSAAnotator) -> String {
        let func_name = |id: FunctionId| self.get_function(id).get_name().to_string();
        let functions = self
            .functions
            .iter()
            .sorted_by_key(|(fn_id, _)| fn_id.0)
            .map(|(fn_id, func)| func.to_string(&func_name, *fn_id, value_annotator))
            .join("\n\n");
        let const_guard = self.constants.read().unwrap();
        if const_guard.is_empty() {
            functions
        } else {
            let constants = const_guard
                .iter()
                .sorted_by_key(|(vid, _)| vid.0)
                .map(|(vid, cv)| format!("  v{} = {:?}", vid.0, cv))
                .join("\n");
            format!("constants:\n{}\n\n{}", constants, functions)
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
        let mut blocks = HashMap::default();
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
                blocks: HashMap::default(),
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

    pub fn next_block_id_bound(&self) -> u64 {
        self.next_block
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
    instructions: Vec<Located<Op>>,
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
            .map(|i| {
                let instruction = i.display_instruction(func_name, &annotate_value);
                match i.get_location() {
                    Some(source_location) => format!("    {} @ {}", instruction, source_location),
                    None => format!("    {}", instruction),
                }
            })
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

    // TODO: Once locations become non-optional, make the internal representation store
    // `SourceLocation` directly.
    pub fn take_instructions(&mut self) -> Vec<Located<Op>> {
        std::mem::take(&mut self.instructions)
    }

    pub fn put_instructions(&mut self, instructions: Vec<impl Into<Located<Op>>>) {
        self.instructions = instructions
            .into_iter()
            .map(|instruction| instruction.into())
            .collect();
    }

    pub fn push_instruction(&mut self, instruction: impl Into<Located<Op>>) {
        self.instructions.push(instruction.into());
    }

    pub fn push_instruction_with_source_location(
        &mut self,
        instruction: Op,
        source_location: SourceLocation,
    ) {
        self.instructions
            .push(Located::with(instruction, source_location));
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

    pub fn get_instruction(&self, i: usize) -> &Located<Op> {
        &self.instructions[i]
    }

    pub fn get_instruction_source_location(&self, i: usize) -> Option<&SourceLocation> {
        self.instructions[i].get_location()
    }

    pub fn set_instruction_source_location(
        &mut self,
        i: usize,
        source_location: Option<SourceLocation>,
    ) {
        *self.instructions[i].location_mut() = source_location;
    }

    pub fn instruction_count(&self) -> usize {
        self.instructions.len()
    }

    pub fn stamp_source_location_from(
        &mut self,
        start: usize,
        source_location: Option<SourceLocation>,
    ) {
        let Some(source_location) = source_location else {
            return;
        };

        for instruction in self.instructions.iter_mut().skip(start) {
            if instruction.get_location().is_none() {
                *instruction.location_mut() = Some(source_location.clone());
            }
        }
    }

    pub fn get_instructions(&self) -> impl DoubleEndedIterator<Item = &Op> {
        self.instructions.iter().map(|instruction| &**instruction)
    }

    pub fn get_instructions_with_source_locations(
        &self,
    ) -> impl DoubleEndedIterator<Item = (&Op, Option<&SourceLocation>)> {
        self.instructions
            .iter()
            .map(|instruction| (&**instruction, instruction.get_location()))
    }

    pub fn get_instructions_mut(&mut self) -> impl Iterator<Item = &mut Op> {
        self.instructions
            .iter_mut()
            .map(|instruction| &mut **instruction)
    }

    pub fn get_instruction_source_locations_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut Option<SourceLocation>> {
        self.instructions
            .iter_mut()
            .map(|instruction| instruction.location_mut())
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

#[cfg(test)]
mod tests {
    use crate::collections::HashMap;

    use crate::compiler::ssa::{
        DefaultSSAAnnotator, Located, SourceLocation, SourcePosition, ValueId,
        hlssa::{Constant, HLSSA, OpCode},
    };

    fn test_location() -> SourceLocation {
        SourceLocation::new(
            "src/main.nr".to_string(),
            SourcePosition::new(3, 5),
            SourcePosition::new(3, 10),
        )
    }

    /// `for_each_const` visits every interned constant exactly once, in place.
    #[test]
    fn for_each_const_visits_all() {
        let ssa = HLSSA::with_main("main".to_string());
        let a = ssa.add_const(Constant::U(8, 1));
        let b = ssa.add_const(Constant::U(8, 2));

        let mut seen = HashMap::default();
        ssa.for_each_const(|vid, cv| {
            seen.insert(*vid, cv.as_ref().clone());
        });

        assert_eq!(seen.len(), 2);
        assert_eq!(seen.get(&a), Some(&Constant::U(8, 1)));
        assert_eq!(seen.get(&b), Some(&Constant::U(8, 2)));
    }

    #[test]
    fn block_instructions_default_to_no_source_location() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let entry = ssa.get_unique_entrypoint_mut().get_entry_mut();

        entry.push_instruction(OpCode::Not {
            result: ValueId(0),
            value: ValueId(1),
        });

        assert_eq!(entry.get_instruction_source_location(0), None);
        assert_eq!(
            entry
                .get_instructions_with_source_locations()
                .map(|(_, location)| location)
                .collect::<Vec<_>>(),
            vec![None]
        );
    }

    #[test]
    fn located_exposes_location_ref_and_take() {
        let location = test_location();
        let mut located = Located::with(ValueId(1), location.clone());

        assert_eq!(located.location(), &Some(location.clone()));
        assert_eq!(located.get_location(), Some(&location));
        *located.location_mut() = Some(location.clone());
        assert_eq!(AsRef::<ValueId>::as_ref(&located), &ValueId(1));

        let located_ref = located.to_ref();
        assert_eq!(*located_ref, &ValueId(1));
        assert_eq!(located_ref.get_location(), Some(&location));

        assert_eq!(located.take(), (ValueId(1), Some(location)));
    }

    #[test]
    fn block_can_store_source_location_for_instruction() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let location = test_location();
        let entry = ssa.get_unique_entrypoint_mut().get_entry_mut();

        entry.push_instruction_with_source_location(
            OpCode::Not {
                result: ValueId(0),
                value: ValueId(1),
            },
            location.clone(),
        );

        assert_eq!(entry.get_instruction_source_location(0), Some(&location));
        assert!(
            ssa.to_string(&DefaultSSAAnnotator)
                .contains("@ src/main.nr:3:5")
        );
    }

    #[test]
    fn raw_instruction_replacement_clears_source_locations() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let entry = ssa.get_unique_entrypoint_mut().get_entry_mut();

        entry.push_instruction_with_source_location(
            OpCode::Not {
                result: ValueId(0),
                value: ValueId(1),
            },
            test_location(),
        );

        let instructions = entry
            .take_instructions()
            .into_iter()
            .map(|instruction| instruction.payload())
            .collect();
        entry.put_instructions(instructions);

        assert_eq!(entry.get_instruction_source_location(0), None);
    }

    #[test]
    fn instruction_replacement_preserves_source_locations() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let location = test_location();
        let entry = ssa.get_unique_entrypoint_mut().get_entry_mut();

        entry.push_instruction(Located::with(
            OpCode::Not {
                result: ValueId(0),
                value: ValueId(1),
            },
            location.clone(),
        ));

        let instructions = entry.take_instructions();
        entry.put_instructions(instructions);

        assert_eq!(entry.get_instruction_source_location(0), Some(&location));
    }
}
