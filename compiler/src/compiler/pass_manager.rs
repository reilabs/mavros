use std::{
    any::{Any, TypeId},
    fmt::Debug,
    fs,
    hash::Hash,
    path::PathBuf,
};

use crate::{
    collections::HashMap,
    compiler::ssa::{
        DefaultSSAAnnotator, Instruction, SSA, SSAType,
        hlssa::{Constant, OpCode, Type},
    },
};

// ---------------------------------------------------------------------------
// AnalysisId — a Copy handle carrying function pointers
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct AnalysisId {
    type_id: TypeId,
    type_name: &'static str,
    dependencies: fn() -> Vec<AnalysisId>,
    compute_and_store: fn(&dyn Any, &mut AnalysisStore),
}

impl AnalysisId {
    pub fn of<
        A: Analysis<Op, Ty, C>,
        Op: Instruction,
        Ty: SSAType,
        C: Clone + Debug + Eq + Hash + 'static,
    >() -> Self {
        Self {
            type_id: TypeId::of::<A>(),
            type_name: std::any::type_name::<A>(),
            dependencies: <A as Analysis<Op, Ty, C>>::dependencies,
            compute_and_store: |ssa_any, store| {
                if !store.contains_type(TypeId::of::<A>()) {
                    let ssa: &SSA<Op, Ty, C> = ssa_any
                        .downcast_ref::<SSA<Op, Ty, C>>()
                        .expect("AnalysisId::compute_and_store: SSA downcast failed");
                    let val = <A as Analysis<Op, Ty, C>>::compute(ssa, store);
                    let dep_ids = <A as Analysis<Op, Ty, C>>::dependencies()
                        .iter()
                        .map(|d| d.type_id)
                        .collect();
                    store.insert_with_deps::<A>(val, dep_ids);
                }
            },
        }
    }
}

impl PartialEq for AnalysisId {
    fn eq(&self, other: &Self) -> bool {
        self.type_id == other.type_id
    }
}
impl Eq for AnalysisId {}

impl std::hash::Hash for AnalysisId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.type_id.hash(state);
    }
}

impl std::fmt::Debug for AnalysisId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AnalysisId({})", self.type_name)
    }
}

// ---------------------------------------------------------------------------
// Analysis trait
// ---------------------------------------------------------------------------

pub trait Analysis<
    Op: Instruction = OpCode,
    Ty: SSAType = Type,
    C: Clone + Debug + Eq + Hash + 'static = Constant,
>: Any + 'static
{
    fn id() -> AnalysisId
    where
        Self: Sized,
    {
        AnalysisId::of::<Self, Op, Ty, C>()
    }

    fn dependencies() -> Vec<AnalysisId>
    where
        Self: Sized,
    {
        vec![]
    }

    fn compute(ssa: &SSA<Op, Ty, C>, store: &AnalysisStore) -> Self
    where
        Self: Sized;
}

// ---------------------------------------------------------------------------
// AnalysisStore — type-erased cache
// ---------------------------------------------------------------------------

pub struct AnalysisStore {
    data: HashMap<TypeId, Box<dyn Any>>,
    deps: HashMap<TypeId, Vec<TypeId>>,
}

impl AnalysisStore {
    pub fn new() -> Self {
        Self {
            data: HashMap::default(),
            deps: HashMap::default(),
        }
    }

    pub fn get<A: 'static>(&self) -> &A {
        self.data
            .get(&TypeId::of::<A>())
            .unwrap_or_else(|| panic!("Analysis {} not found in store", std::any::type_name::<A>()))
            .downcast_ref::<A>()
            .unwrap()
    }

    pub fn try_get<A: 'static>(&self) -> Option<&A> {
        self.data
            .get(&TypeId::of::<A>())
            .and_then(|b| b.downcast_ref::<A>())
    }

    fn insert_with_deps<A: 'static>(&mut self, val: A, dep_ids: Vec<TypeId>) {
        let tid = TypeId::of::<A>();
        self.data.insert(tid, Box::new(val));
        if !dep_ids.is_empty() {
            self.deps.insert(tid, dep_ids);
        }
    }

    fn contains_type(&self, type_id: TypeId) -> bool {
        self.data.contains_key(&type_id)
    }

    /// Remove everything not preserved, plus transitively invalidate
    /// anything whose dependency was removed.
    pub fn apply_preserved(&mut self, preserved: &[AnalysisId]) {
        let preserved_ids: Vec<TypeId> = preserved.iter().map(|a| a.type_id).collect();
        loop {
            let to_remove: Vec<TypeId> = self
                .data
                .keys()
                .filter(|tid| {
                    if !preserved_ids.contains(tid) {
                        return true;
                    }
                    self.deps
                        .get(tid)
                        .map(|deps| deps.iter().any(|d| !self.data.contains_key(d)))
                        .unwrap_or(false)
                })
                .copied()
                .collect();
            if to_remove.is_empty() {
                break;
            }
            for id in &to_remove {
                self.data.remove(id);
                self.deps.remove(id);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pass trait — parametric
// ---------------------------------------------------------------------------

pub trait Pass<
    Op: Instruction = OpCode,
    Ty: SSAType = Type,
    C: Clone + Debug + Eq + Hash + 'static = Constant,
>
{
    fn name(&self) -> &'static str;
    fn needs(&self) -> Vec<AnalysisId> {
        vec![]
    }
    fn run(&self, ssa: &mut SSA<Op, Ty, C>, store: &AnalysisStore);
    fn preserves(&self) -> Vec<AnalysisId> {
        vec![]
    }
}

// ---------------------------------------------------------------------------
// topo_sort — DFS over AnalysisId::dependencies(), skipping what's in store
// ---------------------------------------------------------------------------

fn topo_sort(roots: Vec<AnalysisId>, store: &AnalysisStore) -> Vec<AnalysisId> {
    let mut result = Vec::new();
    let mut visited = crate::collections::HashSet::default();

    fn visit(
        id: AnalysisId,
        store: &AnalysisStore,
        visited: &mut crate::collections::HashSet<TypeId>,
        result: &mut Vec<AnalysisId>,
    ) {
        if store.contains_type(id.type_id) || !visited.insert(id.type_id) {
            return;
        }
        for dep in (id.dependencies)() {
            visit(dep, store, visited, result);
        }
        result.push(id);
    }

    for root in roots {
        visit(root, store, &mut visited, &mut result);
    }
    result
}

// ---------------------------------------------------------------------------
// PassManager — parametric
// ---------------------------------------------------------------------------

pub struct PassManager<
    Op: Instruction = OpCode,
    Ty: SSAType = Type,
    C: Clone + Debug + Eq + Hash + 'static = Constant,
> {
    passes: Vec<Box<dyn Pass<Op, Ty, C>>>,
    analyses: AnalysisStore,
    draw_cfg: bool,
    debug_output_dir: Option<PathBuf>,
    phase_label: String,
}

impl<Op: Instruction, Ty: SSAType, C: Clone + Debug + Eq + Hash + 'static> PassManager<Op, Ty, C> {
    pub fn new(phase_label: String, draw_cfg: bool, passes: Vec<Box<dyn Pass<Op, Ty, C>>>) -> Self {
        Self {
            passes,
            analyses: AnalysisStore::new(),
            draw_cfg,
            debug_output_dir: None,
            phase_label,
        }
    }

    pub fn set_debug_output_dir(&mut self, debug_output_dir: PathBuf) {
        let specific_dir = debug_output_dir.join(self.phase_label.clone());
        if !specific_dir.exists() {
            fs::create_dir(&specific_dir).unwrap();
        }
        self.debug_output_dir = Some(specific_dir);
    }

    #[tracing::instrument(skip_all, name = "PassManager::run", fields(phase = %self.phase_label))]
    pub fn run(&mut self, ssa: &mut SSA<Op, Ty, C>) {
        if let Some(debug_output_dir) = &self.debug_output_dir {
            if debug_output_dir.exists() {
                fs::remove_dir_all(debug_output_dir).unwrap();
            }
            fs::create_dir(debug_output_dir).unwrap();
        }

        let passes = std::mem::take(&mut self.passes);
        for (i, pass) in passes.iter().enumerate() {
            self.run_pass(ssa, pass.as_ref(), i);
        }
        self.passes = passes;
        self.output_final_debug_info(ssa);
    }

    #[tracing::instrument(skip_all, fields(pass = %pass.name()))]
    fn run_pass(
        &mut self,
        ssa: &mut SSA<Op, Ty, C>,
        pass: &dyn Pass<Op, Ty, C>,
        pass_index: usize,
    ) {
        // Ensure all needed analyses are computed (in dependency order)
        let ordered = topo_sort(pass.needs(), &self.analyses);
        for id in ordered {
            (id.compute_and_store)(ssa as &dyn Any, &mut self.analyses);
        }

        self.output_debug_info(ssa, pass_index, pass.name());
        pass.run(ssa, &self.analyses);
        self.analyses.apply_preserved(&pass.preserves());
        self.output_pass_ssa_snapshot(ssa, pass_index, pass.name());
    }

    /// When `MAVROS_DUMP_PASS_SSA` names a directory, write the SSA text after every pass to
    /// `<dir>/<phase>_<index>_<pass>.txt`. Diffing two such runs bisects exactly which pass
    /// introduced a divergence (used by determinism debugging; cheap no-op when unset).
    fn output_pass_ssa_snapshot(&self, ssa: &SSA<Op, Ty, C>, pass_index: usize, pass_name: &str) {
        let Ok(dir) = std::env::var("MAVROS_DUMP_PASS_SSA") else {
            return;
        };
        let dir = PathBuf::from(dir);
        fs::create_dir_all(&dir).unwrap();
        // The fresh-value counter is part of the snapshot: a pass can mint values that die
        // before the dump, and a counter divergence here surfaces as a text divergence only
        // in some later pass.
        fs::write(
            dir.join(format!(
                "{}_{:03}_{}.txt",
                self.phase_label, pass_index, pass_name
            )),
            format!(
                "next_value_id: {}\n{}",
                ssa.value_num_bound(),
                ssa.to_string(&DefaultSSAAnnotator)
            ),
        )
        .unwrap();
    }

    fn output_debug_info(&mut self, ssa: &SSA<Op, Ty, C>, pass_index: usize, pass_name: &str) {
        use crate::compiler::analysis::flow_analysis::FlowAnalysis;

        let Some(debug_output_dir) = &self.debug_output_dir else {
            return;
        };
        if self.draw_cfg {
            if let Some(cfg) = self.analyses.try_get::<FlowAnalysis>() {
                cfg.generate_images(
                    debug_output_dir.join(format!("before_pass_{}_{}", pass_index, pass_name)),
                    ssa,
                    format!("before {}: {}", pass_index, pass_name),
                );
                fs::write(
                    debug_output_dir
                        .join(format!("before_pass_{}_{}", pass_index, pass_name))
                        .join("code.txt"),
                    format!("{}", ssa.to_string(&DefaultSSAAnnotator)),
                )
                .unwrap();
            }
        }
    }

    fn output_final_debug_info(&mut self, ssa: &mut SSA<Op, Ty, C>) {
        use crate::compiler::analysis::flow_analysis::FlowAnalysis;

        if self.analyses.try_get::<FlowAnalysis>().is_none() {
            let cfg = FlowAnalysis::run(ssa);
            self.analyses.insert_with_deps::<FlowAnalysis>(cfg, vec![]);
        }
        let Some(debug_output_dir) = &self.debug_output_dir else {
            return;
        };
        if self.draw_cfg {
            if let Some(cfg) = self.analyses.try_get::<FlowAnalysis>() {
                cfg.generate_images(
                    debug_output_dir.join("final_result"),
                    ssa,
                    "final result".to_string(),
                );
            }
            fs::write(
                debug_output_dir.join("final_result").join("code.txt"),
                format!("{}", ssa.to_string(&DefaultSSAAnnotator)),
            )
            .unwrap();
        }
    }
}
