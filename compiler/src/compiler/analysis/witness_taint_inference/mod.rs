//! This pass infers, for every HLSSA SSA value, whether it is `Pure` (an R1CS-time constant — a
//! value a symbolic executor could assign statically, i.e. one that would become a constant given
//! enough inlining and constant folding) or `Witness` (may depend on the program's input, public
//! or private), and emits one `FunctionWitnessType` per specialized function (per-value
//! `WitnessShape`, per-block cfg-witness, parameter/return shapes, function-level cfg_witness).
//!
//! It is designed to be run directly before `UntaintControlFlow` which uses the output of this pass
//! to bake `WitnessOf<T>` into types, insert casts, guard witness-dependent writes, linearize
//! witness-dependent control flow and threads a cfg-witness flag argument into constrained calls.
//!
//! #### Reachability and the Lattice
//!
//! The value lattice is the 2-point chain `Pure ≤ Witness`. Every IR construct's taint rule is a
//! join of its inputs, so every transfer function is **monotone** and distributes over taint
//! joins. In other words, a value's taint is exactly the join of the taints of the sources that
//! reach it, and taint only ever flows 'downhill' and accumulates.
//!
//! The resultant analysis is a graph-reachability problem. We first build a directed graph whose
//! nodes are taint positions and whose edges a ≥ b mean "if b is Witness then so too is a". Then, a
//! position is Witness iff some unconditional Witness source (`Top`) or some Witness input reaches
//! it. A least fixed point always exists and is the exact (most precise and sound) answer.
//!
//! #### Positions, Not Values
//!
//! Witness-ness is attributed per 'level' of a type, **not** per value: a `Ref<Array<Field>>`, for
//! example, has independent taint at the ref level, the array level, and the field level (mirroring
//! `WitnessShape::{Scalar, Array, Ref}`). A position is `(owner, path)` where the path descends
//! through `Deref`/`Elem`, and owner is a parameter slot, a return slot, an SSA value, the
//! function's cfg flag, a global, or the synthetic always-Witness source Top. Edges then connect
//! positions level-for-level.
//!
//! #### Two Phases and the Safety of Summaries
//!
//! Because every transfer function distributes over taint values, a per-function summary (the
//! transitive closure of its ≥ graph restricted to its formal inputs and outputs — see
//! `FunctionSummary`) captures the function's taint behavior as precisely as if every call were
//! inlined. So:
//!
//! - **Phase 1 (Polymorphic):** Analyze each function once. Build its local ≥ graph from its
//!   instructions; at each call, instantiate the callee's current summary by substituting the
//!   caller's actual argument/result positions for the callee's formals,
//!   merging the resulting edges. We stay symbolic here rather than instantiating concrete `Pure`
//!   / `Witness`. A worklist over the call graph re-queues a function's callers whenever its
//!   summary closure grows until the process reaches the least fixed point. Recursion (including
//!   mutual recursion) is handled entirely by this summary fixpoint.
//! - **Phase 2 (Concrete):** From the entry function, given concrete input taints, walk the call
//!   graph. For each reachable (function, concrete argument shapes, concrete caller-written
//!   return-deref shapes, concrete cfg flag) context, solve the local graph with the callee
//!   summaries instantiated to concrete edges, producing a concrete `FunctionWitnessType`. Clone
//!   the function once per distinct context, register its `FunctionWitnessType`, and rewrite
//!   `Call` targets to the matching clone. The clone-per-context is required because
//!   `UntaintControlFlow` bakes context-specific `WitnessOf` types and a context-specific cfg-flag
//!   parameter into each body.
//!
//! #### Memory and Aliasing
//!
//! Memory needs no separate analysis: aliasing is resolved by **unification**, directly in the ≥
//! graph. Every copy-shaped flow (value copies, phis, `Select`, array/slice element flow, formal
//! binding at calls and returns) links the two sides level for level — covariantly (`dst·p ≥
//! src·p`) at value levels, **two-way** (`dst·p ≡ src·p`) at every Deref-descended level (see
//! `copy_levels` in [`builder`]). A Deref-descended level names shared mutable memory, and a copy
//! makes the two sides aliases of it; the equations collapse all of an object's aliases' pointee
//! positions into one equivalence class.
//!
//! Memory taint rules are then simple and covariant, reading and writing a pointer's own `Deref`
//! subtree:
//!
//! - Store `*ptr = value`: ptr.deref ≥ value at every non-Deref-descended level, plus cfg(block)
//!   into the pointee's witness leaves (scalars descending arrays, and nested-ref *handles* —
//!   which ref a slot holds after a conditional ref store is itself witness-dependent). The
//!   nested-ref *pointee* levels unify, as for any copy.
//! - Load `result = *ptr`: result ≥ ptr.deref (level for level, again unifying nested-ref
//!   levels), and result.top ≥ ptr.top.
//!
//! Reference invariance (the property that the same mutable object read/written through differently
//! annotated aliases must agree) is not encoded as a special rule but instead emerges from the
//! shared equivalence class: a store through any alias raises the class every alias reads.
//!
//! What follows are two worked examples that would trip up a naïve approach:
//!
//! - Writing a pure Field into a slot later inferred as `Ref<WitnessOf<Field>>` must not taint the
//!   value. It doesn't: Store only adds ptr.deref.scalar ≥ value.scalar, never the reverse, so the
//!   slot being Witness (from some other writer/reader) never flows back into the pure value.
//! - Writing a `Ref<Field>` into a `Ref<Ref<WitnessOf<Field>>>` must force the written ref's inner
//!   Field to `Witness`. It does: the store unifies the outer slot's content with the written ref
//!   at the inner level (`outer.deref.deref ≡ written.deref`); whatever use elsewhere reads that
//!   inner level as Witness pins the shared class, and that is exactly the written ref's inner
//!   level. The "WitnessOf" in the declared type is itself inferred—it exists only because some
//!   site uses that inner as Witness, and that site is what pins the shared class.
//!
//! ##### Precision
//!
//! Unification makes the aliasing **equality-based** (Steensgaard-style), which is deliberately
//! coarser than inclusion-based (Andersen-style) points-to: when refs to *distinct* objects merge
//! (a phi or `Select` over refs, an array of refs), their pointee classes are welded together
//! symmetrically and forever, so a witness write through one original binding taints reads through
//! the other even though they never alias at runtime.
//!
//! This is over-taint only — never unsoundness — and costs at most missed `UntaintControlFlow`
//! opportunities on the write-through-original-after-merge pattern. This precision will instead
//! exist in an aggressive Andersen-based alias-splitting that will use a full points-to analysis.
//! This will leave only irreducible may-alias refs in the IR, on which the two formulations
//! coincide.
//!
//! #### Control-Flow Taint
//!
//! A pure value computed under witness-dependent control flow is still a pure value—its bits are a
//! deterministic function of its pure inputs. The Witness-ness of "does this block run" is handled
//! by guarding effects, not by tainting values. So control-flow taint enters at exactly two places:
//!
//! - phi/merge parameters at the join of a witness `JmpIf` take an edge from the *branch condition
//!   itself* (the merge value differs by which witness-chosen arm ran),
//! - writes performed under witness control flow (`Store`/`ArraySet`/`SlicePush`/`InitGlobal` join
//!   cfg(block)—the function's cfg-flag position `Cfg(f)` plus every dominating witness branch
//!   condition—into the written leaf, because the write must be predicated and its post-state is
//!   witness-dependent).
//!
//! `Cfg(f)` feeds only (b): a merge under a pure local condition stays pure even when the function
//! as a whole is called under witness control flow, since either the whole call runs or none of it
//! does.
//!
//! Pure arithmetic under a witness branch gets no cfg edge. The cfg flag is just another input to
//! the summary, instantiated at call sites from the caller's block taint—the same value
//! `UntaintControlFlow` later pushes as the extra constrained-call argument.
//!
//! #### Witness Sources, Calls, Globals, and the Entry Point
//!
//! `WriteWitness` results are unconditionally Witness (edge from Top). Constrained calls
//! instantiate the callee summary (including ref-argument-output back-taint and pinned inputs).
//! The Deref-descended levels of a *returned* ref are caller-writable inputs to the callee, just
//! like the pointees of ref arguments: the caller's writes through the returned ref seed the
//! callee's context, and the returned value's aliases inside the callee absorb that taint via the
//! return's two-way Deref linking (see `Terminator::Return` in [`builder`]).
//!
//! Unconstrained call results are Pure; their input dependence re-enters structurally as
//! `WriteWitness` inserted by `PrepareEntryPoint`, **not** through the call edge. Globals are
//! init-time constants in this IR (there is no WriteGlobal), hence Pure today, but they are modeled
//! as shared positions anyway so most of the machinery for a witness-carrying global already
//! exists.
//!
//! Two extensions are still be needed for full mutable-global support: global Deref-descended
//! levels would have to become summary *outputs* (a callee writing a witness through a ref held in
//! a global currently has no way to export that fact), and `compute_witness_globals` would have to
//! treat such stores — not just `InitGlobal` — as global writes.
//!
//! The entry function is the flattened wrapper_main; its parameters are seeded all Pure and its
//! cfg_witness is Pure, because `PrepareEntryPoint` prepends a `WriteWitness` for every entry
//! parameter — that is where input dependence (public and private alike) enters.
//!
//! #### Soundness
//!
//! This analysis is **sound** in that it never under-taints. Every taint-introducing edge in the IR
//! is accounted for by the per-OpCode rules below; aliasing is **over-approximated** by
//! unification — two refs can only alias by being connected through a chain of copy-shaped flows,
//! each of which unifies their pointee levels — so every real store → load pair is covered; the
//! analysis only ever adds taint, and computes the least solution that closes all edges. A value
//! typed Pure provably cannot depend on any program input — it is an R1CS-time constant.
//!
//! This analysis is **complete** in that it is exact for the abstraction. Distributivity makes
//! summary-based interprocedural analysis lossless versus a full monomorphic inlining, and the
//! least fixed point is the most precise sound assignment—no value is spuriously Witness beyond
//! what the flow-insensitive, unification-aliased memory abstraction forces.
//!
//! This analysis is **total**. Every HLSSA OpCode reaching this stage has an explicit rule,
//! including unexpected opcodes being handled using ICEs.
//!
//! This analysis is **terminating** because positions are finite (bounded by program size × type
//! depth) and the lattice is 2-point, so both taint fixpoints converge;
//! Phase 2 enumerates finitely many (function, arg-shape, return-deref-shape, cfg) contexts
//! (shapes bounded by type structure × 2 points × 2 cfg).

mod builder;
mod graph;
mod phases;
mod position;

use crate::collections::HashMap;
use crate::compiler::{
    analysis::{
        flow_analysis::FlowAnalysis, witness_info::FunctionWitnessType,
        witness_taint_inference::position::Position,
    },
    ssa::{BlockId, FunctionId, SSAAnotator, ValueId, hlssa::HLSSA},
};

/// The `≥` constraint graph over taint [`Position`]s — built per function by [`builder`] and
/// solved by both phases.
type WitnessTaint = graph::TaintGraph<Position>;

// WITNESS TAINT INFERENCE
// ================================================================================================

/// Sound, complete witness-taint inference using a two-phase approach consisting of polymorphic
/// summaries followed by concrete instantiation.
///
/// See the module documentation for more information.
#[derive(Clone, Debug)]
pub struct WitnessTaintInference {
    functions: HashMap<FunctionId, FunctionWitnessType>,
}

impl WitnessTaintInference {
    pub fn new() -> Self {
        WitnessTaintInference {
            functions: HashMap::default(),
        }
    }

    pub fn try_get_function_witness_type(
        &self,
        func_id: FunctionId,
    ) -> Option<&FunctionWitnessType> {
        self.functions.get(&func_id)
    }

    pub fn set_function_witness_type(&mut self, func_id: FunctionId, wt: FunctionWitnessType) {
        self.functions.insert(func_id, wt);
    }

    pub fn remove_function_witness_type(&mut self, func_id: FunctionId) {
        self.functions.remove(&func_id);
    }

    /// Run the analysis, specializing `ssa` per call context and populating one
    /// [`FunctionWitnessType`] per specialized `FunctionId`.
    pub fn run(&mut self, ssa: &mut HLSSA, flow_analysis: &FlowAnalysis) -> Result<(), String> {
        self.functions = phases::run(ssa, flow_analysis);
        Ok(())
    }
}

impl SSAAnotator for WitnessTaintInference {
    fn annotate_value(&self, function_id: FunctionId, value_id: ValueId) -> String {
        let Some(function_wt) = self.functions.get(&function_id) else {
            return "".to_string();
        };
        function_wt.annotate_value(function_id, value_id)
    }

    fn annotate_block(&self, function_id: FunctionId, block_id: BlockId) -> String {
        let Some(function_wt) = self.functions.get(&function_id) else {
            return "".to_string();
        };
        function_wt.annotate_block(function_id, block_id)
    }

    fn annotate_function(&self, function_id: FunctionId) -> String {
        let Some(function_wt) = self.functions.get(&function_id) else {
            return "".to_string();
        };
        function_wt.annotate_function(function_id)
    }
}

// FUNCTION TAINT SUMMARY
// ================================================================================================

/// A function's polymorphic taint summary: its `≥` closure restricted to formal positions.
///
/// Each edge `(output, input)` means `output ≥ input` — "if `input` is Witness then `output` is
/// Witness" — where both endpoints use only *formal* owners (`Param` / `Return` / `Cfg` /
/// `Global` / `Top`), never internal `Value` owners. This is exactly the information a
/// caller needs: bind the formals to actual argument/result positions and the edges drop into the
/// caller's graph (see [`builder`]).
///
/// Inputs are the levels the caller determines: every parameter level, the Deref-descended levels
/// of every return (the caller may write through a returned ref), read globals, the cfg flag, and
/// `Top`. Outputs are the levels the callee communicates back: every return level and the
/// Deref-descended levels of every parameter (ref-argument writes, and inputs pinned from `Top`).
/// Deref-descended levels are both — they name shared memory either side can write.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct FunctionSummary {
    /// `(output, input)` pairs, with each meaning `output ≥ input`.
    edges: Vec<(Position, Position)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::analysis::witness_info::{WitnessShape, WitnessType};
    use crate::compiler::ssa::hlssa::{
        Type,
        builder::{HLEmitter, HLSSABuilder},
    };

    fn pure() -> WitnessShape {
        WitnessShape::Scalar(WitnessType::Pure)
    }
    fn witness() -> WitnessShape {
        WitnessShape::Scalar(WitnessType::Witness)
    }
    fn fr(n: u64) -> ark_bn254::Fr {
        ark_bn254::Fr::from(n)
    }

    /// Run the pass and return the full inference (to inspect non-main clones).
    fn run_wti(ssa: &mut HLSSA) -> WitnessTaintInference {
        let flow = FlowAnalysis::run(ssa);
        let mut wti = WitnessTaintInference::new();
        wti.run(ssa, &flow).unwrap();
        wti
    }

    /// Run the pass and return the (cloned) entry function's witness type.
    fn run(ssa: &mut HLSSA) -> FunctionWitnessType {
        let wti = run_wti(ssa);
        wti.try_get_function_witness_type(ssa.get_main_id())
            .expect("entry should have a witness type")
            .clone()
    }

    /// Find the unique specialized clone named `name` and return its id and witness type (clones
    /// keep the original's name; only clones get a `FunctionWitnessType`).
    fn clone_fwt(
        ssa: &HLSSA,
        wti: &WitnessTaintInference,
        name: &str,
    ) -> (FunctionId, FunctionWitnessType) {
        let matches: Vec<FunctionId> = ssa
            .get_function_ids()
            .filter(|f| {
                wti.try_get_function_witness_type(*f).is_some()
                    && ssa.get_function(*f).get_name() == name
            })
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "expected exactly one specialized clone of `{name}`, found {}",
            matches.len()
        );
        (
            matches[0],
            wti.try_get_function_witness_type(matches[0])
                .unwrap()
                .clone(),
        )
    }

    fn ref_of_shape(inner: WitnessShape) -> WitnessShape {
        WitnessShape::Ref(WitnessType::Pure, Box::new(inner))
    }

    /// `WriteWitness` results are unconditionally Witness; a pure parameter stays Pure.
    #[test]
    fn write_witness_is_witness_pure_stays_pure() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_main_id();
        let mut sb = HLSSABuilder::new(&mut ssa);
        sb.modify_function(main_id, |b| {
            b.function.add_return_type(Type::field());
            b.function.add_return_type(Type::field());
            let entry = b.function.get_entry_id();
            let mut e = b.block(entry);
            let x = e.add_parameter(Type::field());
            let w = e.write_witness(x);
            e.terminate_return(vec![x, w]);
        });
        let fwt = run(&mut ssa);
        assert_eq!(fwt.returns_witness, vec![pure(), witness()]);
    }

    /// A `Cast → WitnessOf` is a witness source: even applied to a pure constant (as
    /// `PrepareEntryPoint` does for the default element of the witness-input array), its result is
    /// Witness.
    #[test]
    fn witness_of_cast_is_witness() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_main_id();
        let mut sb = HLSSABuilder::new(&mut ssa);
        sb.modify_function(main_id, |b| {
            b.function.add_return_type(Type::witness_of(Type::field()));
            let entry = b.function.get_entry_id();
            let mut e = b.block(entry);
            let _x = e.add_parameter(Type::field());
            let c = e.field_const(fr(0));
            let w = e.cast_to_witness_of(c);
            e.terminate_return(vec![w]);
        });
        let fwt = run(&mut ssa);
        assert_eq!(fwt.returns_witness, vec![witness()]);
    }

    /// Worked example #1: a pure value stored into a slot that also receives a witness stays Pure
    /// (the store is covariant: `*ptr ≥ value`, never the reverse), while a load of that slot is
    /// Witness (it reads the pointee, which a witness store tainted).
    #[test]
    fn store_is_covariant_load_reads_pointee() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_main_id();
        let mut sb = HLSSABuilder::new(&mut ssa);
        sb.modify_function(main_id, |b| {
            b.function.add_return_type(Type::field());
            b.function.add_return_type(Type::field());
            let entry = b.function.get_entry_id();
            let mut e = b.block(entry);
            let x = e.add_parameter(Type::field());
            let p = e.alloc(Type::field());
            let c = e.field_const(fr(5));
            e.store(p, c); // pure store
            let w = e.write_witness(x); // Witness
            e.store(p, w); // witness store → pointee becomes Witness
            let r = e.load(p); // reads the (now Witness) pointee
            e.terminate_return(vec![c, r]);
        });
        let fwt = run(&mut ssa);
        // c stayed Pure (covariance); r is Witness (load reads the tainted pointee).
        assert_eq!(fwt.returns_witness, vec![pure(), witness()]);
    }

    /// Arithmetic propagates taint by join: witness ∨ pure = witness; pure ∨ pure = pure.
    #[test]
    fn arithmetic_joins_operand_taint() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_main_id();
        let mut sb = HLSSABuilder::new(&mut ssa);
        sb.modify_function(main_id, |b| {
            b.function.add_return_type(Type::field());
            b.function.add_return_type(Type::field());
            let entry = b.function.get_entry_id();
            let mut e = b.block(entry);
            let x = e.add_parameter(Type::field());
            let w = e.write_witness(x);
            let a = e.add(w, x); // witness ∨ pure = witness
            let c = e.field_const(fr(2));
            let d = e.add(c, x); // pure ∨ pure = pure
            e.terminate_return(vec![d, a]);
        });
        let fwt = run(&mut ssa);
        assert_eq!(fwt.returns_witness, vec![pure(), witness()]);
    }

    /// Interprocedural ref-argument output: a callee that writes a witness into `*p` back-taints the
    /// caller's pointee, so a subsequent load in the caller is Witness. Exercises the summary
    /// instantiation + arg-out mapping onto the argument's deref levels.
    #[test]
    fn call_arg_out_back_taints_caller_ref() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_main_id();
        let mut sb = HLSSABuilder::new(&mut ssa);
        let helper_id = sb.ssa().add_function("helper".to_string());
        // helper(p: Ref<Field>): *p = write_witness(7); return
        sb.modify_function(helper_id, |b| {
            let entry = b.function.get_entry_id();
            let mut e = b.block(entry);
            let p = e.add_parameter(Type::field().ref_of());
            let c = e.field_const(fr(7));
            let w = e.write_witness(c);
            e.store(p, w);
            e.terminate_return(vec![]);
        });
        // main(x): q = alloc; *q = 0; helper(q); return *q
        sb.modify_function(main_id, |b| {
            b.function.add_return_type(Type::field());
            let entry = b.function.get_entry_id();
            let mut e = b.block(entry);
            let _x = e.add_parameter(Type::field());
            let q = e.alloc(Type::field());
            let c0 = e.field_const(fr(0));
            e.store(q, c0);
            e.call(helper_id, vec![q], 0);
            let r = e.load(q);
            e.terminate_return(vec![r]);
        });
        let fwt = run(&mut ssa);
        assert_eq!(fwt.returns_witness, vec![witness()]);
    }

    /// A self-recursive callee that writes a witness into its ref argument: the summary fixpoint must
    /// converge (no infinite loop) and still propagate the arg-out taint to the caller.
    #[test]
    fn recursion_converges_with_arg_out() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_main_id();
        let mut sb = HLSSABuilder::new(&mut ssa);
        let rec_id = sb.ssa().add_function("rec".to_string());
        // rec(p: Ref<Field>): *p = write_witness(1); rec(p); return
        sb.modify_function(rec_id, |b| {
            let entry = b.function.get_entry_id();
            let mut e = b.block(entry);
            let p = e.add_parameter(Type::field().ref_of());
            let c = e.field_const(fr(1));
            let w = e.write_witness(c);
            e.store(p, w);
            e.call(rec_id, vec![p], 0); // self-recursion
            e.terminate_return(vec![]);
        });
        sb.modify_function(main_id, |b| {
            b.function.add_return_type(Type::field());
            let entry = b.function.get_entry_id();
            let mut e = b.block(entry);
            let _x = e.add_parameter(Type::field());
            let q = e.alloc(Type::field());
            let c0 = e.field_const(fr(0));
            e.store(q, c0);
            e.call(rec_id, vec![q], 0);
            let r = e.load(q);
            e.terminate_return(vec![r]);
        });
        let fwt = run(&mut ssa);
        assert_eq!(fwt.returns_witness, vec![witness()]);
    }

    /// Worked example #2 (the case the old `find_alloc_origin` missed): a `Ref<Field>` written into a
    /// `Ref<Ref<Field>>` and loaded back aliases the original at the inner level (the store and the
    /// load both unify nested-ref levels). A witness stored through one alias is therefore visible
    /// through the other — the loaded inner `Field` is Witness via the shared equivalence class.
    #[test]
    fn nested_ref_aliases_unify() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_main_id();
        let mut sb = HLSSABuilder::new(&mut ssa);
        sb.modify_function(main_id, |b| {
            b.function.add_return_type(Type::field());
            let entry = b.function.get_entry_id();
            let mut e = b.block(entry);
            let _x = e.add_parameter(Type::field());
            // inner: Ref<Field>, with a witness stored into its pointee.
            let inner = e.alloc(Type::field());
            let c = e.field_const(fr(9));
            let w = e.write_witness(c);
            e.store(inner, w);
            // outer: Ref<Ref<Field>>, holding `inner`.
            let outer = e.alloc(Type::field().ref_of());
            e.store(outer, inner);
            // Read the ref back out and deref it: aliases `inner`'s (witness) pointee.
            let loaded = e.load(outer);
            let final_val = e.load(loaded);
            e.terminate_return(vec![final_val]);
        });
        let fwt = run(&mut ssa);
        assert_eq!(fwt.returns_witness, vec![witness()]);
    }

    /// Characterization of the accepted unification imprecision: a phi over refs to two distinct
    /// allocations welds their pointee classes together, so a witness store through ONE original
    /// ref taints loads through the *other*, even though the two never alias at runtime
    /// (inclusion-based points-to would keep them apart).
    ///
    /// Over-taint only, never unsoundness.
    #[test]
    fn merged_refs_unify_taint() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_main_id();
        let mut sb = HLSSABuilder::new(&mut ssa);
        sb.modify_function(main_id, |b| {
            b.function.add_return_type(Type::field());
            let entry = b.function.get_entry_id();
            let mut e = b.block(entry);
            let x = e.add_parameter(Type::field());
            let ra = e.alloc(Type::field());
            let rb = e.alloc(Type::field());
            let c0 = e.field_const(fr(0));
            e.store(ra, c0);
            e.store(rb, c0);
            let cond = e.eq(x, c0); // a pure condition
            // The merge phi unifies ra/rb pointees.
            let _merged = e.build_if_else(
                cond,
                vec![Type::field().ref_of()],
                |_| vec![ra],
                |_| vec![rb],
            );
            let w = e.write_witness(x);
            e.store(rb, w); // witness store through rb ONLY
            let v = e.load(ra); // ...but the load through ra sees it via the merged class
            e.terminate_return(vec![v]);
        });
        let fwt = run(&mut ssa);
        assert_eq!(fwt.returns_witness, vec![witness()]);
    }

    /// A conditional store of a *ref* under a witness branch: which ref the slot holds depends on
    /// the witness branch, so the slot's handle level must pick up cf-taint and a double-load
    /// through it must be Witness — even though both candidate pointees hold pure values.
    #[test]
    fn conditional_ref_store_taints_loaded_handle() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_main_id();
        let mut sb = HLSSABuilder::new(&mut ssa);
        sb.modify_function(main_id, |b| {
            b.function.add_return_type(Type::field());
            let entry = b.function.get_entry_id();
            let mut e = b.block(entry);
            let x = e.add_parameter(Type::field());
            let r1 = e.alloc(Type::field());
            let r2 = e.alloc(Type::field());
            let c1 = e.field_const(fr(1));
            let c2 = e.field_const(fr(2));
            e.store(r1, c1);
            e.store(r2, c2);
            let p = e.alloc(Type::field().ref_of());
            e.store(p, r1);
            let w = e.write_witness(x);
            let zero = e.field_const(fr(0));
            let cond = e.eq(w, zero); // witness condition
            // if cond { *p = r2 }
            e.build_if_else(
                cond,
                vec![],
                |e| {
                    e.store(p, r2);
                    vec![]
                },
                |_| vec![],
            );
            let q = e.load(p); // q is r1 or r2 depending on the witness branch
            let v = e.load(q); // ...so v is 1 or 2 depending on the witness branch
            e.terminate_return(vec![v]);
        });
        let fwt = run(&mut ssa);
        assert_eq!(fwt.returns_witness, vec![witness()]);
    }

    /// The dual precision guard: the same ref-into-slot store made *unconditionally* leaves the
    /// double-load Pure (both candidate pointees hold pure values, and no branch chooses).
    #[test]
    fn unconditional_ref_store_stays_pure() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_main_id();
        let mut sb = HLSSABuilder::new(&mut ssa);
        sb.modify_function(main_id, |b| {
            b.function.add_return_type(Type::field());
            let entry = b.function.get_entry_id();
            let mut e = b.block(entry);
            let _x = e.add_parameter(Type::field());
            let r1 = e.alloc(Type::field());
            let r2 = e.alloc(Type::field());
            let c1 = e.field_const(fr(1));
            let c2 = e.field_const(fr(2));
            e.store(r1, c1);
            e.store(r2, c2);
            let p = e.alloc(Type::field().ref_of());
            e.store(p, r1);
            e.store(p, r2);
            let q = e.load(p);
            let v = e.load(q);
            e.terminate_return(vec![v]);
        });
        let fwt = run(&mut ssa);
        assert_eq!(fwt.returns_witness, vec![pure()]);
    }

    /// The dual precision guard: refs that never meet at a merge are NOT unified — a witness store
    /// through one allocation leaves loads through the other Pure.
    #[test]
    fn unmerged_refs_stay_separate() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_main_id();
        let mut sb = HLSSABuilder::new(&mut ssa);
        sb.modify_function(main_id, |b| {
            b.function.add_return_type(Type::field());
            let entry = b.function.get_entry_id();
            let mut e = b.block(entry);
            let x = e.add_parameter(Type::field());
            let ra = e.alloc(Type::field());
            let rb = e.alloc(Type::field());
            let c0 = e.field_const(fr(0));
            e.store(ra, c0);
            e.store(rb, c0);
            let w = e.write_witness(x);
            e.store(rb, w); // witness store through rb
            let v = e.load(ra); // ra was never merged with rb: stays Pure
            e.terminate_return(vec![v]);
        });
        let fwt = run(&mut ssa);
        assert_eq!(fwt.returns_witness, vec![pure()]);
    }

    /// Under witness control flow: the merge phi is Witness (it differs by which witness-chosen arm
    /// ran) and a conditional store's slot is Witness (cfg taints the written leaf), but pure
    /// arithmetic computed in the branch stays a Pure *value* — no cfg edge is attached to arithmetic.
    #[test]
    fn witness_branch_taints_writes_and_merge_not_arithmetic() {
        use crate::compiler::ssa::hlssa::OpCode;

        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_main_id();
        let mut sb = HLSSABuilder::new(&mut ssa);
        sb.modify_function(main_id, |b| {
            b.function.add_return_type(Type::field()); // the merge phi
            b.function.add_return_type(Type::field()); // load after the conditional store
            let entry = b.function.get_entry_id();
            let mut e = b.block(entry);
            let x = e.add_parameter(Type::field());
            let w = e.write_witness(x);
            let zero = e.field_const(fr(0));
            let cond = e.eq(w, zero); // a witness-dependent condition
            let p = e.alloc(Type::field());
            let merge = e.build_if_else(
                cond,
                vec![Type::field()],
                |e| {
                    let c1 = e.field_const(fr(1));
                    let c2 = e.field_const(fr(2));
                    let a = e.add(c1, c2); // pure arithmetic under a witness branch
                    let c7 = e.field_const(fr(7));
                    e.store(p, c7); // conditional store → the slot becomes Witness
                    vec![a]
                },
                |e| {
                    let c0 = e.field_const(fr(0));
                    vec![c0]
                },
            );
            let r = e.load(p);
            e.terminate_return(vec![merge[0], r]);
        });

        let fwt = run(&mut ssa);
        // merge phi Witness (phi at a witness JmpIf), load-after-conditional-store Witness (cfg taint).
        assert_eq!(fwt.returns_witness, vec![witness(), witness()]);

        // The pure arithmetic computed under the witness branch is still a Pure value. Read it off the
        // clone (phase 2 remaps value ids), finding the sole `BinaryArithOp` result.
        let clone_id = ssa.get_main_id();
        let add_result = ssa
            .get_function(clone_id)
            .get_blocks()
            .flat_map(|(_, block)| block.get_instructions())
            .find_map(|instr| {
                if let OpCode::BinaryArithOp { result, .. } = instr {
                    Some(*result)
                } else {
                    None
                }
            })
            .expect("clone should contain the add");
        assert_eq!(fwt.value_witness_types.get(&add_result), Some(&pure()));
    }

    /// Marcin's counterexample: a callee allocates and returns a ref; the caller writes a witness
    /// through the returned ref. The caller's load must be Witness, and — crucially — the callee
    /// clone must see the caller-injected taint: its return shape and its allocation's pointee
    /// must be Witness (reference invariance for returned refs).
    #[test]
    fn caller_write_through_returned_ref_taints_callee() {
        use crate::compiler::ssa::hlssa::OpCode;

        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_main_id();
        let mut sb = HLSSABuilder::new(&mut ssa);
        let helper_id = sb.ssa().add_function("make_ref".to_string());
        // make_ref() -> Ref<Field>: p = alloc; *p = 17; return p
        sb.modify_function(helper_id, |b| {
            b.function.add_return_type(Type::field().ref_of());
            let entry = b.function.get_entry_id();
            let mut e = b.block(entry);
            let p = e.alloc(Type::field());
            let c = e.field_const(fr(17));
            e.store(p, c);
            e.terminate_return(vec![p]);
        });
        // main(x): z = make_ref(); *z = write_witness(x); return *z
        sb.modify_function(main_id, |b| {
            b.function.add_return_type(Type::field());
            let entry = b.function.get_entry_id();
            let mut e = b.block(entry);
            let x = e.add_parameter(Type::field());
            let z = e.call(helper_id, vec![], 1)[0];
            let w = e.write_witness(x);
            e.store(z, w);
            let r = e.load(z);
            e.terminate_return(vec![r]);
        });
        let wti = run_wti(&mut ssa);
        let main_fwt = wti
            .try_get_function_witness_type(ssa.get_main_id())
            .unwrap();
        assert_eq!(main_fwt.returns_witness, vec![witness()]);

        let (clone_id, helper_fwt) = clone_fwt(&ssa, &wti, "make_ref");
        assert_eq!(helper_fwt.returns_witness, vec![ref_of_shape(witness())]);
        // The allocation the returned ref names must itself be witness-pointee in the clone.
        let alloc_result = ssa
            .get_function(clone_id)
            .get_blocks()
            .flat_map(|(_, block)| block.get_instructions())
            .find_map(|instr| {
                if let OpCode::Alloc { result, .. } = instr {
                    Some(*result)
                } else {
                    None
                }
            })
            .expect("clone should contain the alloc");
        assert_eq!(
            helper_fwt.value_witness_types.get(&alloc_result),
            Some(&ref_of_shape(witness()))
        );
    }

    /// A returned ref forwarded up two call levels: the caller's write must back-taint both the
    /// forwarding function's and the allocating function's clones (exercises the Return-deref
    /// summary inputs chaining across calls).
    #[test]
    fn returned_ref_two_level_pass_through() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_main_id();
        let mut sb = HLSSABuilder::new(&mut ssa);
        let inner_id = sb.ssa().add_function("inner".to_string());
        let mid_id = sb.ssa().add_function("mid".to_string());
        sb.modify_function(inner_id, |b| {
            b.function.add_return_type(Type::field().ref_of());
            let entry = b.function.get_entry_id();
            let mut e = b.block(entry);
            let p = e.alloc(Type::field());
            let c = e.field_const(fr(17));
            e.store(p, c);
            e.terminate_return(vec![p]);
        });
        // mid() -> Ref<Field>: just forwards inner()'s ref.
        sb.modify_function(mid_id, |b| {
            b.function.add_return_type(Type::field().ref_of());
            let entry = b.function.get_entry_id();
            let mut e = b.block(entry);
            let r = e.call(inner_id, vec![], 1)[0];
            e.terminate_return(vec![r]);
        });
        sb.modify_function(main_id, |b| {
            b.function.add_return_type(Type::field());
            let entry = b.function.get_entry_id();
            let mut e = b.block(entry);
            let x = e.add_parameter(Type::field());
            let z = e.call(mid_id, vec![], 1)[0];
            let w = e.write_witness(x);
            e.store(z, w);
            let r = e.load(z);
            e.terminate_return(vec![r]);
        });
        let wti = run_wti(&mut ssa);
        let main_fwt = wti
            .try_get_function_witness_type(ssa.get_main_id())
            .unwrap();
        assert_eq!(main_fwt.returns_witness, vec![witness()]);
        let (_, mid_fwt) = clone_fwt(&ssa, &wti, "mid");
        assert_eq!(mid_fwt.returns_witness, vec![ref_of_shape(witness())]);
        let (_, inner_fwt) = clone_fwt(&ssa, &wti, "inner");
        assert_eq!(inner_fwt.returns_witness, vec![ref_of_shape(witness())]);
    }

    /// A ref-returning function that returns its own recursive call's result: the extended
    /// (arg, return-deref, cfg) context key must still memoize to a single context (termination),
    /// and the caller's write must back-taint it.
    #[test]
    fn returned_ref_recursion_terminates() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_main_id();
        let mut sb = HLSSABuilder::new(&mut ssa);
        let rec_id = sb.ssa().add_function("rec".to_string());
        // rec() -> Ref<Field>: return rec()
        sb.modify_function(rec_id, |b| {
            b.function.add_return_type(Type::field().ref_of());
            let entry = b.function.get_entry_id();
            let mut e = b.block(entry);
            let z = e.call(rec_id, vec![], 1)[0];
            e.terminate_return(vec![z]);
        });
        sb.modify_function(main_id, |b| {
            b.function.add_return_type(Type::field());
            let entry = b.function.get_entry_id();
            let mut e = b.block(entry);
            let x = e.add_parameter(Type::field());
            let z = e.call(rec_id, vec![], 1)[0];
            let w = e.write_witness(x);
            e.store(z, w);
            let r = e.load(z);
            e.terminate_return(vec![r]);
        });
        let wti = run_wti(&mut ssa);
        let main_fwt = wti
            .try_get_function_witness_type(ssa.get_main_id())
            .unwrap();
        assert_eq!(main_fwt.returns_witness, vec![witness()]);
        // clone_fwt asserts there is exactly one `rec` clone.
        let (_, rec_fwt) = clone_fwt(&ssa, &wti, "rec");
        assert_eq!(rec_fwt.returns_witness, vec![ref_of_shape(witness())]);
    }

    /// A callee returning an alias of its ref parameter: a witness written through the *returned*
    /// ref must be visible through the *original* argument (needs the `Param·Deref ≥ Return·Deref`
    /// summary edge that Return-deref inputs make expressible).
    #[test]
    fn returned_ref_alias_of_param_back_taints_arg() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_main_id();
        let mut sb = HLSSABuilder::new(&mut ssa);
        let id_id = sb.ssa().add_function("id".to_string());
        // id(p: Ref<Field>) -> Ref<Field>: return p
        sb.modify_function(id_id, |b| {
            b.function.add_return_type(Type::field().ref_of());
            let entry = b.function.get_entry_id();
            let mut e = b.block(entry);
            let p = e.add_parameter(Type::field().ref_of());
            e.terminate_return(vec![p]);
        });
        // main(x): q = alloc; *q = 0; r = id(q); *r = write_witness(x); return *q
        sb.modify_function(main_id, |b| {
            b.function.add_return_type(Type::field());
            let entry = b.function.get_entry_id();
            let mut e = b.block(entry);
            let x = e.add_parameter(Type::field());
            let q = e.alloc(Type::field());
            let c0 = e.field_const(fr(0));
            e.store(q, c0);
            let r = e.call(id_id, vec![q], 1)[0];
            let w = e.write_witness(x);
            e.store(r, w);
            let s = e.load(q);
            e.terminate_return(vec![s]);
        });
        let fwt = run(&mut ssa);
        assert_eq!(fwt.returns_witness, vec![witness()]);
    }

    /// Precision guard: without any witness write through the returned ref, both the caller and
    /// the callee clone stay all-Pure (the Return-deref machinery must not over-taint).
    #[test]
    fn returned_ref_stays_pure_without_caller_write() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_main_id();
        let mut sb = HLSSABuilder::new(&mut ssa);
        let helper_id = sb.ssa().add_function("make_ref".to_string());
        sb.modify_function(helper_id, |b| {
            b.function.add_return_type(Type::field().ref_of());
            let entry = b.function.get_entry_id();
            let mut e = b.block(entry);
            let p = e.alloc(Type::field());
            let c = e.field_const(fr(17));
            e.store(p, c);
            e.terminate_return(vec![p]);
        });
        sb.modify_function(main_id, |b| {
            b.function.add_return_type(Type::field());
            let entry = b.function.get_entry_id();
            let mut e = b.block(entry);
            let _x = e.add_parameter(Type::field());
            let z = e.call(helper_id, vec![], 1)[0];
            let c5 = e.field_const(fr(5));
            e.store(z, c5);
            let r = e.load(z);
            e.terminate_return(vec![r]);
        });
        let wti = run_wti(&mut ssa);
        let main_fwt = wti
            .try_get_function_witness_type(ssa.get_main_id())
            .unwrap();
        assert_eq!(main_fwt.returns_witness, vec![pure()]);
        let (_, helper_fwt) = clone_fwt(&ssa, &wti, "make_ref");
        assert_eq!(helper_fwt.returns_witness, vec![ref_of_shape(pure())]);
    }
}
