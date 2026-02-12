# Implementation Plan

## Guiding Principles

1. **Keep the build compiling** after each step (or at least each phase).
2. **Old and new can coexist** temporarily: WitnessOf and WitnessRef can both exist in
   TypeExpr during migration.
3. **Test after each phase**: run `target/release/test-runner --root noir_tests` to verify
   all 20 tests pass.
4. **Smallest possible diffs**: prefer incremental changes over big-bang rewrites.

---

## Phase 1: Type System Foundation

### Step 1.1: Eliminate `V` annotation parameter and add WitnessOf

**File:** `src/compiler/ir/type.rs`

This is the largest single step. The generic annotation parameter `V` on `Type<V>`,
`TypeExpr<V>`, and all dependent types is removed entirely:

- `Type<V>` → `Type` (struct with just `expr: TypeExpr`, no `annotation` field)
- `TypeExpr<V>` → `TypeExpr`
- Remove `CommutativeMonoid` trait, `Empty` struct, `ConstantTaint` enum
- Remove `annotation` field and all annotation-related methods (`as_pure()`,
  `combine_with_annotation()`, `pure_taint_for_type()`, etc.)
- Add `WitnessOf(Box<Type>)` variant to `TypeExpr`
- Remove `WitnessRef` variant from `TypeExpr`
- Add normalizing constructor `Type::witness_of(inner)` (enforces idempotency)
- Add helper methods:
  - `is_witness_of() -> bool`
  - `unwrap_witness_of() -> &Type` (returns inner if WitnessOf, panics otherwise)
  - `try_unwrap_witness_of() -> Option<&Type>`
  - `strip_witness() -> Type` (strips one level of WitnessOf)
  - `strip_all_witness() -> Type` (recursively strips all WitnessOf)
- Update ALL pattern matches on TypeExpr to handle WitnessOf:
  - `contains_ptrs`: WitnessOf(inner) → inner.contains_ptrs()
  - `get_arithmetic_result_type`: delegate through WitnessOf
  - `is_numeric`: WitnessOf(numeric) → true
  - `is_heap_allocated`: WitnessOf(_) → true (witness values live on tape)
  - `calculate_type_size`: WitnessOf(inner) → 1 (pointer-sized)
  - Display: `WitnessOf(Field)`, `WitnessOf(Array<...>)`, etc.

**Cascading changes:** Removing `V` propagates to every file that uses `Type<V>`:
- `SSA<V>` → `SSA`, `Function<V>` → `Function`, `Block<V>` → `Block`, `OpCode<V>` → `OpCode`
- All pass signatures lose their generic parameter
- `prepare_rebuild::<ConstantTaint>()` → `prepare_rebuild()`
- All `where V: CommutativeMonoid` bounds are removed

**Test:** Compiles (with many downstream updates), existing tests pass.

### Step 1.2: Add CastTarget::WitnessOf and remove old opcodes

**File:** `src/compiler/ssa.rs`

- Add `WitnessOf` variant to `CastTarget` enum
- Remove `OpCode::PureToWitnessRef` (replaced by `Cast { target: WitnessOf }`)
- Remove `OpCode::UnboxField` (replaced by appropriate Cast or identity)
- Rename `Const::WitnessRef` → `Const::Witness`
- Update Cast handling in all passes to handle the new variant
  (most passes can just pass it through unchanged)

**Test:** Compiles, existing tests pass.

### Step 1.3: Add join() operation for types

**File:** `src/compiler/ir/type.rs`

- Implement `Type::join(a: &Type, b: &Type) -> Type` as described in
  [01-type-system.md](01-type-system.md) section 3
- Implement `Type::is_subtype_of(a, b) -> bool` for checking subtyping relation
- Implement `needs_witness_cast(from, to) -> bool` for cast insertion
- Add unit tests for join, subtyping, and needs_witness_cast

**Test:** Unit tests for join/subtyping pass.

---

## Phase 2: Witness Type Inference

### Step 2.1: Create WitnessInfo types

**File:** `src/compiler/witness_info.rs` (NEW)

- Define `WitnessInfo` enum (Pure, Witness, Variable, Join) — modeled after current `Taint`
- Define `WitnessType` enum (Scalar, Array, Ref, Tuple) — modeled after current `TaintType`
- Define `WitnessJudgement` enum (Eq, Le) — same as current `Judgement`
- Implement all helper methods (union, gather_vars, substitute, simplify_and_default,
  toplevel_info, child_type, with_toplevel_info)

Note: This can mostly be a copy-rename of the relevant parts of `taint_analysis.rs`.

**Test:** Compiles.

### Step 2.2: Create WitnessTypeInference pass

**File:** `src/compiler/witness_type_inference.rs` (NEW)

- `WitnessTypeInference` struct with `functions: HashMap<FunctionId, FunctionWitnessType>`
- `run(ssa, flow_analysis)`: entry point
  - Get call graph, compute SCCs in topological order
  - For each SCC: call `analyze_scc()`
- `analyze_scc(scc)`: Mycroft fixpoint
  - Initialize signatures to bottom
  - Iterate: analyze each function, solve constraints, check for changes
  - Converge
- `analyze_function(func_id)`: constraint generation
  - Same logic as current `TaintAnalysis::analyze_function()` with renamed types:
    - Tracks `cfg_witness` per-function and `block_cfg_witness` per-block
      (same as current `cfg_taint` / `block_cfg_taints`)
    - Includes `Le(block_cfg_witness, inner.toplevel)` on Store
      (needed for correct Ref typing under witness branches)
  - For Call: look up callee's current signature (from Mycroft state or already-analyzed)

### Step 2.3: Create WitnessConstraintSolver

**File:** Can reuse `constraint_solver.rs` if we add a generic parameter, or create a
parallel solver in `witness_constraint_solver.rs`.

- Same algorithm as current ConstraintSolver, but operates on WitnessInfo/WitnessType
- Uses existing UnionFind (or a variant that maps to Pure/Witness instead of ConstantTaint)

### Step 2.4: Integration test — run both analyses

**File:** `src/driver.rs`

- After running TaintAnalysis (current), also run WitnessTypeInference (new)
- Compare results: for each value, check that the WitnessType inference result is
  consistent with the TaintType result
- This is a **validation harness** to ensure correctness before switching over

**Test:** Both analyses agree on all 20 test cases.

---

## Phase 3: Switch Monomorphization & UntaintControlFlow to WitnessTypeInference

Since `FunctionWitnessType` and `FunctionTaint` are structurally isomorphic, this phase
is primarily a rename-level refactor: switch both Monomorphization and UntaintControlFlow
from consuming `TaintAnalysis` to consuming `WitnessTypeInference`.

Type application (baking WitnessOf into SSA types) stays in UntaintControlFlow for now
via a renamed `apply_witness_type()` method. The separate WitnessCastInsertion pass is
deferred to Phase 4.

### Step 3.1: Switch Monomorphization to WitnessTypeInference

**File:** `src/compiler/monomorphization.rs`

- Replace imports: `taint_analysis::{...}` → `witness_info::{...}` +
  `witness_type_inference::WitnessTypeInference` + `witness_constraint_solver::WitnessConstraintSolver`
- `Signature` struct: `cfg_witness: WitnessInfo`, `param_witnesses: Vec<WitnessType>`,
  `return_witnesses: Vec<WitnessType>`
- `run()`: accept `&mut WitnessTypeInference` instead of `&mut TaintAnalysis`
- Use `WitnessConstraintSolver` instead of `ConstraintSolver`
- `monomorphize_main_signature()`: operate on `FunctionWitnessType`
- `monomorphize_main_taint()` → `monomorphize_main_witness()`: same logic with
  `WitnessType::Scalar(WitnessInfo::Witness)` etc.
- Call site signatures: read from `value_witness_types` and `block_cfg_witness`

### Step 3.2: Switch UntaintControlFlow to WitnessTypeInference

**File:** `src/compiler/untaint_control_flow.rs`

- Replace imports: `taint_analysis::{...}` → `witness_info::{...}` +
  `witness_type_inference::WitnessTypeInference`
- `run()`: accept `&WitnessTypeInference` instead of `&TaintAnalysis`
- `run_function()`: accept `&FunctionWitnessType` instead of `&FunctionTaint`
- Rename `typify_taint()` → `apply_witness_type()`: use `WitnessType` instead of `TaintType`
  - `TaintType::Primitive(taint)` → `WitnessType::Scalar(info)`
  - `TaintType::NestedImmutable(top, inner)` → `WitnessType::Array(top, inner)`
  - `TaintType::NestedMutable(top, inner)` → `WitnessType::Ref(top, inner)`
  - `TaintType::Tuple(top, children)` → `WitnessType::Tuple(top, children)`
  - `taint.expect_constant().is_witness()` → `info.expect_constant().is_witness()`
- All taint field accesses:
  - `function_taint.value_taints` → `function_wt.value_witness_types`
  - `function_taint.block_cfg_taints` → `function_wt.block_cfg_witness`
  - `function_taint.cfg_taint` → `function_wt.cfg_witness`
  - `function_taint.returns_taint` → `function_wt.returns_witness`
  - `Taint::Constant(ConstantTaint::Witness)` → `WitnessInfo::Witness`
  - `ConstantTaint::Pure/Witness` → `ConstantWitness::Pure/Witness`
- Control flow linearization logic (JmpIf→Select, guarded stores/asserts, CFG witness
  parameters) is unchanged — just operating on the renamed types

### Step 3.3: Update driver.rs

**File:** `src/driver.rs`

- Remove `TaintAnalysis` from active pipeline
- Remove `compare_with_taint_analysis()` call
- Feed `WitnessTypeInference` to `Monomorphization::run()`
- Feed `WitnessTypeInference` to `UntaintControlFlow::run()`
- Update debug output calls to use `witness_inference`

**Test:** Full pipeline works. All 14 noir_tests pass.

---

## Phase 4: WitnessCastInsertion & UntaintControlFlow Separation

### Step 4.1: Create WitnessCastInsertion pass

**File:** `src/compiler/witness_cast_insertion.rs` (NEW)

A separate pass that runs after monomorphization. Walks specialized functions,
applies witness types to SSA (baking WitnessOf into TypeExpr), and inserts explicit
`Cast { target: WitnessOf }` where type mismatches exist.

This extracts the type-application logic (currently `apply_witness_type()` in
UntaintControlFlow) into its own pass and adds explicit Cast instruction insertion.

- `run(ssa, witness_analysis)` — entry point
- `apply_witness_types(function, function_wt)` — bake WitnessOf into block params,
  instruction types, return types
- `insert_casts(function, type_map)` — per-function, walks instructions + terminators
- `needs_witness_cast(actual, expected)` — recursive type comparison
- `emit_scalar_cast(value)` → single Cast instruction
- `emit_array_cast(function, value, from_elem, to_elem)` → conversion loop
  (adapted from `WitnessToRef::emit_array_conversion_loop()`)
- `emit_tuple_cast(function, value, from_fields, to_fields)` → project + cast + mk_tuple

Note: `WitnessOf(Array<...>)` as a cast target is not handled — panic with clear error.

### Step 4.2: Simplify UntaintControlFlow

**File:** `src/compiler/untaint_control_flow.rs`

With WitnessCastInsertion handling type application, UntaintControlFlow is simplified:
- Remove `apply_witness_type()` — types are already baked into SSA
- Remove the SSA `prepare_rebuild()` pattern for type conversion
- Read WitnessOf types directly from SSA TypeExpr
- Control flow linearization logic unchanged

### Step 4.3: Update downstream passes to check WitnessOf directly

Since `ConstantTaint` is eliminated, all downstream passes that currently check
`annotation.is_witness()` must instead check `type.is_witness_of()`:

- `ExplicitWitness`: check `WitnessOf` in TypeExpr instead of ConstantTaint annotation
- `WitnessLowering` (ex-WitnessToRef): check `WitnessOf` in TypeExpr
- R1CGen, CodeGen: check `WitnessOf` in TypeExpr

**Test:** Full pipeline works. All tests pass.

---

## Phase 5: Cleanup

### Step 5.1: Remove old TaintAnalysis

- Delete `src/compiler/taint_analysis.rs`
- Remove all references to `TaintType`, `FunctionTaint`, `Taint` enum
- Remove the validation harness from Phase 2

### Step 5.2: Rename/clean up constraint solver and union find

- Update types in constraint_solver.rs and union_find.rs to use WitnessInfo
- Or: if we created parallel files in Phase 2, delete the old ones

**Test:** All 20 tests pass. No references to old taint types remain.

---

## Phase 6: Downstream Pass Updates

Note: Some of these may be absorbed into earlier phases (e.g., Step 4.2 already handles
the WitnessOf checks). This phase covers any remaining downstream updates.

### Step 6.1: Rework WitnessToRef → WitnessLowering

**File:** `src/compiler/passes/witness_to_ref.rs` (rename to `witness_lowering.rs`)

- Input: `SSA` with WitnessOf types
- Instead of converting `Field[Witness]` → `WitnessRef`, now:
  - `WitnessOf(Field)` values are already typed correctly
  - `Cast { target: WitnessOf }` is lowered here to runtime conversion for AD
  - Instruction lowering (Sub→Add+MulConst, etc.) stays the same
  - Array/tuple conversion loops: check WitnessOf in TypeExpr
- The `Constrain` → `NextDCoeff + BumpD` expansion stays the same

### Step 6.2: Update R1CGen and CodeGen

- Replace `WitnessRef` type checks with `WitnessOf` checks
- `Const::Witness` instead of `Const::WitnessRef`
- VM representation: `WitnessOf(X)` values → witness tape references (same as before)

### Step 6.3: Update minor passes

- WitnessWriteToVoid, WitnessWriteToFresh, PrepareEntryPoint, DCE,
  FixDoubleJumps, Mem2Reg, RCInsertion: pattern match updates for
  removed `V` parameter and WitnessOf instead of WitnessRef

**Test:** All 20 tests pass with full pipeline.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Type explosion from WitnessOf nesting | Low | Medium | Idempotency prevents unbounded nesting |
| Mycroft fixpoint doesn't converge | Very Low | High | Lattice height is finite and small; add iteration counter with panic |
| Cast insertion misses a boundary | Medium | High | Add type-checking verification pass after monomorphization |
| Pattern match exhaustiveness | Low | Low | Rust compiler catches this; add WitnessOf arms incrementally |
| Performance regression from deeper types | Low | Medium | WitnessOf adds one Box per level; profile if needed |
| Downstream passes break | Medium | Medium | Staged migration; keep old code available on branch |

## Estimated Complexity

| Phase | Estimated Effort | Description |
|-------|-----------------|-------------|
| Phase 1 | Medium | Type system changes, many pattern match updates |
| Phase 2 | High | New inference pass, Mycroft fixpoint implementation |
| Phase 3 | High | Monomorphization rewrite + cast insertion |
| Phase 4 | Medium | UntaintControlFlow adaptation |
| Phase 5 | Low-Medium | Cleanup and removal of old code |
| Phase 6 | Medium | Downstream pass updates |

## Testing Strategy

1. **Unit tests** for join/subtyping lattice operations
2. **Comparison tests** between old TaintAnalysis and new WitnessTypeInference (Phase 2)
3. **Integration tests** at each phase boundary (all 20 noir_tests)
4. **Verification pass** after monomorphization to check type consistency
5. **Regression testing** on the complete pipeline after each phase
