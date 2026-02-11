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

### Step 1.1: Add WitnessOf to TypeExpr

**File:** `src/compiler/ir/type.rs`

- Add `WitnessOf(Box<Type<V>>)` variant to `TypeExpr<V>`
- Add normalizing constructor `Type::witness_of(inner, annotation)` (enforces idempotency)
- Add helper methods:
  - `is_witness_of() -> bool`
  - `unwrap_witness_of() -> &Type<V>` (returns inner if WitnessOf, panics otherwise)
  - `try_unwrap_witness_of() -> Option<&Type<V>>`
  - `strip_witness() -> Type<V>` (strips one level of WitnessOf)
  - `strip_all_witness() -> Type<V>` (recursively strips all WitnessOf)
- Update ALL pattern matches on TypeExpr to handle WitnessOf:
  - `equal_up_to_annotation`: WitnessOf(X) == WitnessOf(Y) iff X == Y
  - `as_pure`: WitnessOf(inner) → WitnessOf(inner.as_pure())
  - `contains_ptrs`: WitnessOf(inner) → inner.contains_ptrs()
  - `get_arithmetic_result_type`: delegate through WitnessOf
  - `is_numeric`: WitnessOf(numeric) → true
  - `is_heap_allocated`: WitnessOf(_) → true (witness values live on tape)
  - `calculate_type_size`: WitnessOf(inner) → 1 (pointer-sized)
  - Display: `WitnessOf(Field)`, `WitnessOf(Array<...>)`, etc.
- Keep `WitnessRef` variant for now (coexistence)

**Test:** Compiles, existing tests pass (WitnessOf not yet used anywhere).

### Step 1.2: Add CastTarget::WitnessOf

**File:** `src/compiler/ssa.rs`

- Add `WitnessOf` variant to `CastTarget` enum
- Update Cast handling in all passes to handle the new variant
  (most passes can just pass it through unchanged)

**Test:** Compiles, existing tests pass.

### Step 1.3: Add join() operation for types

**File:** `src/compiler/ir/type.rs`

- Implement `Type::join(a: &Type<V>, b: &Type<V>) -> Type<V>` as described in
  [01-type-system.md](01-type-system.md) section 3
- Implement `Type::is_subtype_of(a, b) -> bool` for checking subtyping relation
- Add unit tests for join and subtyping

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
- Implement conversion: `WitnessType` → `TaintType` (for compatibility during migration)

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
  - Same logic as current `TaintAnalysis::analyze_function()` but:
    - No `cfg_taint` tracking
    - No `block_cfg_taints`
    - No `Le(cfg_taint, inner)` on Store
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

## Phase 3: Monomorphization Update

### Step 3.1: Switch Monomorphization to Type-based signatures

**File:** `src/compiler/monomorphization.rs`

- Change `Signature` to use `Vec<Type<Empty>>` for params/returns
- Update `monomorphize_main_signature` to produce WitnessOf types
- Update `specialize_function` to use WitnessTypeInference results
- Keep using current ConstraintSolver for resolving per-function constraints
  (but fed with WitnessType data instead of TaintType)

### Step 3.2: Create WitnessCastInsertion pass

**File:** `src/compiler/witness_cast_insertion.rs` (NEW)

A separate pass that runs after monomorphization. Walks specialized functions and inserts
explicit `Cast { target: WitnessOf }` where type mismatches exist.

- `run(ssa, witness_analysis)` — entry point
- `insert_casts(function, type_map)` — per-function, walks instructions + terminators
- `needs_witness_cast(actual, expected)` — recursive type comparison
- `emit_scalar_cast(value)` → single Cast instruction
- `emit_array_cast(function, value, from_elem, to_elem)` → conversion loop
  (adapted from `WitnessToRef::emit_array_conversion_loop()`)
- `emit_tuple_cast(function, value, from_fields, to_fields)` → project + cast + mk_tuple

Note: `WitnessOf(Array<...>)` as a cast target is not handled — panic with clear error.

### Step 3.3: Switch driver to use new Monomorphization + CastInsertion

**File:** `src/driver.rs`

- Replace TaintAnalysis call with WitnessTypeInference
- Feed new Monomorphization with WitnessType results
- At this point, the SSA after monomorphization has WitnessOf types in TypeExpr

**Test:** Pipeline produces correct SSA (may need to temporarily bridge to old
UntaintControlFlow).

---

## Phase 4: UntaintControlFlow Update

### Step 4.1: Update UntaintControlFlow for WitnessOf types

**File:** `src/compiler/untaint_control_flow.rs`

- Input: `SSA<Empty>` where types contain WitnessOf
- Walk each function:
  1. For each value, convert its type: WitnessOf(X) → X with annotation Witness;
     plain X → X with annotation Pure
  2. For each JmpIf, check condition type: if WitnessOf, **panic**
  3. Determine CFG taint (for now: always Pure because we panicked on witness JmpIf)
  4. No function taint parameters needed (for now)
- Output: `SSA<ConstantTaint>` where types have ConstantTaint annotations

Key function:
```rust
fn convert_type(typ: &Type<Empty>) -> Type<ConstantTaint> {
    match &typ.expr {
        TypeExpr::WitnessOf(inner) => {
            // Recursively convert inner, then mark outer as Witness
            let inner_conv = convert_type_inner(inner);
            Type { expr: inner_conv.expr, annotation: ConstantTaint::Witness }
        }
        TypeExpr::Field => Type::field(ConstantTaint::Pure),
        TypeExpr::U(n) => Type::u(*n, ConstantTaint::Pure),
        TypeExpr::Array(inner, size) => {
            let inner_conv = convert_type(inner);
            Type {
                expr: TypeExpr::Array(Box::new(inner_conv), *size),
                annotation: ConstantTaint::Pure,
            }
        }
        // ... etc.
    }
}
```

### Step 4.2: Remove TaintAnalysis dependency from UntaintControlFlow

- UntaintControlFlow no longer needs `TaintAnalysis` as input
- It derives ConstantTaint annotations directly from WitnessOf types

**Test:** Full pipeline works. All 20 tests pass.

---

## Phase 5: Cleanup

### Step 5.1: Remove old TaintAnalysis

- Delete `src/compiler/taint_analysis.rs`
- Remove all references to `TaintType`, `FunctionTaint`, `Taint` enum
- Remove the validation harness from Phase 2

### Step 5.2: Remove WitnessRef from TypeExpr

- Delete `TypeExpr::WitnessRef` variant
- Update all pattern matches (will be compiler-guided via exhaustive match)
- Rename `Const::WitnessRef` → `Const::Witness`

### Step 5.3: Remove PureToWitnessRef opcode

- Delete `OpCode::PureToWitnessRef` variant
- All uses should have been replaced by `Cast { target: WitnessOf }`
- Update all pattern matches

### Step 5.4: Clean up UnboxField

- Delete `OpCode::UnboxField` variant
- Replace uses with appropriate Cast or identity

### Step 5.5: Rename/clean up constraint solver and union find

- Update types in constraint_solver.rs and union_find.rs to use WitnessInfo
- Or: if we created parallel files in Phase 2, delete the old ones

**Test:** All 20 tests pass. No references to old taint types remain.

---

## Phase 6: Downstream Pass Updates

### Step 6.1: Update ExplicitWitness

**File:** `src/compiler/passes/explicit_witness.rs`

- Replace `is_witness_ref()` checks with WitnessOf-aware checks
- Since this pass operates on `SSA<ConstantTaint>` (after UntaintControlFlow), it
  still uses ConstantTaint annotations. **Minimal changes needed.**
- Main change: where it currently checks for `WitnessRef` type, check for WitnessOf.

### Step 6.2: Rework WitnessToRef → WitnessLowering

**File:** `src/compiler/passes/witness_to_ref.rs` (rename to `witness_lowering.rs`)

- Input: `SSA<ConstantTaint>` with WitnessOf types
- Instead of converting `Field[Witness]` → `WitnessRef`, now:
  - `WitnessOf(Field)` values are already typed correctly
  - Cast operations (`Cast { target: WitnessOf }`) are lowered here
  - Instruction lowering (Sub→Add+MulConst, etc.) stays the same
  - Array/tuple conversion loops: adapt to check WitnessOf instead of ConstantTaint
- The `Constrain` → `NextDCoeff + BumpD` expansion stays the same

### Step 6.3: Update R1CGen and CodeGen

- Replace `WitnessRef` type checks with `WitnessOf` checks
- `Const::Witness` instead of `Const::WitnessRef`
- VM representation: `WitnessOf(X)` values → witness tape references (same as before)

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
