# Pipeline Restructuring

## 1. Complete Pipeline (New)

```
Phase 0: Frontend
    SSA::from_program()                    // Noir AST → SSA (no type parameter)

Phase 1: Simplification (UNCHANGED except V removal)
    Defunctionalize
    PrepareEntryPoint
    RemoveUnreachableFunctions
    RemoveUnreachableBlocks
    MakeStructAccessStatic
    DCE

Phase 2: Witness Type Analysis (REPLACES TaintAnalysis)
    FlowAnalysis                           // Build CFG + call graph
    WitnessTypeInference                   // NEW: infer WitnessOf types

Phase 3: Monomorphization (MODIFIED)
    Monomorphization                       // Specialize functions by WitnessOf signatures
    // Output: SSA with WitnessOf types, but no explicit casts yet
    // Subtyping X < WitnessOf(X) still conceptually holds

Phase 3.5: Cast Insertion (NEW)
    WitnessCastInsertion                   // Insert real Cast(WitnessOf) conversions
    // Output: SSA with WitnessOf types and explicit casts
    // *** Subtyping disappears here — WitnessOf(X) and X are now independent types ***

Phase 4: Control Flow Linearization (FULLY REWORKED UntaintControlFlow)
    UntaintControlFlow                     // Reads WitnessOf types directly
                                           // Handles witness JmpIf → Select, guards stores
                                           // Adds CFG witness parameters (WitnessOf(U(1)))
                                           // Full feature parity with current

Phase 5: Optimization (UNCHANGED except V removal)
    FixDoubleJumps, Mem2Reg, ArithmeticSimplifier, CSE,
    ConditionPropagation, DeduplicatePhis, DCE, PullIntoAssert,
    Specializer, etc.

Phase 6: Explicit Witness (MODIFIED)
    ExplicitWitness                        // Checks WitnessOf types (not ConstantTaint)
    FixDoubleJumps

Phase 7: R1CS Generation (MODIFIED)
    WitnessWriteToFresh                    // Adapted for WitnessOf types
    DCE
    FixDoubleJumps
    R1CGen

Phase 8: Witness Generation (MODIFIED)
    WitnessWriteToVoid                     // Adapted
    DCE
    RCInsertion
    FixDoubleJumps
    CodeGen

Phase 9: AD (MODIFIED)
    WitnessLowering                        // Replaces WitnessToRef, operates on WitnessOf
    RCInsertion
    FixDoubleJumps
    CodeGen
```

## 2. Pass-by-Pass Changes

### 2.1 TypeExpr (ir/type.rs)

**Changes:**
- Add `WitnessOf(Box<Type>)` variant to `TypeExpr`
- Remove `WitnessRef` variant
- Add normalizing constructor `Type::witness_of()` (enforces idempotency)
- Add `is_witness_of()` predicate
- Add `unwrap_witness_of()` helper (returns inner type if WitnessOf, self otherwise)
- Add `strip_all_witness()` helper (recursively removes all WitnessOf wrappers)
- Update `equal_up_to_annotation()`, `as_pure()`, `contains_ptrs()`, `is_heap_allocated()`,
  `get_arithmetic_result_type()`, `calculate_type_size()`, Display impl
- Add `join(a, b) -> Type` function implementing the join lattice operation
- Add `needs_witness_cast(from, to) -> bool` function
- Update CastTarget enum: add `WitnessOf` variant

**Impact:** Touches every file that pattern-matches on TypeExpr (most of the compiler).

### 2.2 SSA instructions (ssa.rs)

**Changes:**
- Remove `PureToWitnessRef` opcode (replaced by `Cast { target: CastTarget::WitnessOf }`)
- Remove `UnboxField` opcode (replaced by identity or cast)
- Add `CastTarget::WitnessOf` variant
- Update `WriteWitness` — its `witness_annotation` field may change or be removed
  (the WitnessOf type is now in the result's type, not a separate annotation)
- `Const::WitnessRef` → `Const::Witness` (rename for clarity)
- `MulConst` — may stay as-is (it's a lowered instruction for AD)

**Impact:** All passes that pattern-match on OpCode.

### 2.3 WitnessTypeInference (NEW: witness_type_inference.rs)

**Replaces:** `taint_analysis.rs`

**Structure:**
```rust
pub struct WitnessTypeInference {
    functions: HashMap<FunctionId, FunctionWitnessType>,
    last_ty_var: usize,
}

pub struct FunctionWitnessType {
    pub parameters: Vec<WitnessType>,
    pub returns: Vec<WitnessType>,
    pub value_types: HashMap<ValueId, WitnessType>,
    pub judgements: Vec<Judgement>,
}
```

**Key methods:**
- `run(ssa, flow_analysis)` — entry point
- `analyze_scc(scc)` — Mycroft fixpoint for an SCC
- `analyze_function(func_id)` — constraint generation for one function
- `solve_constraints(func)` — constraint solving (reuses existing solver logic)

### 2.4 Monomorphization (monomorphization.rs — MODIFIED)

**Changes:**
- `Signature` uses `Vec<Type>` instead of `Vec<TaintType>` + `Taint`
- `request_specialization` clones function and sets up work item
- `specialize_function` resolves constraints and updates types (no cast insertion)
- `monomorphize_main_signature` uses actual types with WitnessOf
- Simpler than current: no taint-specific logic needed

### 2.4.5 WitnessCastInsertion (witness_cast_insertion.rs — NEW)

**Purpose:** After monomorphization, insert explicit `Cast { target: WitnessOf }` operations
at type boundaries. Handles scalar casts, array conversion loops, tuple field-by-field.

**Key methods:**
- `run(ssa, witness_analysis)` — entry point
- `insert_casts(function, type_map)` — per-function cast insertion
- `emit_deep_cast(function, value, from, to)` — compound type conversion
- `emit_array_conversion_loop()` — adapted from WitnessToRef

### 2.5 UntaintControlFlow (untaint_control_flow.rs — FULLY REWORKED)

**Current role:** Convert `SSA<Empty>` → `SSA<ConstantTaint>`, handle witness branches.

**New role:** With `V` eliminated, there is no type annotation conversion. UntaintControlFlow
operates directly on `SSA` with `WitnessOf` types baked into `TypeExpr`. It performs the same
control-flow transformations as the current pass, but reads `WitnessOf` types instead of
`ConstantTaint` annotations.

**Input:** `SSA` (with WitnessOf types, after monomorphization + cast insertion)
**Output:** `SSA` (with WitnessOf types, witness branches linearized)

**Algorithm:**

1. **Determine block CFG witness-ness.** Walk the CFG and identify which blocks are
   dominated by a witness-conditional JmpIf. A block is "witness-conditional" if it is
   only reachable through a JmpIf whose condition has a `WitnessOf` type.

2. **For each witness JmpIf**, apply the same linearization as the current pass:
   - Replace `JmpIf(cond, then_block, else_block)` with `Jmp(merge_block)`
   - At the merge block, insert `Select(cond, then_val, else_val)` for each phi/block
     parameter that differs between branches
   - The Select result type is the join of the branch types (which will contain WitnessOf
     since the condition is WitnessOf)

3. **Guard stores** in witness-conditional blocks:
   ```
   // Original: store(ref, new_value)  [in witness-conditional block]
   // Becomes:
   old_value = load(ref)
   guarded = select(cfg_witness_flag, new_value, old_value)
   store(ref, guarded)
   ```
   This ensures stores under witness branches don't unconditionally overwrite.

4. **Guard assert_eq** in witness-conditional blocks:
   ```
   // Original: assert_eq(a, b)  [in witness-conditional block]
   // Becomes:  assert_eq(select(cfg_flag, a, b), b)
   ```

5. **Add CFG witness parameters.** When blocks merge after a witness-conditional region,
   the "which branch was taken" flag is threaded as a block parameter of type
   `WitnessOf(U(1))`. This replaces the current `u(1, ConstantTaint::Witness)` parameter.

**Key difference from current pass:** No type annotation conversion (`Type<Empty>` →
`Type<ConstantTaint>`) because `V` is eliminated. The pass just transforms control flow,
and types already contain `WitnessOf` where needed. Downstream passes (ExplicitWitness,
WitnessLowering) check `WitnessOf` in `TypeExpr` directly.

**Helper: is_witness_type**
```rust
fn is_witness_type(typ: &Type) -> bool {
    matches!(&typ.expr, TypeExpr::WitnessOf(_))
}
```

### 2.6 ExplicitWitness (passes/explicit_witness.rs — MODIFIED)

**Current:** Checks `ConstantTaint` annotations to decide transformations.

**After this refactor:** With `V` eliminated, ExplicitWitness checks `WitnessOf` in `TypeExpr`
directly instead of `ConstantTaint` annotations. Wherever the current code checks
`annotation == ConstantTaint::Witness`, the new code checks `matches!(type.expr, TypeExpr::WitnessOf(_))`.

**Key changes:**
- Replace `type.annotation.is_witness()` → `type.is_witness_of()`
- Replace `Type::witness_ref()` → `Type::witness_of(Type::field())`
- WriteWitness handling: result type is `WitnessOf(input_type)` (already in TypeExpr)
- Constrain/AssertEq checks: look at `WitnessOf` in operand types

### 2.7 WitnessLowering (replaces passes/witness_to_ref.rs — MODIFIED)

**Current WitnessToRef:**
- Converts `Field[Witness]` → `WitnessRef`
- Inserts `PureToWitnessRef` operations
- Lowers `Sub(wit,wit)` → `Add(a, MulConst(-1,b))`
- Generates array conversion loops
- Handles `Constrain` → `NextDCoeff + BumpD` expansion

**New WitnessLowering:**
- Checks `WitnessOf` in `TypeExpr` instead of `ConstantTaint::Witness` annotation
- `WitnessOf(Field)` values are already correctly typed — no type conversion needed
- `PureToWitnessRef` is gone (replaced by `Cast { target: WitnessOf }` earlier in pipeline)
- `Cast { target: WitnessOf }` is lowered here to runtime conversion for AD
- Sub/Mul/Constrain lowering stays the same
- Array conversion loops now work with WitnessOf types

### 2.8 R1CGen and CodeGen (MINOR CHANGES)

- Pattern matches on `WitnessRef` → `WitnessOf`
- `Const::WitnessRef` → `Const::Witness`
- `is_witness_ref()` → `is_witness_of()`
- VM representation: values of type `WitnessOf(X)` are represented as witness
  references (pointers into the witness tape) at runtime, same as current `WitnessRef`.

### 2.9 Constraint Solver (constraint_solver.rs — MINOR RENAME)

The constraint solver operates on `WitnessInfo` instead of `Taint`, but the algorithm is
identical. Changes are mostly renames:
- `Taint` → `WitnessInfo`
- `TaintType` → `WitnessType`
- `ConstantTaint::Pure/Witness` → `WitnessInfo::Pure/Witness`

### 2.10 Union Find (union_find.rs — MINOR RENAME)

Same algorithm, renamed types:
- `taint_mapping: HashMap<TypeVariable, ConstantTaint>` →
  `witness_mapping: HashMap<TypeVariable, WitnessStatus>` (Pure/Witness enum)

## 3. Files to Create

| File | Description |
|------|-------------|
| `src/compiler/witness_type_inference.rs` | New type inference pass |
| `src/compiler/witness_info.rs` | WitnessInfo, WitnessType, Judgement types |
| `src/compiler/witness_cast_insertion.rs` | New cast insertion pass |

## 4. Files to Delete

| File | Reason |
|------|--------|
| `src/compiler/taint_analysis.rs` | Replaced by witness_type_inference.rs |

## 5. Files to Modify (Major)

| File | Changes |
|------|---------|
| `src/compiler/ir/type.rs` | Add WitnessOf, remove WitnessRef, add join() |
| `src/compiler/ssa.rs` | Remove PureToWitnessRef/UnboxField, add CastTarget::WitnessOf |
| `src/compiler/monomorphization.rs` | Use Type-based signatures (no cast insertion — separate pass) |
| `src/compiler/untaint_control_flow.rs` | Fully rework: read WitnessOf types, handle witness branches, no V conversion |
| `src/compiler/passes/witness_to_ref.rs` | Rework to WitnessLowering with WitnessOf types |
| `src/compiler/constraint_solver.rs` | Rename types (Taint→WitnessInfo) |
| `src/compiler/union_find.rs` | Rename types |
| `src/driver.rs` | Update pipeline orchestration |

## 6. Files to Modify (Minor — pattern match updates)

| File | Changes |
|------|---------|
| `src/compiler/passes/explicit_witness.rs` | WitnessRef→WitnessOf in type checks |
| `src/compiler/passes/witness_write_to_void.rs` | Minor type adjustments |
| `src/compiler/passes/witness_write_to_fresh.rs` | Minor type adjustments |
| `src/compiler/passes/prepare_entry_point.rs` | Minor type adjustments |
| `src/compiler/passes/dce.rs` | Pattern match updates |
| `src/compiler/passes/fix_double_jumps.rs` | Pattern match updates |
| `src/compiler/passes/mem2reg.rs` | Pattern match updates |
| `src/compiler/passes/rc_insertion.rs` | Pattern match updates |
| `src/compiler/r1cs/` | WitnessRef→WitnessOf in R1CS generation |
| `src/compiler/codegen/` | WitnessRef→WitnessOf in code generation |
| `src/vm/` | Const::WitnessRef→Const::Witness |

## 7. Migration Strategy

The changes can be staged to keep the codebase compiling at each step:

1. **Add WitnessOf to TypeExpr** (alongside WitnessRef): both exist temporarily
2. **Build WitnessTypeInference**: can coexist with TaintAnalysis
3. **Update Monomorphization**: switch to Type-based signatures
4. **Rework UntaintControlFlow**: linearize witness branches using WitnessOf types
5. **Remove WitnessRef, TaintAnalysis**: clean up old code
6. **Update downstream passes**: ExplicitWitness, WitnessLowering, etc.

See [05-implementation-plan.md](05-implementation-plan.md) for the detailed order.
