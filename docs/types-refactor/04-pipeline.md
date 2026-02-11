# Pipeline Restructuring

## 1. Complete Pipeline (New)

```
Phase 0: Frontend
    SSA::from_program()                    // Noir AST → SSA<Empty>

Phase 1: Simplification (UNCHANGED)
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
    // Output: SSA<Empty> with WitnessOf types, but no explicit casts yet

Phase 3.5: Cast Insertion (NEW)
    WitnessCastInsertion                   // Insert explicit Cast ops where X used as WitnessOf(X)
    // Output: SSA<Empty> with WitnessOf types and explicit casts

Phase 4: Control Flow Analysis (MODIFIED UntaintControlFlow)
    UntaintControlFlow                     // Verify no witness JmpIf, compute CFG taint,
                                           // add function taint params, convert to SSA<ConstantTaint>

Phase 5: Optimization (UNCHANGED)
    FixDoubleJumps, Mem2Reg, ArithmeticSimplifier, CSE,
    ConditionPropagation, DeduplicatePhis, DCE, PullIntoAssert,
    Specializer, etc.

Phase 6: Explicit Witness (MODIFIED)
    ExplicitWitness                        // Check WitnessOf types instead of annotations
    FixDoubleJumps

Phase 7: R1CS Generation (MINOR CHANGES)
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
    WitnessLowering                        // Replaces WitnessToRef
    RCInsertion
    FixDoubleJumps
    CodeGen
```

## 2. Pass-by-Pass Changes

### 2.1 TypeExpr (ir/type.rs)

**Changes:**
- Add `WitnessOf(Box<Type<V>>)` variant to `TypeExpr`
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
- `Signature` uses `Vec<Type<Empty>>` instead of `Vec<TaintType>` + `Taint`
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

### 2.5 UntaintControlFlow (untaint_control_flow.rs — MODIFIED)

**Current role:** Convert SSA<Empty> → SSA<ConstantTaint>, handle witness branches.

**New role:**
1. Walk the SSA, which now has WitnessOf types baked in
2. For each JmpIf, check if condition type contains WitnessOf → **panic**
3. Compute ConstantTaint annotations from WitnessOf types:
   - `WitnessOf(_)` → `ConstantTaint::Witness`
   - everything else → `ConstantTaint::Pure`
4. Add function taint parameters where CFG taint is Witness
   (for now: never, since we panic on witness JmpIf)
5. Convert `SSA<Empty>` with WitnessOf types → `SSA<ConstantTaint>`

**Type conversion:**
```rust
fn type_to_annotated(typ: &Type<Empty>) -> Type<ConstantTaint> {
    match &typ.expr {
        TypeExpr::WitnessOf(inner) => {
            let inner_annotated = type_to_annotated(inner);
            Type {
                expr: inner_annotated.expr,  // strip WitnessOf from TypeExpr
                annotation: ConstantTaint::Witness,
            }
        }
        TypeExpr::Field => Type::field(ConstantTaint::Pure),
        TypeExpr::U(n) => Type::u(*n, ConstantTaint::Pure),
        // ... recursively handle Array, Tuple, etc.
    }
}
```

Wait — this loses information. If we have `Array<WitnessOf(Field), 5>`, converting to
`ConstantTaint` would give `Array<Field[Witness], 5>[Pure]`. But `WitnessOf(Array<Field, 5>)`
would give `Array<Field[Pure], 5>[Witness]`. These are distinct, which is correct.

However, `WitnessOf(Array<WitnessOf(Field), 5>)` → `Array<Field[Witness], 5>[Witness]`.
This preserves the information at both levels.

**Key insight:** The conversion from `WitnessOf`-based types to `ConstantTaint`-annotated
types is lossless because ConstantTaint at each structural level captures exactly whether
that level was wrapped in WitnessOf.

### 2.6 ExplicitWitness (passes/explicit_witness.rs — MODIFIED)

**Current:** Checks `ConstantTaint` annotations to decide transformations.

**After this refactor:** Still checks `ConstantTaint` annotations (because UntaintControlFlow
converts WitnessOf types to ConstantTaint-annotated types). So **minimal changes needed** in
this pass — the types it sees are in the same format as before.

The main change: `WitnessRef` references become `WitnessOf` references in type checks.

### 2.7 WitnessLowering (replaces passes/witness_to_ref.rs — MODIFIED)

**Current WitnessToRef:**
- Converts `Field[Witness]` → `WitnessRef`
- Inserts `PureToWitnessRef` operations
- Lowers `Sub(wit,wit)` → `Add(a, MulConst(-1,b))`
- Generates array conversion loops
- Handles `Constrain` → `NextDCoeff + BumpD` expansion

**New WitnessLowering:**
- Instead of converting to `WitnessRef`, converts `WitnessOf(Field)` types
  (now checking for `WitnessOf` in TypeExpr instead of `ConstantTaint::Witness`)
- `PureToWitnessRef` is gone (replaced by Cast with WitnessOf earlier)
- `Cast { target: WitnessOf }` is lowered here if needed for AD
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
| `src/compiler/monomorphization.rs` | Use Type-based signatures, insert casts |
| `src/compiler/untaint_control_flow.rs` | Convert WitnessOf→ConstantTaint, verify no witness JmpIf |
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
4. **Update UntaintControlFlow**: convert WitnessOf→ConstantTaint
5. **Remove WitnessRef, TaintAnalysis**: clean up old code
6. **Update downstream passes**: ExplicitWitness, WitnessLowering, etc.

See [05-implementation-plan.md](05-implementation-plan.md) for the detailed order.
