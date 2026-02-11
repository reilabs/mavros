# Type Inference Algorithm

## 1. Overview

The type inference pass replaces the current `TaintAnalysis`. Given an `SSA<Empty>` where all
types are "pure" (no `WitnessOf`), it determines which values should have `WitnessOf`-wrapped
types based on how witness values flow through the program.

**Input:** `SSA<Empty>` (types without WitnessOf)
**Output:** `WitnessTypeAnalysis` — a map from each `(FunctionId, ValueId)` to its inferred
type (which may contain WitnessOf).

## 2. Internal Representation

### WitnessInfo (replaces Taint)

```rust
enum WitnessInfo {
    Pure,                                    // Not wrapped in WitnessOf
    Witness,                                 // Wrapped in WitnessOf
    Variable(TypeVar),                       // Unknown, to be determined
    Join(Box<WitnessInfo>, Box<WitnessInfo>), // Join of two (= Union in current system)
}
```

This is computationally isomorphic to the current `Taint` enum:
- `Pure` = `Constant(ConstantTaint::Pure)`
- `Witness` = `Constant(ConstantTaint::Witness)`
- `Variable` = `Variable(TypeVariable)`
- `Join` = `Union`

The key semantic difference: `Witness` now means "wrapped in WitnessOf" rather than "tainted."

### WitnessType (replaces TaintType)

```rust
enum WitnessType {
    Scalar(WitnessInfo),                         // Field, U(n), Function
    Array(WitnessInfo, Box<WitnessType>),        // Array/Slice: container + element
    Ref(WitnessInfo, Box<WitnessType>),          // Ref: ref itself + element
    Tuple(WitnessInfo, Vec<WitnessType>),        // Tuple: container + fields
}
```

Again isomorphic to `TaintType`:
- `Scalar` = `Primitive`
- `Array` = `NestedImmutable`
- `Ref` = `NestedMutable`
- `Tuple` = `Tuple`

### FunctionWitnessType (replaces FunctionTaint)

```rust
struct FunctionWitnessType {
    parameters: Vec<WitnessType>,
    returns: Vec<WitnessType>,
    value_types: HashMap<ValueId, WitnessType>,
    // NOTE: NO cfg_taint — that's now handled by UntaintControlFlow
    judgements: Vec<Judgement>,
}
```

### Judgement (same structure as current)

```rust
enum Judgement {
    Eq(WitnessInfo, WitnessInfo),
    Le(WitnessInfo, WitnessInfo),   // Le(a, b) means a <: b
}
```

## 3. Constraint Generation

Constraint generation is per-function, traversing blocks in BFS order (same as current
TaintAnalysis). For each instruction, we generate WitnessType assignments and Judgement
constraints.

### Initialization

For each function:
1. Block parameters get fresh type variables at each structural position
2. Function parameters = entry block parameters
3. Return types get fresh type variables
4. Constants get `Pure` at all positions

### Per-instruction rules

These mirror the current taint analysis rules. The key rules:

**Binary arithmetic / comparison:**
```
result_witness_type = join(lhs_witness_type, rhs_witness_type)
```

**Select (ternary):**
```
result = join(cond_type, then_type, else_type)
```

**Alloc:**
```
result = Ref(Pure, fresh_type_for_element)
```

**Store:**
```
Given: store(ptr: Ref(_, inner), value)
Constraint: value <: inner   (deep)
```
No CFG taint constraint — we panic on witness JmpIf, so all stores are unconditional.

**Load:**
```
Given: load(ptr: Ref(ptr_info, inner))
result = inner.with_toplevel(join(inner.toplevel, ptr_info))
```

**ArrayGet:**
```
Given: array_get(arr: Array(arr_info, elem), idx)
result = elem.with_toplevel(join(arr_info, idx.toplevel, elem.toplevel))
```

**ArraySet:**
```
Given: array_set(arr: Array(arr_info, elem), idx, value)
result = Array(arr_info, join(elem, idx_as_type, value))
```

**MkSeq:**
```
result = Array(Pure, join(all element types))
```

**MkTuple:**
```
result = Tuple(Pure, [type_of(elem1), type_of(elem2), ...])
```

**TupleProj:**
```
Given: tuple_proj(t: Tuple(top, [T1, ..., Tn]), i)
result = Ti.with_toplevel(join(top, Ti.toplevel))
```

**Cast / Truncate / Not:**
```
result = same witness type as input
```

**WriteWitness:**
```
Given: write_witness(value: T)
result = T.with_all_positions(Witness)  // everything becomes WitnessOf
```

**Call (static):**
```
Given: call(f, args) -> results
  1. Look up f's FunctionWitnessType (may be a placeholder for SCC members)
  2. Instantiate fresh type variables (same as current instantiate_from)
  3. Add Eq constraints: actual_arg <=> formal_param (bidirectional)
  4. Add Eq constraints: actual_result <=> formal_return (bidirectional)
  5. Import callee's judgements into caller's judgements
```

Note: We use **bidirectional Eq** (not Le) for call site constraints. This is critical
for Ref type soundness: if a Ref is passed to a callee that stores a wider type through
it, Eq ensures the caller's ref type is updated accordingly. For non-Ref types, Eq and Le
produce equivalent results because each call site gets fresh variables.

**Terminators:**

*Return:*
```
For each (actual, declared) return pair:
  Constraint: actual <: declared
```

*Jmp:*
```
For each (passed_value, block_param) pair:
  Constraint: passed_value <: block_param
```

*JmpIf:*
```
If loop entry:
  Constraint: condition.toplevel <: Pure  (loop conditions must be pure)
Else (if-then-else):
  For each merge point parameter:
    Constraint: condition.toplevel <: merge_param.toplevel
  // This propagates the condition's witness-ness to merge point values
```

Note: the condition being WitnessOf at a non-loop JmpIf is NOT rejected during type
inference. We simply propagate the information. The later UntaintControlFlow pass will
panic if it encounters a witness JmpIf.

### Differences from Current TaintAnalysis

1. **No `cfg_taint` tracking.** CFG taint (which blocks run under witness control flow)
   is not inferred here. It's deferred to the modified UntaintControlFlow pass.

2. **No `block_cfg_taints`.** Block-level CFG taint is also deferred.

3. **Store constraint simplified.** No `Le(cfg_taint, inner.toplevel)` constraint for
   stores under witness control flow (since we panic on witness JmpIf).

## 4. Constraint Solving

The constraint solver is largely the same as the current `ConstraintSolver`, operating on
the WitnessInfo lattice:

1. **Simplify Joins algebraically**: `Join(Pure, x) = x`, `Join(Witness, _) = Witness`
2. **Inline equalities**: union-find over type variables
3. **Blow up Le of Join**: `Le(Join(a, b), c)` → `Le(a, c)` + `Le(b, c)`
4. **Simplify Le with constants**: `Le(Pure, _)` is trivially true, `Le(Witness, x)` forces `x = Witness`
5. **Iterate** until fixpoint
6. **Default** unbound variables to Pure

This is exactly the current algorithm with renamed types.

## 5. Mycroft-Style Fixpoint for Recursion

### Background

Mycroft (1984) showed that recursive definitions can be typed by iterating type inference to
a fixpoint. The key insight: for recursive functions, the function's own type is unknown when
analyzing its body, so we start with a "bottom" estimate and refine.

### Polymorphic Signatures and the Witness Seed

The inference produces **polymorphic** function signatures — type variables with constraints,
not concrete types. This matches the current TaintAnalysis approach:

- Each function's signature has type variables at each structural position
- Constraints capture relationships ("if param[0] is Witness, then return is Witness")
- **The inference does NOT know which function is main**
- The "witness seed" (forcing main params to WitnessOf) happens in **Monomorphization**,
  which instantiates the polymorphic signatures with concrete types

For recursive SCCs, the Mycroft-style handling is needed because when function F calls G
(both in the same SCC), G's signature must be available during F's analysis. We handle
this by creating **placeholder signatures** for all SCC members before analyzing any of them,
then solving the combined constraint system.

In practice, the constraint solver's iterative simplification + defaulting (unbound vars → Pure)
gives us the Mycroft "bottom" for free. The solver implicitly computes the least fixpoint.

### Algorithm

```
function infer_all_types(ssa: SSA<Empty>, flow: FlowAnalysis):
    call_graph = flow.get_call_graph()
    sccs = call_graph.get_sccs_in_reverse_topological_order()
    // Reverse topological order = bottom-up: callees before callers.
    // Same as current post-order traversal.

    for scc in sccs:
        if scc.len() == 1 && !scc[0].is_self_recursive():
            // Non-recursive function: single pass suffices
            // (all callees already have final signatures)
            analyze_function(scc[0])
        else:
            // Recursive SCC: create placeholder signatures, analyze all members,
            // solve combined constraints (implicit Mycroft fixpoint)
            analyze_scc(scc)

function analyze_scc(scc: Set<FunctionId>):
    // Phase 1: Create placeholder signatures for all SCC members
    // Each gets a FunctionWitnessType with fresh type variables
    for f in scc:
        placeholder[f] = create_placeholder_signature(f)  // fresh vars, no constraints
        store_signature(f, placeholder[f])

    // Phase 2: Analyze all members
    // Each function's analysis will reference other SCC members' placeholders
    // via instantiate_from, creating inter-linked constraints
    for f in scc:
        analyze_function(f)
        // analyze_function looks up callee signatures from stored signatures
        // For intra-SCC calls, it finds the placeholder and instantiates from it

    // Phase 3: The constraint solver handles the rest
    // When monomorphization later instantiates these functions with concrete types,
    // the solver resolves all inter-linked constraints across the SCC.
    // Unbound variables default to Pure (= Mycroft "bottom"), giving least fixpoint.
```

**Alternative: Explicit Mycroft Iteration for SCCs**

If the combined constraint system proves too complex for the solver, we can fall back to
explicit iteration (described below). This is a strictly more powerful but more complex
approach:

### Definitions

**bottom_signature(f)**: A function signature where all parameter and return WitnessTypes
have `Pure` at every position. This represents the "most optimistic" assumption — that the
function doesn't need any witness types. The fixpoint iteration then widens from here.

```rust
fn bottom_signature(f: &Function) -> FunctionWitnessSignature {
    FunctionWitnessSignature {
        params: f.get_param_types().iter()
            .map(|t| construct_pure_witness_type(t))   // all Pure
            .collect(),
        returns: f.get_return_types().iter()
            .map(|t| construct_pure_witness_type(t))   // all Pure
            .collect(),
    }
}
```

**assumption**: An assumption equates a type variable in the function's constraint set with
a concrete value from the current signature estimate. This is how we feed the "current
guess" of a recursive function's signature into the constraint solver. It works exactly
like the current `ConstraintSolver::add_assumption()` which pushes `Judgement::Eq` for
corresponding taint/witness positions.

**instantiation at call sites**: When analyzing a function body and encountering a `Call`
to another function in the SCC, we look up the callee's current signature estimate and
create fresh type variables for the call. Constraints equate the fresh variables with
both the caller's argument types AND the callee's signature. This ensures each call site
can be specialized independently (the same function may be called with different witness
types from different sites).

### Mycroft Fixpoint for an SCC

```
function mycroft_fixpoint(scc: Set<FunctionId>):
    // Phase 1: Initialize with bottom signatures (all Pure)
    for f in scc:
        signature[f] = bottom_signature(f)

    // Phase 2: Iterate until fixpoint
    // Convergence bound: each scalar position can widen at most once (Pure → Witness).
    // For N total scalar positions across all functions in the SCC, at most N iterations.
    max_iterations = 2 * total_scalar_positions(scc)  // generous safety bound
    for iteration in 0..max_iterations:
        changed = false
        for f in scc:
            // Analyze function body using current signature estimates for SCC members.
            // For functions outside the SCC, their final signatures are already available.
            new_analysis = analyze_function_body(f, signatures)

            // Solve constraints for this function.
            solver = ConstraintSolver::new(new_analysis)

            // Add assumptions: equate the function's inferred param/return type variables
            // with the current signature estimate. This "pins" the signature to the
            // current guess, allowing the solver to propagate from there.
            for (param, sig_param) in zip(new_analysis.params, signature[f].params):
                solver.add_assumption(param, sig_param)
            for (ret, sig_ret) in zip(new_analysis.returns, signature[f].returns):
                solver.add_assumption(ret, sig_ret)
            solver.solve()

            // Extract resolved signature from the solved constraints
            new_signature = extract_signature(solver, new_analysis)

            // Check if signature changed (monotone widening — only grows)
            if new_signature != signature[f]:
                signature[f] = join_signatures(signature[f], new_signature)
                changed = true

        if !changed:
            break  // Fixpoint reached

    // Phase 3: Final analysis with fixpoint signatures
    // Re-analyze each function one more time with the converged signatures
    // to get final value-level types (not just function signatures).
    for f in scc:
        final_analysis = analyze_function_body(f, signatures)
        solve_and_store(f, final_analysis)
```

### Convergence Guarantee

The lattice for each scalar position has height 2:
```
Pure < Witness
```

For a function signature with `N` scalar positions (across all parameters and returns),
the lattice height is `2N`. Since each iteration must widen at least one position (or
reach fixpoint), convergence happens in at most `2N` iterations.

In practice, convergence is much faster (typically 2-3 iterations) because witness-ness
propagates through multiple positions simultaneously.

### Example: Recursive Function

```noir
fn recursive_sum(arr: [Field; N], i: u32) -> Field {
    if i == 0 { return arr[0]; }
    return arr[i] + recursive_sum(arr, i - 1);
}
```

**Iteration 0 (bottom):**
```
recursive_sum: ([Field; N], u32) -> Field   // all Pure
```

Body analysis with main call `recursive_sum(witness_arr, witness_i)`:
```
arr: Array<WitnessOf(Field), N>    (from caller)
i: WitnessOf(u32)                  (from caller)
arr[i]: WitnessOf(Field)           (witness index into witness array)
recursive_sum(arr, i-1): Field     (using bottom estimate — still pure)
arr[i] + recursive_sum: WitnessOf(Field)  (join with witness arr[i])
return: WitnessOf(Field)
```

New signature: `([WitnessOf(Field); N], WitnessOf(u32)) -> WitnessOf(Field)`

**Iteration 1:**
```
recursive_sum: ([WitnessOf(Field); N], WitnessOf(u32)) -> WitnessOf(Field)
```

Body analysis:
```
arr[i]: WitnessOf(Field)
recursive_sum(arr, i-1): WitnessOf(Field)  (using iteration 1 estimate)
arr[i] + recursive_sum: WitnessOf(Field)
return: WitnessOf(Field)
```

Signature unchanged → **fixpoint reached** after 2 iterations.

### Mutual Recursion Example

```
fn f(x: Field) -> Field { return g(x) + 1; }
fn g(y: Field) -> Field { return f(y) * 2; }
```

Both `f` and `g` form an SCC. If called from main with `WitnessOf(Field)`:

**Iteration 0:** `f: Field -> Field`, `g: Field -> Field`
**Iteration 1:** Analyzing f: `g(x)` returns `Field` (bottom), so `f` returns `WitnessOf(Field)` (from x being witness). Similarly for g.
  New: `f: WitnessOf(Field) -> WitnessOf(Field)`, `g: WitnessOf(Field) -> WitnessOf(Field)`
**Iteration 2:** Signatures unchanged → fixpoint.

## 6. Applying Inference Results

After inference, each value has a `WitnessType` that describes its WitnessOf structure.
To convert this back to actual `Type<V>` values:

```rust
fn apply_witness_type(base_type: Type<Empty>, wt: &WitnessType) -> Type<Empty> {
    let inner_type = match (&base_type.expr, wt) {
        (TypeExpr::Field, WitnessType::Scalar(info)) |
        (TypeExpr::U(_), WitnessType::Scalar(info)) => {
            if info.is_witness() {
                Type::witness_of(base_type, Empty)
            } else {
                base_type
            }
        }
        (TypeExpr::Array(inner, size), WitnessType::Array(top_info, inner_wt)) => {
            let inner_applied = apply_witness_type(*inner, inner_wt);
            let arr = Type::array(inner_applied, *size, Empty);
            if top_info.is_witness() {
                Type::witness_of(arr, Empty)
            } else {
                arr
            }
        }
        // ... similar for Tuple, Ref, Slice
    };
    inner_type
}
```

This produces types like `WitnessOf(Field)`, `Array<WitnessOf(Field), 5>`, etc.

## 7. Comparison: Old TaintAnalysis vs New WitnessTypeInference

| Aspect | Old (TaintAnalysis) | New (WitnessTypeInference) |
|--------|---------------------|---------------------------|
| Representation | `TaintType` overlay on `Type<Empty>` | `WitnessType` (same structure, different semantics) |
| Variables | `Taint::Variable(TypeVariable)` | `WitnessInfo::Variable(TypeVar)` |
| Join | `Taint::Union` | `WitnessInfo::Join` |
| CFG taint | Tracked per-block | Not tracked (deferred to UntaintControlFlow) |
| Store constraints | Includes `Le(cfg_taint, inner)` | Only `Le(value, inner)` |
| Recursion | Post-order only (breaks on SCC) | Mycroft fixpoint on SCCs |
| Output | `FunctionTaint` with value_taints | `FunctionWitnessType` with value_types |
| Applied to types | Separate `typify_taint()` in UntaintControlFlow | `apply_witness_type()` produces actual types with WitnessOf |

## 8. Edge Cases and Considerations

### Function pointers
Function pointers are always Pure (no WitnessOf). This is consistent with the current
system where `Function` type always has `Pure` taint.

### Slice length
`slice_len` always returns `u32` (Pure), even for slices with witness elements. This
matches current behavior.

### Constants
All constants are Pure. `WitnessOf` types are only introduced through `WriteWitness`.

### Global variables
Globals are always Pure. `ReadGlobal` produces Pure types.

### Dynamic calls
Not supported — panic (same as current).
