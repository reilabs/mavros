# Types Refactor: WitnessOf(X) — Overview

## Motivation

The current system uses a separate "taint" layer (`ConstantTaint`, `TaintType`) overlaid on
top of the SSA type system (`Type<V>`) to track which values depend on witness inputs. This
leads to a two-tier type system where taint information is disconnected from the actual types,
requiring complex passes to reconcile them (UntaintControlFlow, WitnessToRef).

The new design integrates witness tracking directly into the type system via a `WitnessOf(X)`
type constructor, making witness-dependence a first-class property of types rather than an
external annotation.

## Key Ideas

1. **`WitnessOf(X)`** is a new type constructor: `WitnessOf(Field)` means "a witness
   observation of a field element." The subtyping relation `X < WitnessOf(X)` means any
   pure value can be used where a witness is expected (implicit upcast).

2. **`WriteWitness` is typed as `X -> WitnessOf(X)`**: it promotes a pure value into the
   witness domain. There is no inverse operation.

3. **Type inference replaces taint analysis**: instead of a separate constraint system over
   taints, we re-infer types under the subtyping relation `X < WitnessOf(X)`. Since main
   parameters are already wrapped with `WriteWitness`, witness types propagate naturally.

4. **Monomorphization specializes functions**: duplicates function definitions based on
   their WitnessOf-augmented type signatures, as the current system does with taint
   signatures.

5. **A separate cast insertion pass makes subtyping explicit**: after monomorphization,
   `WitnessCastInsertion` inserts explicit `Cast` operations where a value of type `X`
   is used in a context requiring `WitnessOf(X)` (e.g., storing into an array typed
   `Array<WitnessOf(Field)>`, passing arguments to specialized functions).

6. **Control-flow untainting is deferred**: for now, we panic if a `JmpIf` condition has a
   `WitnessOf` type. A separate pass (modified `UntaintControlFlow`) handles CFG taint
   tracking and function taint parameters.

## What Changes

| Component | Current | New |
|-----------|---------|-----|
| Witness tracking | Separate `TaintType` overlay | `WitnessOf(X)` in `TypeExpr` |
| Type annotation `V` | `Empty` / `ConstantTaint` | Kept for now (minimize blast radius) |
| `WitnessRef` type | Separate `TypeExpr::WitnessRef` | Replaced by `WitnessOf(X)` |
| Taint analysis | `taint_analysis.rs` | New `witness_type_inference.rs` |
| Constraint solver | Union-find over `Taint` variables | Similar, but over WitnessOf type lattice |
| Monomorphization | Specializes by `TaintType` signature | Specializes by `Type` signature (with WitnessOf); no casts |
| WitnessCastInsertion | (did not exist) | NEW: inserts explicit Cast at type boundaries after monomorphization |
| UntaintControlFlow | Converts `SSA<Empty>` to `SSA<ConstantTaint>`, handles witness branches | Simplified: verifies no witness JmpIf, infers CFG taint, adds taint params |
| ExplicitWitness | Checks `ConstantTaint` annotations | Checks for `WitnessOf` in types |
| WitnessToRef | Converts witness types to `WitnessRef` | Reworked: lowering pass operates on `WitnessOf` types |
| Cast opcode | `Field <-> U(n)` | Extended with `WitnessOf` target |
| `PureToWitnessRef` | Separate opcode | Replaced by `Cast` with WitnessOf target |

## Pipeline (Before vs After)

### Current Pipeline
```
SSA<Empty>
  -> Defunctionalize -> PrepareEntryPoint -> RemoveUnreachable -> MakeStructAccessStatic -> DCE
  -> FlowAnalysis -> TaintAnalysis -> Monomorphization -> UntaintControlFlow
     (produces SSA<ConstantTaint>)
  -> ExplicitWitness -> ... -> WitnessToRef -> R1CS / CodeGen
```

### New Pipeline
```
SSA<Empty>
  -> Defunctionalize -> PrepareEntryPoint -> RemoveUnreachable -> MakeStructAccessStatic -> DCE
  -> FlowAnalysis -> WitnessTypeInference -> Monomorphization (specialization only)
  -> WitnessCastInsertion (insert explicit Cast ops at type boundaries)
  -> UntaintControlFlow (CFG taint + verification; produces SSA<ConstantTaint>)
  -> ExplicitWitness (checks WitnessOf types) -> ... -> WitnessLowering -> R1CS / CodeGen
```

## Document Index

- [01-type-system.md](01-type-system.md) — WitnessOf type system design and subtyping rules
- [02-type-inference.md](02-type-inference.md) — Type inference algorithm (Mycroft-style fixpoint)
- [03-monomorphization.md](03-monomorphization.md) — Monomorphization and cast insertion
- [04-pipeline.md](04-pipeline.md) — Pipeline restructuring and pass-by-pass changes
- [05-implementation-plan.md](05-implementation-plan.md) — Step-by-step implementation order
