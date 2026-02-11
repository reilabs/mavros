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
   observation of a field element." During type inference and monomorphization, the
   subtyping relation `X < WitnessOf(X)` holds, allowing implicit upcasts.

2. **After monomorphization + cast insertion, subtyping disappears.** `WitnessOf(X)` and `X`
   become fully independent types with no implicit conversion. `Cast(WitnessOf)` is a real
   runtime conversion (like current `PureToWitnessRef`), not a no-op.

3. **`WriteWitness` is typed as `X -> WitnessOf(X)`**: it promotes a pure value into the
   witness domain. There is no inverse operation.

4. **Type inference replaces taint analysis**: instead of a separate constraint system over
   taints, we re-infer types under the subtyping relation `X < WitnessOf(X)`. Since main
   parameters are already wrapped with `WriteWitness`, witness types propagate naturally.

5. **Monomorphization specializes functions**: duplicates function definitions based on
   their WitnessOf-augmented type signatures, as the current system does with taint
   signatures.

6. **A separate cast insertion pass makes subtyping explicit**: after monomorphization,
   `WitnessCastInsertion` inserts explicit `Cast` operations where a value of type `X`
   is used in a context requiring `WitnessOf(X)`. These are real conversions, not type
   annotations.

7. **The `V` annotation parameter is eliminated.** Types become plain `TypeExpr` (no
   generic annotation). `WitnessOf` in `TypeExpr` carries all witness information.
   `Empty`, `ConstantTaint`, and the `CommutativeMonoid` trait are removed.

8. **UntaintControlFlow is fully reworked.** It reads WitnessOf types directly (no
   ConstantTaint conversion), handles witness-dependent branches (JmpIf→Select, guarded
   stores), and adds CFG witness parameters — full feature parity with current.

## What Changes

| Component | Current | New |
|-----------|---------|-----|
| Witness tracking | Separate `TaintType` overlay | `WitnessOf(X)` in `TypeExpr` |
| Type annotation `V` | `Empty` / `ConstantTaint` | **Eliminated** — types have no annotation parameter |
| `WitnessRef` type | Separate `TypeExpr::WitnessRef` | Replaced by `WitnessOf(X)` |
| `SSA<V>` | Generic over annotation type | Plain `SSA` (no type parameter) |
| Taint analysis | `taint_analysis.rs` | New `witness_type_inference.rs` |
| Constraint solver | Union-find over `Taint` variables | Similar, but over WitnessOf type lattice |
| Monomorphization | Specializes by `TaintType` signature | Specializes by `Type` signature (with WitnessOf) |
| WitnessCastInsertion | (did not exist) | NEW: inserts real Cast conversions at type boundaries |
| UntaintControlFlow | Converts `SSA<Empty>` to `SSA<ConstantTaint>`, handles witness branches | Fully reworked: reads WitnessOf types, handles witness branches, adds CFG witness params |
| ExplicitWitness | Checks `ConstantTaint` annotations | Checks for `WitnessOf` in types |
| WitnessToRef | Converts witness types to `WitnessRef` | Reworked: lowering pass operates on `WitnessOf` types |
| Cast opcode | `Field <-> U(n)` | Extended with `WitnessOf` target (real conversion) |
| `PureToWitnessRef` | Separate opcode | Replaced by `Cast` with WitnessOf target |
| Subtyping `X < WitnessOf(X)` | N/A | **Only exists during inference and monomorphization.** Gone after cast insertion. |

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
SSA  (no type parameter — WitnessOf is in TypeExpr)
  -> Defunctionalize -> PrepareEntryPoint -> RemoveUnreachable -> MakeStructAccessStatic -> DCE
  -> FlowAnalysis -> WitnessTypeInference -> Monomorphization (specialization only)
  -> WitnessCastInsertion (insert real Cast conversions — subtyping disappears here)
  -> UntaintControlFlow (reads WitnessOf types, handles witness branches, adds CFG params)
  -> ExplicitWitness (checks WitnessOf types) -> ... -> WitnessLowering -> R1CS / CodeGen
```

## Document Index

- [01-type-system.md](01-type-system.md) — WitnessOf type system design and subtyping rules
- [02-type-inference.md](02-type-inference.md) — Type inference algorithm (Mycroft-style fixpoint)
- [03-monomorphization.md](03-monomorphization.md) — Monomorphization and cast insertion
- [04-pipeline.md](04-pipeline.md) — Pipeline restructuring and pass-by-pass changes
- [05-implementation-plan.md](05-implementation-plan.md) — Step-by-step implementation order
