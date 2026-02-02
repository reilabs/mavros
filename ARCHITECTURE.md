# Architecture

This document describes the architecture of Mavros - a Noir-to-R1CS compiler and witness generation engine for the [Spartan proof system](https://eprint.iacr.org/2019/550).

## Project Purpose

This project compiles [Noir](https://noir-lang.org/) programs into everything the Spartan proof system needs:

1. **R1CS constraint matrices** (A, B, C) where constraints have the form `Aw ∘ Bw = Cw`
2. **Witness generation program** (witgen) - computes the witness vector `w` and evaluates `Aw`, `Bw`, `Cw`
3. **Automatic differentiation program** (AD) - computes random linear combinations of R1CS rows needed by Spartan's sumcheck protocol

## Compilation Stages

### Stage 1: Noir Frontend

We use the Noir compiler (from a custom branch: `reilabs/noir/tree/spartan-vm`) to parse and type-check Noir source code, producing Noir's SSA representation. We stop at an early SSA stage before Noir's own optimization passes run.

We then convert Noir's SSA to our own SSA representation (`compiler/ssa_gen/`). Our SSA uses mostly immutable data structures, which makes transformation passes easier to write and reason about, at the cost of some data copying.

### Stage 2: Taint Analysis and Monomorphization

Noir (at the stage we integrate) does not distinguish between values that are always constant across all runs and values that depend on witness elements. This distinction is critical for R1CS compilation:

- **Pure values**: Constants that are the same across all executions → baked into R1CS matrix coefficients
- **Witness values**: Input-dependent values → placed in the witness vector

**Taint analysis** (`taint_analysis.rs`) propagates this information through the program. Starting from main function parameters (which are marked as Witness), we compute the taint of every value in the program.

**Monomorphization** (`monomorphization.rs`) then specializes generic functions based on the taints of their arguments. A function called with all-Pure arguments has to be compiled differently than one called with Witness arguments.

**Untainting control flow** (`untaint_control_flow.rs`) ensures that all branch conditions depend only on Pure values. This is required because the R1CS matrix *shape* is fixed at compile time - different execution paths would require different numbers of constraints, which is impossible. When a branch condition depends on Witness values, this pass transforms the code to evaluate both branches and select the result, converting control flow to data flow.

### Stage 3: Optimization Passes

After monomorphization, we run a series of optimization passes on the full SSA. Key passes include:

| Pass | Purpose |
|------|---------|
| **Mem2Reg** | Promote memory operations to SSA registers |
| **CSE** | Eliminate common subexpressions |
| **Condition Propagation** | Propagate known conditions into branches |
| **DCE** | Remove dead code |
| **Arithmetic Simplifier** | Algebraic simplifications (e.g., `x * 1 → x`) |
| **Specializer** | Inline and specialize functions based on cost analysis |

The **Specializer** pass uses a **symbolic instrumentation engine** (`analysis/instrumenter.rs`) to drive optimization decisions. This engine symbolically executes functions, tracking which values would become pure if a loop were unrolled and estimating the constraint cost for different specializations.

### Stage 4: Explicit Witness Lowering

The **Explicit Witness** pass (`passes/explicit_witness.rs`) lowers operations based on operand taints. The same high-level operation compiles differently depending on whether its operands are Pure or Witness.

For example, multiplication of two Pure values is just direct computation - the result is baked into the R1CS matrix coefficients. But multiplying two Witness values requires allocating a fresh witness variable for the result and emitting a constraint `a × b - result = 0` to enforce correctness.

This pass inserts `WriteWitness` and `Constrain` instructions where needed to bridge the gap between high-level Noir operations and R1CS constraints.

### Stage 5: The Witgen/AD Split

At this point, the pipeline splits into two branches: **witgen** (witness generation) and **R1CS/AD** (constraint generation and automatic differentiation).

The split is driven entirely by how the `WriteWitness` instruction is interpreted:

**Before the split**, `WriteWitness` is **bidirectional**:
- It takes a "hint" value as input (how to compute the witness)
- It returns a "witnessified" value as output (the value now marked as Witness)

**For witgen**, we:
- Ignore the returned witnessified value
- Replace all uses of the return value with the hint input

**For R1CS/AD**, we:
- Drop the hint input entirely
- Keep only the witnessified output

This is driven by selectively calling the **WitnessWriteToFresh** / **WitnessWriteToVoid** passes.

This design allows DCE (dead code elimination) to automatically remove operations that are only needed on one side of the pipeline. For example, hint computation logic is removed from the R1CS/AD path, while witgen gets additional opportunities for CSE.

### Stage 6: R1CS Generation and Further Optimization

**R1CS generation** (`r1cs_gen.rs`) executes the SSA symbolically and extracts the constraint matrices A, B and C.

### Stage 7: Bytecode Compilation and VM Execution

Finally, we compile the optimized SSA to bytecode (`compiler/codegen/`) and execute it in our VM (`vm/`).

**Witgen execution** produces:
- The witness vector `w`
- The constraint evaluations `Aw`, `Bw`, `Cw`

**AD execution** produces:
- `coeffs · A` - random linear combination of A's rows
- `coeffs · B` - random linear combination of B's rows
- `coeffs · C` - random linear combination of C's rows

## Key Design Decisions

### Automatic Differentiation for Row Linear Combinations

Spartan's sumcheck protocol requires computing random linear combinations of R1CS matrix rows: given random coefficients `coeffs`, we need `coeffs · A`, `coeffs · B`, and `coeffs · C`.

The naive approach would be to materialize the full matrices and then multiply. But for large circuits this is prohibitively expensive in terms of memory consumption.

Our key insight: if you view the constraint evaluation as a function `f(w) = coeffs · A · w` (a scalar), then the gradient `∂f/∂w` equals `coeffs · A` - exactly the row linear combination we need. The same applies to B and C.

This means we can compute row linear combinations by running the constraint evaluation program in reverse-mode automatic differentiation. The AD program has the same structure as the original - it visits the same operations in the same order - but instead of computing witness values, it accumulates derivatives. Each constraint contributes its coefficient to the gradient of its operands.

### Lazy Gradient Accumulation

The `PureToWitnessRefs` pass (`passes/box_fields.rs`) optimizes AD by wrapping field values in heap-allocated boxes that accumulate their derivatives.

**Problem**: If the expression `(a + b)` appears in multiple constraints with coefficients `c1, c2`, naive AD propagates separately:
- Add `c1` to `a`'s gradient, add `c1` to `b`'s gradient
- Add `c2` to `a`'s gradient, add `c2` to `b`'s gradient
- = 4 operations

**Solution**: Boxed values accumulate incoming gradients. On destruction (via GC), the combined gradient propagates once:
- Accumulate `c1 + c2` in the boxed `(a + b)` node
- On destruction: add `(c1 + c2)` to `a`, add `(c1 + c2)` to `b`
- = 2 propagation operations

This is lazy reverse-mode AD - gradient propagation is deferred until values are no longer needed.

### Lookups (LogUp)

The `Lookup` and `DLookup` opcodes handle PLONK-style lookup constraints. We extend Spartan with the [LogUp argument](https://eprint.iacr.org/2022/1530) to support these. `Lookup` is used in witgen; `DLookup` computes the corresponding derivatives for AD. The VM has special handling for these instructions, to reuse yet-unused parts of the
output vector to store intermediate results and then compute the challenge-based values in a final pass after execution is done.