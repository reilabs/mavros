# Architecture

This document aims to serve as a description of Mavros' architecture and act as a guide for where to
look for certain parts of the pipeline. Mavros is a Noir-to-R1CS compiler and witness generation
engine for the [spartan proof system](https://eprint.iacr.org/2019/550).

## Project Purpose

Mavros compiles [Noir](https://noir-lang.org/) for [spartan](https://github.com/microsoft/Spartan),
a high performance ZK proof system. Mavros aims to generate all of the components that Spartan
requires:

1. **R1CS constraint matrices** ($A$, $B$, $C$) where constraints have the form $Aw \circ Bw = Cw$.
2. The **witness generation program** (witgen) that computes the witness vector $w$ and evaluates
   $Aw$, $Bw$, and $Cw$.
3. An **automatic differentiation program** (AD) that computes the random linear combinations of
   R1CS rows needed by Spartan's sumcheck protocol.

## Key Design Decisions

Mavros implements an [enhanced](#key-design-decisions) version of Spartan that has been augmented
with support for the [LogUp](#logup-lookups) lookup argument, and improved performance as described
below.

### Automatic Differentiation for Row Linear Combinations

The [sum-check](https://en.wikipedia.org/wiki/Sum-check_protocol) performed as part of Spartan
requires computing random linear combinations of rows of the R1CS matrix. Given random coefficients
$c_r$, we need $c_r \cdot A$, $c_r \cdot B$ and $c_r \cdot C$. The naïve approach to this would
require materializing the full matrices and then multiplying them out, but this is prohibitively
expensive memory-wise for large circuits.

**Mavros' key insight** is that if you view the constraint evaluation as a function
$f(w) = c_r \cdot A \cdot w$ (a scalar), then the gradient $\delta f / \delta w$ is equal to
`c_r \cdot a`. This is _exactly_ the row linear combination that we need. The same applies to $B$
and $C$.

As a result, we can compute the row linear combinations by **running the constraint evaluation
program in _reverse_** using reverse-mode
[automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation). This program
(the AD program mentioned [above](#project-purpose)) has the same structure as the original—visiting
the same operations in the same order—but instead of computing witness values it accumulates
derivatives. Each constraint **contributes its coefficient to the gradient of its operands**.

### Lazy Gradient Accumulation

The `PureToWitnessRef` opcode optimizes AD by wrapping field values in heap-allocated boxes that
accumulate their derivatives.

**Problem**: If the expression `(a + b)` appears in multiple constraints with coefficients `c1, c2`,
naive AD propagates separately:

- Add `c1` to `a`'s gradient, add `c1` to `b`'s gradient
- Add `c2` to `a`'s gradient, add `c2` to `b`'s gradient
- = 4 operations

**Solution**: Boxed values accumulate incoming gradients. On destruction (via GC), the combined
gradient propagates once:

- Accumulate `c1 + c2` in the boxed `(a + b)` node
- On destruction: add `(c1 + c2)` to `a`, add `(c1 + c2)` to `b`
- = 2 propagation operations

This is lazy reverse-mode AD, as gradient propagation is deferred until values are no longer needed.

### LogUp Lookups

Spartan as implemented in Mavros is extended with PLONK-style lookup constraints using the
[LogUp argument](https://eprint.iacr.org/2022/1530). We have two additional opcodes to support this:

- `Lookup` is used in witgen
- `DLookup` computes the corresponding derivatives for AD

The VM has special handling for these instructions, reusing yet unused parts of the output vector to
store intermediate results and then compute the challenge-based values in a final pass after
execution is complete.

## Compilation Stages

While implementing processes foreign to normal compilers, Mavros has a fairly standard batch-based
compiler architecture with multiple stages, each of which consists of multiple passes. These stages
are described below.

### Stage 1: Noir Compilation

Mavros uses (a custom [branch](https://github.com/reilabs/noir/tree/spartan-vm) of) the Noir
compiler to parse and type-check Noir source code. This provides familiar diagnostics to the user of
Mavros, while ensuring that the input code is correct Noir.

We use this to convert Noir's monomorphic AST this to our own SSA representation
([`compiler/ssa_gen`](../src/compiler/ssa_gen/)). This SSA relies mostly on immutable data
structures to make transformations easier to write and reason about, at the cost of some data
copying.

### Stage 2: Witness Type Inference and Monomorphization

Noir (at the the point of our integration) does not distinguish between values that are always
constant across all runs and values that depend on witness elements. This distinction is critical
for R1CS compilation, with values split into two categories:

- **Pure values** (`Pure`) are constants that are the same across all executions and are baked into
  the R1CS matrix coefficients.
- **Witness values** (`Witness`) are values that depend on the input, and are placed in the witness
  vector.

**Witness type inference**
([`witness_type_inference.rs`](../src/compiler/witness_type_inference.rs)) propagates this
information through the program. Starting from main function parameters (which are marked as
`Witness`), we compute the taint of every value in the program.

**Untainting control flow** ([`untaint_control_flow.rs`](../src/compiler/untaint_control_flow.rs))
ensures that all branch conditions depend only on `Pure` values. This is required because the R1CS
matrix _shape_ is fixed at compile time - different execution paths would require different numbers
of constraints, which is impossible.

It does this by first specializing generic functions based on the taints of their arguments
(monomorphization). When a branch condition depends on `Witness` values, this pass transforms the
code to evaluate both branches and select the result, **converting control flow to data flow**.

### Stage 3: Optimization Passes

After monomorphization, we run a series of optimization passes on the full SSA. Key passes include
but are not limited to:

| Pass                      | Purpose                                                |
| ------------------------- | ------------------------------------------------------ |
| **Mem2Reg**               | Promote memory operations to SSA registers             |
| **CSE**                   | Eliminate common subexpressions                        |
| **Condition Propagation** | Propagate known conditions into branches               |
| **DCE**                   | Remove dead code                                       |
| **Arithmetic Simplifier** | Algebraic simplifications (e.g., `x * 1 → x`)          |
| **Specializer**           | Inline and specialize functions based on cost analysis |

The **Specializer** pass uses a **symbolic instrumentation engine**
([`instrumenter.rs`](../src/compiler/analysis/instrumenter.rs)) to drive optimization decisions.
This engine symbolically executes functions, tracking which values would become pure if a loop were
unrolled and estimating the constraint cost for a variety of speculative specialization decisions.

### Stage 4: Explicit Witness Lowering

The **Explicit Witness** pass ([`explicit_witness.rs`](../src/compiler/passes/explicit_witness.rs))
lowers operations in a witness-taint-dependent manner. The same high-level operation can be compiled
differently depending on whether its operands are `Pure` or `Witness`. This pass inserts
`WriteWitness` and `Constrain` instructions where needed to bridge the gap between the high-level
Noir operations and R1CS constraints.

> [!NOTE]
>
> As an example, the **multiplication** of two `Pure` values $a$ and $b$ is just direct computation
> that can be baked into the coefficients of the `R1CS` matrix.
>
> Multiplying two `Witness` values, however, requires allocating a fresh witness variable for the
> result, and emitting a constraint $a \times b - \text{result} = 0$ to enforce correctness.

### Stage 5: The Witgen/AD Split

At this point, the pipeline splits into two branches, driven directly by the way that the
`WriteWitness` instruction is interpreted:

- **Witgen** handles the witness generation
- **R1CS/AD** handles the generation of the constraint system and the automatic differentiation
  process.

**Before the split**, `WriteWitness` is **bidirectional**. It takes a 'hint' value as input (how to
compute the witness), and returns a 'witnessified' value as output, now marked as `Witness`.

- For **witgen**, we ignore the returned 'witnessified' value, and replace all uses of the value
  with the hint input.
- For **R1CS/AD**, we drop the hint input entirely and keep _only_ the 'witnessified' value.

This is driven by selectively calling the
[`WitnessWriteToFresh`](../src/compiler/passes/witness_write_to_fresh.rs) and
[`WitnessWriteToVoid`](../src/compiler/passes/witness_write_to_void.rs) passes.

This design allows DCE (dead code elimination) to automatically remove operations that are only
needed on one side of the pipeline. For example, the hint computation logic is removed from the
R1CS/AD path, while witgen acquires additional opportunities for common-subexpression eliminiation
(CSE).

### Stage 6: R1CS Generation and Further Optimization

**R1CS generation** (`r1cs_gen.rs`) executes the SSA symbolically and extracts the constraint
matrices $A$, $B$ and $C$.

### Stage 7: Bytecode Compilation and VM Execution

Finally, we compile the optimized SSA to bytecode ([`codegen/`](../src/compiler/codegen/)) and
execute it in the specialized Mavros Virtual Machine ([`vm/`](../vm/)). (`vm/`).

Executing the witness generator program produces:

- The **witness vector** $w$ and,
- The **constraint evaluations** $Aw$, $Bw$ and $Cw$.

Executing the AD program produces, for $c_r$ some random coefficients:

- $c_r \cdot A$, a random linear combination of $A$'s rows
- $c_r \cdot B$, a random linear combination of $B$'s rows
- $c_r \cdot C$, a random linear combination of $C$'s rows
