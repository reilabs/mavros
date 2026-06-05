# Witness Type Inference

Witness type inference decides which SSA values are pure compile-time data and which values may
depend on private witness input. The pass also monomorphizes functions on this distinction, because
the same original function can need different witness types at different call sites.

The implementation lives under
[`witness_type_inference`](../compiler/src/compiler/analysis/witness_type_inference.rs):

- `signature.rs` defines the boundary representation: specialization keys, boundary ports,
  summaries, variable-shaped witness shapes, and the shape conversion/query helpers.
- `fixpoint.rs` builds dependency graphs from SSA bodies and computes function summaries.
- `solver.rs` solves the closed specialization graph over original functions without mutating SSA.
- `specialization.rs` materializes solved specs as clones, rewires calls, and emits the final
  `FunctionWitnessType` annotations.

## Core Model

The pass represents each witness-relevant position in a value with a variable. Scalars have one
variable. Arrays, slices, refs, and tuples have a variable for the container plus variables for
their nested positions.

A directed edge `a -> b` means:

> if `a` becomes witness, then `b` must also become witness.

There is also an `always` source. An edge from `always` marks a value position as unconditionally
witness, for example the result of `WriteWitness` or a position requested by a specialization key.

The instruction rules are local and monotone: each opcode only adds variables and edges. There are
no witness-driven loops in the body builder. Closure is just graph reachability from `always` or
from selected boundary ports.

## Boundary Summaries

Before specializing, WTI computes a summary for every original function. A summary is a graph over
the function boundary:

- parameter ports,
- return ports,
- one CFG-witness port,
- the synthetic `always` source.

An edge `parameter[0].field -> return[0]` means that any specialization where that parameter
position is witness must also treat the return position as witness.

The body builder constructs a full variable graph for the function, then `summarize_body` asks which
boundary ports are reachable from each boundary source. That reachability relation is the function
summary.

Calls are handled by projecting the callee's current summary through the caller's argument, result,
and CFG variables. This is what makes pointer-return effects visible to callers: if a callee summary
says a returned ref can force an input ref payload to witness, the caller receives exactly that edge
between the corresponding call-site variables.

## Fixed Point and Recursion

Summaries are inferred over the static call graph. The call graph is condensed into SCCs and then
processed callee-before-caller.

For an acyclic caller, all callee summaries are already complete, so the caller body is scanned once
to convergence locally. For a recursive SCC, the functions inside the SCC are repeatedly rescanned
until none of their boundary summaries gains a new edge.

This terminates because each summary contains a finite number of boundary ports, and the loop only
adds edges.

## Closed Specialization Keys

A `SpecKey` contains concrete witness shapes for parameters, returns, and CFG. Incoming keys are not
used directly. They are first closed under the function summary:

1. Convert every witness position in the requested key to an active boundary port.
2. Add the synthetic `always` port.
3. Follow summary edges to fixed point.
4. Convert the active ports back to concrete witness shapes.

The closed key is the canonical specialization identity. If a call asks for an all-pure key but the
callee summary proves that this key necessarily returns a witness ref, the key is closed to the
witness-returning form before lookup or cloning.

## Solving Specs

The solver maintains a queue of closed keys. Solving a key never clones or rewrites SSA. It builds
the specialization graph over the original function, computes reachable witness variables, records
the concrete witness type of every original value and block CFG, and queues closed callee keys for
constrained static call sites.

The collected fixpoint result is a `SolvedProgram`: solved specs, the root spec id, and the resolved
callee spec for every constrained call site in every caller spec.

## Materialization and Rewriting

Once the solve queue is exhausted, materialization begins. Each solved key gets one duplicate of its
original function plus an original-to-clone value map. The engine then rewrites every constrained
static call target in the cloned functions to the specialized callee selected during solving and
sets the entry point to the root specialization.

Unused cloned specs are not removed by WTI itself; later cleanup passes can remove unreachable
functions.
