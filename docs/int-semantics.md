# Integer Semantics in Mavros

Mavros compiles Noir, so the semantics of any given integer operation is up to Noir and not us. This
document explains how we handle this distinction.

We have one centralized [record](../int-semantics/) of the semantics, and every evaluator has to
demonstrate a _relation_ to that model that can be checked in a test. This document is the
human-readable counterpart to [`register.rs`](../int-semantics/src/register.rs), which is a
best-effort check that this holds.

## Noir as the Reference

The exact semantics of these operations are required to match the currently-pinned version of Noir,
which is the one `compiler/Cargo.toml` names: tag `v1.0.0-beta.22`, commit
`c57152f91260ecdb9faad4efc20abb14b6d2ece7`. Every citation below was read in that tree, so the table
is re-checkable rather than merely plausible; a Noir bump is the point at which to read it again.
Paths are relative to `noir/compiler/`.

| Rule                                      | Where                                                                 |
| ----------------------------------------- | --------------------------------------------------------------------- |
| what folds, and what counts as a failure  | `ir/instruction/binary.rs::eval_constant_binary_op`                   |
| shift lowering and the amount bound       | `opt/remove_bit_shifts.rs::enforce_bitshift_rhs_lt_bit_size`          |
| signed `div`/`mod`/`lt`                   | `opt/expand_signed_math.rs`                                           |
| cast semantics                            | `ssa_gen/context.rs::insert_safe_cast`                                |
| what may be dead-code eliminated          | `opt/die.rs` + `ir/instruction.rs::requires_acir_gen_predicate`       |
| a rejection is a runtime failure          | `opt/remove_unreachable_instructions.rs`                              |
| `iN as Field` is a type error, not a cast | `noirc_frontend/src/hir/type_check/errors.rs`                         |
| unconstrained code gets the same checks   | `noirc_evaluator/src/brillig/…/brillig_binary.rs::add_overflow_check` |

The rules themselves:

- **`+`, `-`, `*` reject on overflow**, in both constrained and unconstrained code. Noir's
  `unchecked` flag is an internal SSA optimization that never reaches the monomorphized AST mavros
  builds from, and `wrapping_add` and friends are stdlib functions that route through `Field` and
  cast back.
- **`/` and `%` reject a zero divisor**, and signed `INT_MIN / -1`. Signed division truncates toward
  zero and the remainder takes the dividend's sign.
- **`<<` rejects when the amount is at or past the operand's width**, and otherwise **wraps**:
  `200u8 << 1` is `144`. Losing bits off the top is permissible. `>>` rejects on the same condition,
  and is arithmetic for a signed operand and logical for an unsigned one.
- **A shift's amount is typed as the left operand's own type.** Noir's elaborator unifies the two,
  which is what makes checking the amount against `rhs_type.bit_size()` correct.
- **Casts** truncate the low bits when narrowing and sign-extend when widening a signed source.
  `iN as Field` is not a cast at all: the frontend rejects it with "Only unsigned integer types may
  be casted to Field".
- **A failing operation is not dead.** `die.rs` makes a `Binary` eliminable only when it does not
  require an ACIR-gen predicate, and a checked `Add`/`Sub`/`Mul` — or a shift whose amount is
  non-constant or out of range — does.
- **A rejection is a runtime constrain failure, not a compile error.** This is why the out-of-range
  tests live in `noir_failure_tests/` and not in a compile-failure corpus.
- **Unconstrained code follows the same rules.** Brillig's VM happens to answer `0` for an
  over-shift, but `brillig_gen` emits the check ahead of the operation, so that fallback is
  unreachable in generated code.

## The Model

`mavros-int-semantics` states the semantics as two functions, because "Noir rejects this" and "a
total backend must still produce a bit pattern" are both crucial questions, but ones that have
potentially different answers.

```rust
pub fn eval(op, sign, bits, lhs, rhs_bits, rhs) -> Outcome;      // Value(v) | Rejected(reason)
pub fn residue(op, sign, bits, lhs, rhs_bits, rhs) -> Option<Raw>;
```

`eval` is **the Noir contract**: what an execution of this operation must do. Its four rejection
reasons, `Overflow`, `DivByZero`, `DivOverflow`, `ShiftAmount`, are the list of ways an integer
operation can fail, and each of them is enforced by a specific module of guard IR.

`residue` is **the backend contract**: the deterministic pattern that a _total_ evaluator produces,
including on inputs `eval` rejects. The VM cannot decline and nor can LLVM. What they must not do is
answer _differently_ from each other, because differing can mean disagreeing about a witness, not
only about a value. `residue` returning `None` means _deliberately unspecified_ — it says so for a
zero divisor and for signed `INT_MIN / -1`, because LLVM calls those undefined and hence there is
genuinely no shared answer. At an unspecified point the obligation sits solely on the guard IR, and
an evaluator that reaches such a point is entitled to treat it as a compiler bug.

The two are tied together inside the crate: `eval(p) == Value(v)` implies `residue(p) == Some(v)`.
So conforming to `residue` on accepted inputs is conforming to Noir.

`rhs_bits` is a separate parameter on purpose. A shift's amount can be declared narrower than the
value it shifts, and an amount's own width is the only thing that says what its bit pattern means.

## What We Check

The register in `int-semantics/src/register.rs` names each evaluator, the relation that holds for
it, the files it lives in and the tests that check said relation. It asserts that every registered
file still exists, that every registered relation still has a test of that name in the file that
claims it, and that this document has exactly the registered tags as sections in the register's
order. So a renamed test, a moved evaluator or a section added here without an entry there is a
failing test rather than a document that has quietly stopped being true.

None of that can catch a site nobody thought to register, so it also includes a best-effort sweep:
every crate outside `int-semantics/` is searched for a `.wrapping_*` or `.overflowing_*` call, and
any that is neither inside a registered evaluator nor on a short, reasoned waiver list fails the
test.

## The Register

| Tag           | What                                    | Relation                            |
| ------------- | --------------------------------------- | ----------------------------------- |
| `const-fold`  | the SCCP lattice's constant folder      | refinement                          |
| `specializer` | the specializer's fold at a call site   | delegation to `const-fold`          |
| `r1cs-fold`   | the R1CS symbolic executor              | exact, and may not decline          |
| `cost-model`  | the cost interpreter                    | delegation to `residue`             |
| `vm`          | the bytecode interpreter's opcodes      | total, and equal to `residue`       |
| `llvm`        | the LLVM backend, which reaches WASM    | the VM's                            |
| `value-range` | the interval domain's arithmetic        | soundness                           |
| `totality`    | the PRE totality oracle                 | total exactly where nothing rejects |
| `guard-ir`    | the checks that make a rejection happen | rejects exactly what `eval` rejects |

### `const-fold`: the SCCP Lattice's Constant Folder

`analysis/click_cooper/lattice.rs`. The compiler's speculative folder: it runs long before the guard
IR exists, on whatever constants the analysis happens to have proved.

The relation is **refinement** in both directions. It may answer `None` as declining to fold is
always safe, but it may never answer a value the model does not give, and it may never fold an input
the model **rejects**. That second half is the one that is easy to get wrong: folding an overflowing
`2 + 254` at `u8` to `0` does not merely give a wrong answer, it _deletes a rejection_ the program
was required to have, because the operation disappears along with it.

The width discipline stays here rather than moving into the model. HLSSA types an `IntArith` result
as `int{max(s1, s2)}`, so a fold that changed a constant's width would change the program's types;
that is a rule about this IR, and the model has no opinion on it.

### `specializer`: the Specializer's Fold at a Call Site

`passes/specializer.rs`. It folds a callee body against one call site's arguments, and so mints
constants the lattice never saw; "the lattice would have folded it correctly first" was never a
defence for a second implementation here. The relation is therefore **delegation**: it evaluates no
integer of its own. Its `cmp`, `arith`, `sext`, `bit_range`, `cast` and `not` are all calls into
`const-fold`.

### `r1cs-fold`: the R1CS Symbolic Executor

`codegen/hlssa_to_r1cs.rs`. The **final** evaluator. It runs _after_ the guard IR, so an operation
reaching it has already had its rejection enforced or proved unnecessary, and something has to be
written into the constraint system. It must not decline.

That makes `residue` its obligation rather than `eval`:

- an accepted input produces Noir's value;
- a rejected input the model still specifies produces the pattern every other backend produces, so
  that a program mavros wrongly accepted is at least wrong the same way everywhere;
- an input the model leaves unspecified is an ICE, because reaching one means the guard IR did not
  run first, which is an ordering bug in the compiler rather than an error in the program.

### `cost-model`: the Cost Interpreter

`analysis/instrumenter.rs`. It estimates the circuit a specialization decision would produce, over
dummy signature values. A wrong value here picks a different branch and changes the emitted program,
so it **must be exact** on accepted inputs.

The relation is **delegation to `residue`**, with `unwrap_or(0)` where the model declines.
Everywhere the intended semantics _has_ an opinion, this agrees with it.

### `vm`: the Bytecode Interpreter's Opcodes

`vm/src/bytecode.rs`. This one does **not** delegate because it's a hot dispatch loop over `u64`
cells with a separate `Int128` lane, while the model computes in `u128` throughout. Delegating would
mean computing `eval` and discarding it which is not great for performance.

The relation is thus checked instead of enforced: **total, and equal to `residue` wherever the model
specifies a pattern**. Total means no panic, no undefined behavior and no process abort — a witness
generator that aborts reports nothing, where one that answers reports a failed execution. Every
answer must also sit inside the operand width, which is the masked-cell invariant the rest of the
interpreter relies on.

### `llvm`: the LLVM Backend

`codegen/llssa_to_llvm.rs`. It emits instructions and never computes a value, so a Rust mirror of
its choices would only be checking a copy. Instead the lowering itself is called with two constant
operands and LLVM's own constant folder answers: what comes back is the real lowering composed with
LLVM's definition of the instruction it chose.

### `value-range`: the Interval Domain's Arithmetic

`analysis/value_range_analysis.rs`. It answers a range rather than a value, so equality is not
expressible. What it needs to provide is **soundness**: every value a model-_accepted_ execution can
produce lies inside the range it answers.

A rejected execution produces no value at all, so the analysis owes it nothing. That's what licenses
the transfer to answer the non-wrapping interval for an operation that would overflow, rather than
widening to the whole type.

### `totality`: the PRE Totality Oracle

`passes/partial_redundancy_elimination/totality.rs`. It decides whether an operation may be
speculated to a point where it was not bound to execute. Speculating something that can reject would
turn a program that succeeds into one that fails.

The relation is a table check: a group may be answered an unconditional `true` **iff** the model
rejects none of its inputs. The bitwise operations are the only ones that qualify. Driven off the
model, so the day a rejection reason is added, this fails.

### `guard-ir`: the Checks that Make a Rejection Happen

`passes/shared/overflow_guard.rs`, `divmod_guard.rs` and `shift_guard.rs`, one per rejection reason
plus `DivOverflow` sharing the division one. Every other entry in this register is about what an
_accepted_ execution answers. This one is about which executions are accepted at all, and it is the
only entry whose failure mode is a program Noir rejects producing a proof.

Two of these rejections are stated twice, because an operand that is a witness cannot be checked
with a pure comparison: `witness_bitwise.rs::emit_shift_amount_check` owes the amount bound and
`witness_integer_arith.rs`'s guarded rangechecks owe overflow, both built out of constraints
instead. They are registered alongside the `shared/` modules for that reason and the pairs are held
together by the corpus below, which renders every rejecting program with **witness** operands, and
by the hand-written `pure_shift_amount_oob_fails` / `witness_shift_amount_oob_fails` pair that
exercises one rejection down both routes.

Mavros builds HLSSA straight from the monomorphized AST and never runs Noir's own SSA pipeline, so
nothing upstream has already attached a failure to a failable operation. Every rejection in a
compiled program is one of these modules having planted it. Each has three consumers that must not
drift — the guarded lowering, the unguarded lowering, and DCE, which must keep the check when it
deletes the operation, because Noir keeps a dead overflowing add.

The relation is ensured end to end rather than by a unit test using the **generated rejection
corpus**: `int-semantics/src/corpus.rs` renders one Noir program per `(operation, reading, reason)`
from `eval` itself, each asserting the answer a total backend produces — `residue`, not `eval` — so
that deleting the guard makes the program _pass_, which an expect-failure row reports as a failure.

That corpus renders each cell at the **narrowest** width the model rejects at. On its own that is
coverage of the _reasons_ and not of the arithmetic each check is built from, so an off-by-one in a
width-dependent bound would be invisible to it. It therefore renders a second program at the
**widest** rejecting width for the reasons whose bound the width actually decides — `Overflow`'s
magnitude limits, `ShiftAmount`'s `amount < bits`, and `DivOverflow`'s `INT_MIN`. `DivByZero` gets
no second program: its check is `rhs == 0`, the same test at every width. The accepting corpus is
what checks every width for the operations themselves, and it is `assert_eq` on values rather than
on whether the program ran.

One predicate is outside both corpora by construction. `overflow_provably_impossible` **deletes** a
check where the range domain proves nothing can fail, and a check that was wrongly deleted leaves no
program behind to be rejected.

## The Generated Corpus

Every conformance test above calls an evaluator directly, which only ever reaches the _folding_
path: a constant in, a constant out. The generated corpus checks the other one. Every operand is a
`main` parameter and therefore a witness, which defeats constant folding outright and pushes the
value through guard emission, the witness lowering, R1CS construction, the VM and the WASM backend,
with an `assert_eq` that makes the answer visible to the existing R1CS-satisfaction oracle in every
lane.

It is generated from the model rather than written by hand.
`MAVROS_BLESS=1 cargo test -p mavros-int-semantics --test generated_corpus` rewrites it, so a
semantic change arrives as a reviewable diff of Noir programs.

## Known Divergences

These are conformance gaps, not deferred optimizations. Each is a place where mavros does not
currently implement Noir.

- **Signed integers wider than 64 bits are unsupported.** Noir has `i128`; mavros caps a signed
  reading at 64 bits and rejects the type outright.
- **`u128 <<` is unsupported**, and panics rather than compiling.
- **A shift at a non-power-of-two width would corrupt an in-range amount.** The amount mask is
  `amount & (bits - 1)`, which is a modulo only where the width is a power of two; at `u3` an amount
  of `1` masks to `0` in the VM and in the WASM backend while `hlssa_to_r1cs` shifts by `1`, so the
  two halves of a proof would disagree. This is structurally unreachable rather than fixed — Noir
  has no such width and `BitRange` keeps its source's type — and `shift_guard::shift_operand_bits`,
  the single funnel both pure shift lowerings take their width from, asserts it stays that way.
- **A dead _witness_ `Add`/`Sub`/`Mul` loses its rejection.** DCE keeps the overflow check when it
  deletes a dead operation, but only when both operands are pure. The check a live witness operation
  gets is a range check on a field result rather than a comparison, and DCE cannot build that.
