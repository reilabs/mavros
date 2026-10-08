# Integers in Mavros

While Mavros compiles Noir, the frontend it is integrated with goes beyond the standard Noir
integers to instead support big integers natively. Integer types (and values) can be written for
integers with anywhere from two bits to 16384 bits, and can be used without restriction in Noir
code.

An integer type `intN` is an `N`-bits-wide integer (created in Noir using `iN` or `uN` depending on
the semantics you want), represented as a 2's compliment bit pattern. Every operation is supported
on every integer size, allowing:

- arithmetic: `+`, `-`, `*`, `/`, `%`, under both readings;
- comparisons: `==`, `<` (unsigned and signed), and assertions of each;
- bitwise: `&`, `|`, `^`, `!`;
- shifts: `<<`, logical `>>`, arithmetic `>>`;
- casts: narrowing, widening, sign extension, `Field` to integer and integer to `Field`;
- bit windows, and decompositions into bits or bytes (`to_bits`, `to_radix`);
- spread and unspread, the bit-interleaving primitives behind lookup-based bitwise operations.

There is no distinction in ability or functionality between the integers that Noir supports
upstream, and the expanded integer types allowed by the frontend Mavros is paired with. Arrays,
slices, tuples, references, globals, function arguments, and so on are all supported just as for the
native integers.

The frontend is pinned in `compiler/Cargo.toml` to `worldfnd/noir`'s `provekit-v2` revision
`59467ac7c66cc27dcf70edc22a9d8b655a9fdf76`. The fork extends integer widths and retains `BigInt`
literals through monomorphization; Mavros lowers them directly to `IntBits`, preserving every bit.
The existing SSA oracle tests remain the arithmetic reference for these extended widths, while
`compiler/tests/frontend_bigints.rs` also exercises the source and ABI boundaries.

That said, there are currently a few small limits, most of which we hope to lift in future:

- **Parameters to and Returns from `main`:** The frontend permits only the standard integer types
  (`u8`, `u16`, `u32`, `u64`, `u128`, `i8`, `i16`, `i32`, and `i64`) at this boundary, including
  inside arrays and structs. Wider values can be assembled from arrays of these types inside a
  program.
- **Field Casts:** Casting an integer wider than the field modulus to a `Field` is a compile-time
  error, because this would admit two integer values that produce the same field element (a
  soundness hole).
- **Wide Integers and WASM:** Currently the WASM runner places limits on the number of locals that a
  function can host, as well as the stack size, so functions using extremely wide integers may fail
  to compile at the moment.

## Semantics

The semantics of integers in Mavros match those of upstream Noir, extended to the additional integer
widths. In particular, the following all hold.

- **`+`, `-` and `*` reject on overflow**, under the operation's reading. `wrapping_add` and its
  relatives are stdlib functions that go through `Field` and cast back.
- **`/` and `%` reject a zero divisor**, and signed `INT_MIN / -1` and `INT_MIN % -1`. Signed
  division truncates toward zero, and the remainder takes the dividend's sign.
- **`<<` and `>>` reject an amount at or past the width.** Otherwise `<<` wraps (`200u8 << 1` is
  `144`), and `>>` is arithmetic for a signed reading and logical for an unsigned one. The amount
  has the shifted value's type.
- **Casts** keep the low bits when narrowing and sign-extend a signed source when widening.
  `uN as Field` is the unsigned value; `iN as Field` is a type error in Noir.
- **A witnessed `Field` to integer cast is a relabel.** Noir's `as` truncates with a bit window
  first, so it truncates everywhere. The bare cast, as `mavros_stdlib`'s `unsafe_cast` uses it,
  carries the element through unchanged and leaves the program to bound it with a later range check
  or lookup. That bound is required: the compiler treats the result as within its width and drops
  checks on that basis. On known values the cast reads the low bits, so the two agree whenever the
  value fits.
- **A rejection is a runtime failure**, not a compile error: an unsatisfiable constraint or a
  witness-generation trap. A failing operation is never removed as dead code, and unconstrained code
  follows the same rules.

## For Developers

The following sections are designed to be read by developers interacting with the way Mavros handles
integers, and is not needed to simply use them.

### Representation

An integer known at compile time is stored as a bit pattern at its full width, and is computed by
the VM or by WASM as needed. Witnessed integers are held in the constraint system as either:

- **a single field element** if they are narrow enough to fit below the field modulus, or;
- **a series of limbs** if they exceed the field modulus, where each limb is small enough to _not_
  exceed the field modulus.

The bit pattern representation is identical regardless of whether the source carried a signed or
unsigned integer type. Internally to Mavros, the signedness is a property of the _operation_ rather
than of the type, so the SSA types integers simply as `intN`.

Operations are lowered to constraints depending on their representation, and those constraints
ensure that the semantics match regardless of the underlying representation. The following are worth
understanding:

- **Overflow:** This is handled using a range check. A sum's top limb takes no carry out, so an
  overflowing sum fails the range check on that limb. A subtraction that underflows leaves it
  negative, which fails the same check. A product also holds every partial product past the top limb
  to be zero in addition to the range-check on the top limb.
- **Power of Two:** A shift multiplies or divides a value by a value in a lookup table of
  `(n, 2^n)`. At a power-of-two width the tables keys are exactly the legal amounts, so the lookup
  also rejects an amount past the width.
- **Signed Operations:** These reuse the unsigned ones, and then apply additional checks as needed.
  A signed sum or difference checks the sign bits and its answer against the carry out of the top
  limb. Signed products, quotients and remainders check that the magnitude fits the signed range for
  its sign. Arithmetic shifts are simply the logical shift with any vacated bits filled from the
  sign.
- **Witnessed Branches:** Inside a branch on a witness, an operation's checks are multiplied by the
  branch condition, so the answer is zero where the branch is not taken (and hence a non-taken
  branch cannot fail).
- **Rejections:** These take one of two forms: an operation on a witness rejects through an
  unsatisfiable constraint, and one on known values traps during witness generation.

### Conformance

The `mavros-int-semantics` crate states the expected semantics of Mavros' integers using two
functions:

```rust
pub fn eval(op: IntOp, lhs: &IntBits, rhs: &IntBits) -> Outcome; // Value(v) | Rejected(reason)
pub fn residue(op: IntOp, lhs: &IntBits, rhs: &IntBits) -> Option<IntBits>;
```

- `eval` is **what Noir expects**. Its rejection reasons, `Overflow`, `DivByZero`, `DivOverflow` and
  `ShiftAmount`, are every way an integer operation can fail.
- `residue` is **what the backends' should do**: the pattern a total evaluator produces, including
  on inputs `eval` rejects, so that no two backends disagree about a witness. `None` marks a zero
  divisor and signed `INT_MIN / -1`, which LLVM leaves undefined; there, only the checks stand
  between the program and an answer.

`eval(p) == Value(v)` implies `residue(p) == Some(v)`. An `IntBits` carries its width and an `IntOp`
its reading, so neither function takes either as a parameter, and both operands must have one width.

Locations that implement the evaluation of integer semantics are checked for conformance using the
`int-semantics/src/register.rs` which names each evaluator, the relation it owes the model, its
files, and the tests that check the relation. Its tests fail if a registered file or test
disappears, or if the sections below stop matching the register.

The following are the currently-registered evaluators.

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

`int-semantics/src/corpus.rs` creates a Noir program from the model for every tuple of operation,
reading, and rejection reason. Its operands are `main` parameters, so they are witnesses, which
keeps constant folding out and drives each value through the checks, the witness lowering, R1CS, the
VM and WASM.

Each rejecting program asserts the `residue` answer, so deleting a check makes it pass, which its
expect-failure run reports. Each reason is rendered at the narrowest rejecting width and, where the
width decides the bound (`Overflow`, `ShiftAmount`, `DivOverflow`), at the widest too. The accepting
programs check every width's values with `assert_eq`.

`MAVROS_BLESS=1 cargo test -p mavros-int-semantics --test generated_corpus` regenerates it, so a
change to the rules arrives as a reviewable diff of Noir programs.
