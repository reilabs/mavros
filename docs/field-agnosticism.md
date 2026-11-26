# Fields in Mavros

Mavros compiles a Noir program over a prime field chosen at compile time from the ones it supports.
Every portion of the compiler and runtime operates over that field, including the constraint system,
witness generation, and automatic differentiation, as well as the verifier.

Currently, Mavros supports the following fields.

| Field        | Modulus                                               | Bits |
| ------------ | ----------------------------------------------------- | ---- |
| `bn254`      | the BN254 scalar field, `0x30644e72e131a029…f0000001` | 254  |
| `goldilocks` | `2^64 − 2^32 + 1`                                     | 64   |

Key to this is that a program's integers have the same semantics on every field, regardless of their
size, matching the [described](./int-semantics.md) semantics. Do note that performance
characteristics of integers may vary with the field due to their encoding. The program's `Field`
type, on the other hand, is always a value of the selected field.

## Fields in Noir

Mavros' field agnosticism is supported by a customized version of
[noir](https://github.com/worldfnd/noir), the same that provides support for our arbitrary-sized
integers. This fork can operate over the same fields as Mavros, and has a tailored standard library
designed to be field agnostic wherever possible.

When using Mavros to compile your Noir programs you don't need to worry about this directly. Simply
[selecting](#selecting-a-field) a field will be sufficient to configure our Noir fork correctly as
well.

## Selecting a Field

Choosing the field that Mavros uses is simple, but depends on your interaction modality:

- **CLI:** On the command line, pass the `--field` argument, with either `bn254` or `goldilocks` as
  the value. This can be passed before or after any subcommand, and if not passed defaults to
  `bn254` for backwards compatibility.
- **API:** If calling Mavros through its API, the field is selected by the type parameter `F` on the
  functions in `mavros_compiler::api`. The available options are currently `mavros_artifacts::Bn254`
  and `mavros_artifacts::Goldilocks`.

The selected field is carried by every compiled program in its header. This ensures that trying to
load an artifact compiled for a field other than the configured one will result in an error, instead
of silently generating wrong results.

## Field Semantics

The `Field` type in Mavros has the following semantics regardless of which field is configured:

- **Arithmetic:** All arithmetic operations are computed modulo the field's prime, for both
  compile-time and runtime evaluation. The standard library's `std::field::modulus_*` functions
  describe the configured prime.
- **Literals:** All `Field` literals must be below the modulus. A literal at or above the modulus
  results in a compile-time error. A negative literal `-n` is `p - n`, so has a value that changes
  based on the configured field.
- **Casts _to_ `Field`:** Casts from `uN` into `Field` must be injective, and ones that are not are
  refused by the frontend. A cast is injective only when the largest `uN` value, $2^N - 1$, is below
  the modulus, so `N` may be at most 253 on `bn254` and at most 63 on `goldilocks`. This means that
  field casts may be denied on some fields but not others. Signed integers `iN` cannot be cast to
  `Field`.
- **Casts _from_ `Field`:** Casts out of the configured `Field` keep the low bits of the element's
  canonical value. The answer may thus depend on the field.
- **Bit and Byte Decompositions:** The decompositions of a `Field` are _at most_ as wide as the
  field. A wider decomposition of a witness is a compile-time error for constrained code, and in
  unconstrained code a wider byte decomposition fails at witness generation time while a wider bit
  decomposition is zero-padded.

## Changes from Upstream Noir

In order to make it easier to work in a field-agnostic manner, Mavros has some changes from upstream
Noir that are important to understand:

- **Standard Library:** Certain standard library operations have been removed or reworked to better
  accommodate field agnostic operation. Where possible, the compiler will direct you to the
  alternative. In addition, `std::field::*` provides operations and information only about the
  currently-configured field.
- **Parameters to and Returns from `main`:** The program's `main` function may take and return the
  following types: `iN`, `uN`, `bool`, `str`, and arrays, tuples and structs of these types. Most
  importantly it can _never_ take or return a `Field`. This ensures that parameters and returns
  cannot change their semantics based on the configured field; to work with field elements one must
  convert to and from them inside `main`. We do not support data bus parameters, and zero-length
  array or string parameters are also refused.
- **Lookup Soundness:** The soundness of the LogUp lookup argument is based on the size of the field
  it runs over. To ensure that changing the field does not silently degrade the argument's
  soundness, Mavros repeats the argument using independent challenges until it reaches a sufficient
  number of soundness bits. This soundness can be specified by passing `--logup-soundness` to the
  CLI, or via the API, and defaults to 128 bits.

## For Developers

The following sections are intended to be read by developers working on how Mavros handles fields,
and are not needed to make use of field agnosticism.

### Abstracting the Field

The field over which the compiler is operating is described by the following four types located in
`mavros-artifacts/src/field`:

- **`FieldId`:** Defines a field and its wire code, used for tagging artifacts.
- **`FieldConfig`:** Provides a compile-time description of the configured field, carried on every
  SSA. Every accessor that depends on the field is an exhaustive match on `FieldId`, and it has no
  `Default` instance.
- **`FieldElement`:** A field value known at compile time, simply named `Field` inside the
  compiler's code. Arithmetic or comparison between elements of different fields are panics that
  should be unreachable except via API misuse. This is the type that all compile time evaluators
  perform computations using.
- **`MavrosField`:** The trait implemented by raw field element representation types that allows the
  VM to operate using a uniform interface regardless of the selected field element.

No compiler pass should statically name a field, instead **always relying on the one passed in or
configured**.

### Integers on Narrow Fields

When a witnessed integer cannot fit within a single field element it is lowered using a limb-wise
encoding specialized to the field in question. This lowering occurs in `WideWitnessInts`; before
this pass in the pipeline witnessed integers are represented as native big ints, and afterward as
limb-wise encodings. A pure integer stays whole, since the VM and WASM compute it at any width,
except as an element of a sequence, which is split into limbs so that a lookup can read each entry
as one field element. For more information, see the developer section of the
[integer semantics](./int-semantics.md#for-developers) documentation.

### Lookup Soundness

`logup_soundness_report` in `compiler/src/compiler/codegen/hlssa_to_r1cs.rs` sizes the lookup
argument. One repetition at a random challenge fails with probability at most `D/|F|` for
`D = table_entries + num_lookups`, so `K` independent repetitions give `K · (⌊log₂ p⌋ − ⌈log₂ D⌉)`
bits. `K` is the smallest count that meets `--logup-soundness`.

A challenge that zeroes the denominator of a row whose numerator is zero (a table entry of
multiplicity zero, or a lookup with a zero flag) leaves that row's inverse free, which adds at most
another `D/|F|`. Flooring `log₂ p` absorbs that factor of two on `goldilocks`, while on `bn254` the
count can be up to 0.41 bits per repetition high, far inside its margin. `compute_logup_soundness`
states the model in full.

The driver refuses with `Error::LogupSoundnessUnsupported` on any target that needs more than
`MAX_LOGUP_REPETITIONS` (16), or with a field too small for one repetition to give any bits
(`⌊log₂ p⌋ ≤ ⌈log₂ D⌉`).

The repetitions share the multiplicities, so the pre-commitment witness does not change.
`WitnessLayout::logup_repetitions` records `K`, and the challenges are laid out as
`[α₀, β₀, α₁, β₁, …]`, with `β` present _only_ when a table has a value column. A `β` shared between
repetitions would cap the soundness, and the prover must **draw every challenge independently**.

### Testing

The functional test runner compiles and runs every corpus program on both fields, on the VM and on
WASM, and reports a separate success rate for each field. Goldilocks is also tested through
hand-built SSA in `compiler/tests/bigints/goldilocks.rs`, which covers field arithmetic and constant
folding at the field's boundary values, the integer operations either side of each lowering switch,
and the casts across the field boundary.

### Adding a Field

1. Add the field to the pinned Noir fork (its `acvm::FieldId`, field configuration and the
   `#[field(..)]` predicates its standard library gates on), bump the pin in `compiler/Cargo.toml`,
   and map it in `noir_field_id` (`compiler/src/driver.rs`).
2. Add a `FieldId` variant in Mavros, then follow the compile errors through `FieldConfig`'s
   exhaustive matches and `FieldElement`. However, `FieldId::ALL` and `FieldId::from_code` are not
   checked this way and must be extended by hand.
3. Implement `MavrosField` for the field's raw type.
4. Add it to the `#[interpreter(fields(..))]` list in `vm/src/bytecode.rs`, and pin its opcode
   numbers in the `opcode_numbers_are_pinned` test.
5. Add its WASM runtime entry points, in the ABI shape, and in `FieldRuntimeSymbols::of`. Add it to
   the `the_runtime_archive_defines_the_field_entry_points` test which checks them through a linked
   module, and to the `every_field_has_a_wasm_lane` test.
6. Add it to the compiler's test `FIELDS` (`compiler/src/compiler/util.rs`), which the conformance
   sweeps run over, and to CI in `.github/workflows/test-runner.yml`: the corpus job's `field`
   matrix, a download step for its report, and the report list of the "Merge Field Reports" step.
7. Gate any standard library builtin whose constants or algorithm do not hold on the new field. The
   builtins that assume BN254 are marked `FIELD-ASSUMPTION` in `mavros_stdlib`, each with what it
   would need.
8. Run the corpus on the new field, and record each fixture whose outcome depends on the field in
   `FIELD_EXPECTATION_OVERRIDES` (`compiler/src/bin/test_runner/expectations.rs`).
