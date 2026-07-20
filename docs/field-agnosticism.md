# Field-Agnostic Mavros

The current state of mavros is tied to BN254, but the project should, in general, be agnostic to the
field it is operating over (see #120). This is a multi-stage process that will occur as follows:

- [x] **Phase 1:** Identify all locations where we make assumptions about fields that are not
      field-agnostic.
- [ ] **Phase 2:** Add a façade over the existing BN264 using a field abstraction. This should
      produce byte-identical output as it is just an interface change.
- [ ] **Phase 3:** Add a `FieldConfig` to the SSA, route the four evaluators, unify `two_pow` and
      kill the modulus/width literals.
- [ ] **Phase 4:** Make it so the VM takes monomorphic field operations, with specializations only
      for BN254 for now. Make it so the bytecode carries its field. Maybe this looks like `VM<F>`
      but that would require users to instantiate the VM with the correct field.
- [ ] **Phase 5:** Add support for goldilocks through the SSA and the VM using the field
      abstraction. This will need to use `crypto-primitives` as the runtime-switchable backing for
      the field elements. This must support true field width, and not rely on stuffing goldilocks
      into larger limb counts.

The current state has each site of a field assumption tagged with `// FIELD-ASSUMPTION: <tag>`
directly in code. The `<tag>` values match the section headings below.

By convention, definitions and structurally distinct algorithm sites get their own marker.
Repetitive clusters of the same operation get one marker per file with a count.

## How to Read the Checklist

- **`[ ]` / `[x]`** — migration status.
- **Phase** — which roadmap phase touches it (P2 = façade/direct-refs, P3 = `FieldConfig` on SSA, P4
  = monomorphize VM/backend, P5 = goldilocks + per-field width).
- **Kind** — `type-parametric` (already abstracts over the field via `Field`/`PrimeField`; follows
  the alias automatically once it's generic) vs `hardcoded` (names bn254 / a literal / a fixed width
  and must be rewritten) vs `representation` (assumes the 4-limb/32-byte physical layout).

Completion gate (Phase 5): `rg 'ark_bn254|into_bigint\(|\.0\.0|FELT_LIMBS|FIELD_SIZE|\b254\b'` over
`compiler/ vm/ mavros-artifacts/ opcode-gen/ wasm-runtime/` returns only façade-internal hits.

---

## Layer 1 — Type Alias & Direct Field References

### `L1-alias` — Single Type Alias (Hardcoded, P2)

- [ ] `mavros-artifacts/src/lib.rs:7` — `pub type Field = ark_bn254::Fr;` Re-exported at
      `compiler/src/compiler/mod.rs:11` (`pub use mavros_artifacts::Field`) and `vm/src/lib.rs:5`.
      **The root.** Becomes `pub type Field = FieldElement`.

### `L1-direct-ref` — Direct `ark_bn254::Fr` Refs Bypassing the Alias (Hardcoded, P2)

107 occurrences across 21 files. Most are constant synthesis (`ark_bn254::Fr::from(..)`) or intern
keys; all funnel through the façade (`FieldElement::from_u128` / `field_const`). By file (count):

- [ ] `compiler/src/compiler/codegen/hlssa_to_r1cs.rs` (57) — `Value::Const(ark_bn254::Fr)`,
      coefficient arithmetic, `two_pow`, seal/lookup coefficients. **Largest cluster** (also see
      L4).
- [ ] `compiler/src/compiler/passes/instruction_lowering/pure_guards.rs` (7) — `two_pow`,
      `MODULUS_BIT_SIZE` (see L4).
- [ ] `compiler/src/compiler/passes/prepare_entry_point.rs` (6) —
      `field_const(ark_bn254::Fr::from(..))`.
- [ ] `compiler/src/compiler/passes/witness_lowering.rs` (3) — constant synthesis.
- [ ] `compiler/src/bin/test_runner.rs` (3) — `Fr::rand`, `Fr::new_unchecked(BigInt::new([..]))`
      (see L3-frame).
- [ ] `compiler/src/compiler/util.rs` (2) — test `fr()` helper.
- [ ] `compiler/src/compiler/ssa/hlssa/builder.rs` (2) — `field_const` signature (see L2-builder).
- [ ] `compiler/src/compiler/passes/simplifier.rs` (2) — `Fr::zero()/one()`, `.inverse()` (see L4).
- [ ] `compiler/src/compiler/passes/merge_identical_functions.rs` (2) — constant synthesis.
- [ ] `compiler/src/compiler/passes/common_subexpression_elimination.rs` (2) —
      `FConst(ark_bn254::Fr)` intern key (`:696/753`).
- [ ] `compiler/src/compiler/passes/array_sroa.rs` (2) — test `fr()` helper.
- [ ] `compiler/src/compiler/passes/array_boundary_expansion.rs` (2) — test `fr()` helper.
- [ ] `compiler/src/compiler/analysis/witness_taint_inference/mod.rs` (2) — test `fr()` helper.
- [ ] `wasm-runtime/src/lib.rs` (1) — `use ark_bn254::Fr` (see L3/L4).
- [ ] `compiler/src/compiler/ssa/hlssa_to_llssa.rs` (1) — field-constant limb lowering (see
      L2-lower).
- [ ] `compiler/src/compiler/ssa/hlssa/mod.rs` (1) — `Constant::Field(ark_bn254::Fr)` (see
      L2-ir-const).
- [ ] `compiler/src/compiler/passes/partial_redundancy_elimination/totality.rs` (1) —
      `Constant::Field(Fr::zero())` + field-totality reasoning.
- [ ] `compiler/src/compiler/lowering/expression_converter.rs` (1) — Noir literal →
      `Constant::Field` (also L3-width254).
- [ ] `compiler/src/api.rs` (1) — `Fr::rand` input setup.
- [ ] `compiler/src/abi_helpers.rs` (1) — `Fr::from(byte)` ABI decode.

### `L1-serde` — artifact/ABI serialization pinned to `[u64;4]` (representation, P5)

- [ ] `mavros-artifacts/src/lib.rs:18-43` — `lc_serde` serializes each coeff as `[u64; 4]` via
      `into_bigint().0`. Generalize to a tagged / variable-length limb encoding.
- [ ] `mavros-artifacts/src/lib.rs:401` — `InputValueOrdered::Field => vec![4]` (field size in
      words).

---

## Layer 2 — IR Constants & Builder

### `L2-ir-const` — HLSSA Field Constant (Hardcoded, P2)

- [ ] `compiler/src/compiler/ssa/hlssa/mod.rs:1994` — `Constant::Field(ark_bn254::Fr)`. Becomes
      `Constant::Field(FieldElement)`. Must stay `Copy + Eq + Hash` for interning/CSE
      (`SSAConstants`, `ssa/mod.rs:97`, interns by value).

### `L2-builder` — Field-Constant Builder (Hardcoded, P2)

- [ ] `compiler/src/compiler/ssa/hlssa/builder.rs:240` —
      `fn field_const(&mut self, value: ark_bn254::Fr)`. ~30 call-sites; signature becomes
      `FieldElement`. Zero/one init at `builder.rs:679`.

### `L2-lower` — HLSSA→LLSSA Erases Field-ness to Raw Limbs (Representation, P4/P5)

- [ ] `compiler/src/compiler/ssa/hlssa_to_llssa.rs:669-682` — `Constant::Field(fr)` → 4×i64
      `LLStruct` from `fr.0.0` (raw Montgomery limbs). Becomes `field.montgomery_backend_limbs()`
      (length = `F::STORAGE_CELLS`). These are the Montgomery **storage** limbs; the canonical-limb
      round-trip op is `L3-limb-op`.

### `L2-seam` — Symbolic-Executor `Value`/`Context` (Type-Parametric, P3)

- [ ] `compiler/src/compiler/analysis/symbolic_executor.rs:100` —
      `fn of_field(f: Field, ctx: &mut Context)`. Already threads a `ctx`; the `FieldConfig` rides
      in the `ctx`. `materialize_constants` (`:203`) builds constant `V`s once. This is the plumbing
      seam for the four evaluators below.

---

## Layer 3 — Fixed Physical Shape (4 limb / 32 byte / 254 bit)

### `L3-felt-limbs` — The Limb-Count Constant (Representation, P4→P5)

- [ ] `vm/src/bytecode.rs:16` — `pub const FELT_LIMBS: usize = 4;`. Becomes `F::STORAGE_CELLS`.
      Consumers: `codegen/bytecode/mod.rs:1737`, `codegen/bytecode/layout.rs:58,71,85,228,271`.

### `L3-field-size` — 32-byte Field Size in the Test Harness (Representation, P5)

- [ ] `compiler/src/bin/test_runner.rs:469` — `const FIELD_SIZE: usize = 32; // 4 x i64 = 32 bytes`.
      Used at ~20 pointer-arithmetic sites
      (`:508,573-575,637-643,686,719,726-736,889,892,949,1018-1028`). Becomes per-field
      (`F::STORAGE_CELLS * 8`).

### `L3-opcode-width` — The Opcode Macro's Hardcoded 4-limb Field Operand (Representation, P4)

- [ ] `opcode-gen/src/lib.rs:152` — `HostType::Field => 4, // TODO parameterize` (operand stride).
- [ ] `opcode-gen/src/lib.rs:126-137` — getter decodes `ark_ff::Fp(BigInt([r0,r1,r2,r3]))`.
- [ ] `opcode-gen/src/lib.rs:208-215` — serializer writes `#i.0.0[0..3]`. Stride becomes
      `F::STORAGE_CELLS` (const per `VM<F>` monomorphization); encode/decode via
      `F::from_limbs_le`/`F::to_limbs_le`.

### `L3-llstruct` — LLVM/WASM Field Struct (Representation, P5)

- [ ] `compiler/src/compiler/ssa/llssa/mod.rs:788-806` — `LLStruct::field_elem()` / `limbs()` =
      4×`Int(64)` ("BN254 field element in Montgomery form"). Becomes per-field width.
- [ ] `compiler/src/compiler/codegen/llssa_to_llvm.rs:585-592` —
      `field_llvm_type()`/`limbs_llvm_type()`.

### `L3-limb-op` — The `field.to_limbs`/`from_limbs` LLSSA Op (Representation, P4/P5)

A first-class LLSSA instruction that round-trips a field value through its **canonical
(non-Montgomery) raw limbs** — the substrate the LLVM backend uses to implement
`to_bits`/`to_bytes`/bitwise ops. Distinct from the Montgomery **storage** limbs in
`L2-lower`/`L3-frame`. Fixed at 4×i64 today; becomes per-field width.

- [ ] `compiler/src/compiler/ssa/llssa/mod.rs:624-627` — the `field.to_limbs`/`field.from_limbs` op
      (dump); builder emitters at `llssa/builder.rs:178-184`.
- [ ] `compiler/src/compiler/codegen/llssa_to_llvm.rs:120-121,652-664,1236-1263` — declares and
      calls the `__field_to_limbs`/`__field_from_limbs` externs (`[4 x i64]` ↔ Montgomery). Runtime
      impls at `wasm-runtime/src/lib.rs:123-152` (see `L4-inverse`/`L3-llstruct`).
- [ ] `compiler/src/compiler/ssa/hlssa_to_llssa.rs:1217,1230,1871,2324,3071,3096` (6 sites) —
      emitters that round-trip through the op to lower bit/byte/bitwise ops.

### `L3-width254` — The 254-bit Field-Width Literal (Hardcoded, P3)

- [ ] `compiler/src/compiler/ssa/hlssa/type_system.rs:276` —
      `TypeExpr::Field => 254, // TODO: parametrize`.
- [ ] `compiler/src/compiler/lowering/expression_converter.rs:1091` —
      `Some(AstType::Field) => (254, false)`.
- [ ] `compiler/src/compiler/lowering/expression_converter.rs:1101` —
      `AstType::Field => (CastTarget::Field, 254)`.
- [ ] `compiler/src/compiler/passes/instruction_lowering/bit_range.rs:292,392` —
      `assert!(bits <= 254, ...)`. All become `FieldConfig::field_bit_size()`.

### `L3-frame` — The VM Frame's 4-limb Field Read/Write (Representation, P4)

- [ ] `vm/src/interpreter.rs:123-129` — `read_field` builds `Fp(BigInt([a0,a1,a2,a3]))`.
- [ ] `vm/src/interpreter.rs:185-194` — `write_field` stores `field.0.0[0..3]`.
- [ ] `compiler/src/compiler/codegen/bytecode/layout.rs:228` — `for_each_constant_word` emits
      `val.0.0[i]` for `i in 0..FELT_LIMBS`.
- [ ] `compiler/src/bin/test_runner.rs:530-537` — `Fr::new_unchecked(BigInt::new([l0..l3]))`
      reconstruction. All route through `F::from_frame_cells`/`F::to_frame_cells` (bn254 identical).

---

## Layer 4 — Static Field Computation

### `L4-eval` — Compile-Time Field Arithmetic (Type-Parametric, P3)

Each does add/sub/mul/div/inverse/eq/lt/cast/to_bits over the concrete field; each already has a
context/operand from which to read the `FieldConfig`.

- [ ] `compiler/src/compiler/analysis/click_cooper/lattice.rs:177-297` — SCCP
      `eval_binary`/`eval_cmp`/`eval_cast` (shared by click_cooper/SCS/PRE). Reads field from the
      self-describing `Constant::Field` operands; needs `&FieldConfig` only for `eval_cast`
      int→Field (`:251`).
- [ ] `compiler/src/compiler/codegen/hlssa_to_r1cs.rs` — `Value` impl (`:41-48` enum, `:437` impl);
      LC coefficient arithmetic `add/neg/mul/div` (`:89-148,214-229`).
- [ ] `compiler/src/compiler/passes/specializer.rs` — `Val` + `ConstVal::Field` side table
      (`:49-56,123`), residual field arith (`:205-241`).
- [ ] `compiler/src/compiler/analysis/instrumenter.rs` — `Value::Field` impl (`:117-137`), field
      arith (`:247-331`).

### `L4-inverse` — Modular Inverses (Hardcoded-by-Modulus, P3/P4)

- [ ] `compiler/src/compiler/passes/simplifier.rs:243` — `(*denom).inverse().unwrap()`
      (Div→Mul-by-inverse).
- [ ] `compiler/src/compiler/analysis/click_cooper/lattice.rs:185` — SCCP field `Div`.
- [ ] `compiler/src/compiler/codegen/hlssa_to_r1cs.rs:141-143` — `Value::div`
      (`ark_bn254::Fr::ONE / rhs`).
- [ ] `compiler/src/compiler/analysis/instrumenter.rs:327-331` — field `Div`.
- [ ] `vm/src/interpreter.rs:477` — lookup batch-inversion `running_prod.inverse().unwrap()` (P4,
      `F::inverse`).
- [ ] `wasm-runtime/src/lib.rs:200-217` — `__field_div` (P4/P5).
- [ ] `vm/src/bytecode.rs:1270` — `div_field` opcode body (P4).

### `L4-decompose` — Bit/Byte Decomposition & Range Reasoning (Representation + Modulus, P3/P5)

- [ ] `compiler/src/compiler/passes/specializer.rs:392-408,565-576` — `to_bits` const-fold
      (`into_bigint().to_bits_le()`).
- [ ] `compiler/src/compiler/passes/specializer.rs:477-479,518-524` — field `bit_range`/cast reads
      low limb `f.into_bigint().as_ref()[0]`.
- [ ] `compiler/src/compiler/analysis/instrumenter.rs:604-630,830-917` —
      `bit_range_op`/`to_bits`/`to_radix`.
- [ ] `compiler/src/compiler/codegen/hlssa_to_r1cs.rs:637-736,802-863` —
      `bit_range`/`sext`/`to_bits`/`not`/`rangecheck`/`to_radix`/spread.
- [ ] `vm/src/bytecode.rs:1767-1815,1242-1252` — `rngchk`/`to_bytes_be|le`/`truncate_f_to_u` (slice
      `into_bigint().0` limbs).
- [ ] `compiler/src/compiler/passes/instruction_lowering/witness_bitwise.rs:230,260,340-341,465` —
      bitmask/sign-extend via `two_pow`.

### `L4-low-limb` — Low-Limb Integer Extraction in Witgen (Representation, P4)

Sites that read **raw limb 0** (`.0.0[0]`) of a witness/output field to recover a small integer — a
lookup-table index, a flag, or a multiplicity count — assuming the value fits in the low limb of the
4-limb layout. Becomes an `F::to_limbs_le()[0]` / low-word accessor.

- [ ] `vm/src/interpreter.rs:300,520,527,537,563,584` (6 sites) — multiplicity count and lookup
      table/flag/index reads (`wit[i].0.0[0]`, `phase1.out_{a,b,c}[..].0.0[0]`).
- [ ] `compiler/src/bin/test_runner.rs:826,875` — `Field::from(out_wit_pre_comm[i].0.0[0])` and
      `let limbs = field.0.0` in output decoding.

### `L4-modulus-literal` — Literal Prime Digits (Hardcoded, P5)

- [ ] `compiler/src/compiler/passes/instruction_lowering/bit_range.rs:196-269` —
      `decompose_canonical_field_bytes`. Structurally bn254-specific: assumes **32 bytes, four
      64-bit limbs, a 2×128-bit modulus split**, with literal constants
      `modulus_hi = 0x30644e72e131a029b85045b68181585d` (`:201`),
      `modulus_lo_m1 = 0x2833e84879b9709143e1f593f0000000` (`:202`),
      `mod_limb2 = 0x2833e84879b97091` (`:247`), `mod_limb3 = 0x43e1f593f0000000` (`:248`). Must be
      generalized to a per-field-width canonicalization deriving these from `FieldConfig`
      (`modulus_bytes_be`, `field_bit_size`) while preserving bn254's exact `field_const` emission
      order.

### `L4-modulus-query` — Modulus/Width derived from the type (Type-Parametric, P3)

Already abstract via `<Field as PrimeField>::MODULUS` — follows the alias, but should read
`FieldConfig` so the `[u64;4]` limb assumption in `.0` is also erased.

- [ ] `compiler/src/compiler/analysis/value_range_analysis.rs:358-364` — `bn254_modulus()`
      (`MODULUS.0`), `field_top` (`:113`).
- [ ] `compiler/src/compiler/passes/instruction_lowering/witness_integer_arith.rs:653-665` —
      `range_fits_field_injectively` (`MODULUS.0`, `hi-lo < p`).
- [ ] `compiler/src/compiler/passes/instruction_lowering/pure_guards.rs:755` — `MODULUS_BIT_SIZE`
      (names `ark_bn254::Fr` directly).

### `L4-sign` — Sign Canonicalization (`value > p/2 ⇒ negative`) (Type-Parametric, P3)

- [ ] `mavros-artifacts/src/lib.rs:60-66` — `field_to_string` (`MODULUS_MINUS_ONE_DIV_TWO`).
- [ ] `compiler/src/compiler/codegen/hlssa_to_r1cs.rs:577-606` — signed `Lt` decode via
      two's-complement threshold `half`.

### `L4-two-pow` — Six Duplicated `two_pow = Field::from(2).pow([e])` (Type-Parametric, P3 — unify)

Collapse onto one `FieldConfig::two_pow(exp)`:

- [ ] `compiler/src/compiler/codegen/hlssa_to_r1cs.rs:22` (returns `ark_bn254::Fr`).
- [ ] `compiler/src/compiler/passes/instruction_lowering/pure_guards.rs:857` (returns
      `ark_bn254::Fr`).
- [ ] `compiler/src/compiler/passes/instruction_lowering/bit_range.rs:429` (returns `Field`).
- [ ] `compiler/src/compiler/passes/instruction_lowering/witness_bitwise.rs:340` (returns `Field`).
- [ ] `compiler/src/compiler/passes/instruction_lowering/witness_integer_arith.rs:585` (returns
      `Field`).
- [ ] `compiler/src/compiler/passes/lookup_spilling.rs:690` (returns `Field`; also `two_pow_u128` at
      `:694`).

---

## Layer 5 — Dependency, Header & Field Selection

- [ ] **Cargo deps** — root `Cargo.toml:33-34` (`ark-bn254`, `ark-ff` from crates.io; consumed by
      `compiler`, `vm`, `mavros-artifacts`, `wasm-runtime`). Add `crypto-primitives` (git, PR #38)
      in Phase 5, behind the façade impls only.
- [ ] **Program header** — `vm/src/bytecode.rs:~2475` (`to_binary_with_debug_info`) / `~2614`
      (`parse_program_header`) have **no magic/version word** today. Phase 4 prepends
      `[MAGIC, VERSION, field_id]`; `run_witgen`/`run_ad` (`vm/src/interpreter.rs:355,713`) select
      the `VM<F>` monomorphization from `field_id`. **The one deliberate corpus bump.**
- [ ] **Field selection** — `compiler/src/api.rs` / `driver.rs` gain a `--field` selector seeding
      `SSA.field` and the header (Phase 5).
- [ ] **Per-field WASM runtime** — `compiler/src/wasm_runtime.rs:22` (`locate_or_build`) +
      `WasmCompileOpts.runtime_lib` (`llssa_to_llvm.rs:50`) select a per-field runtime lib (same
      `__field_*` symbol names, linker-bound). `wasm-runtime/src/lib.rs` gains a second field impl.
- [ ] **JS host** — `wasm-runner/src/field.ts` hardcodes bn254 modulus/R/R_INV; needs a goldilocks
      path (Phase 5; out of scope for Rust corpus validation).

---

## Non-Issues

These are already deliberately field agnostic and should not be touched.

- `spread_bits` / `unspread_bits` (VM) — pure u64 bit-tricks; only their `Field::from(..)` wrappers
  change.
- The generic `SSA<Op, Ty, C>` layer — agnostic; bn254 enters only via the concrete
  `Constant::Field` variant.
- `mavros-opcode-gen` dispatch/`DISPATCH`/`OPCODE_NAMES` generation — opcode _identities_ are
  unchanged (single op-set, field chosen at the module level), only the `HostType::Field`
  width/encode path (L3-opcode-width).
- Hash functions (poseidon/blake3/sha256) — these are compiled **Noir** replacements
  (`mavros_stdlib/replacements/*.nr`), not Rust; they arithmetize through the ordinary field ops
  above, so there is no Rust hash library to swap.
