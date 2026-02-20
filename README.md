# Mavros

Noir to R1CS compiler with witness generation and automatic differentiation binaries, built for use with Spartan.

## Workspace Crates

| Crate | Description |
|---|---|
| `mavros` (root) | Main compiler crate. Contains the CLI binary, compiler pipeline (SSA → R1CS), driver, and public API. |
| `mavros-artifacts` | Shared data types: `R1CS`, `R1C`, `WitnessLayout`, `ConstraintsLayout`, `Field`. Used by both the compiler and VM. |
| `mavros-vm` | Bytecode interpreter that runs witness generation and automatic differentiation programs. |
| `opcode-gen` | Proc-macro crate that generates VM opcode definitions. |
| `ssa-builder` | Proc-macro crate with helpers for constructing SSA IR. |
| `mavros-wasm-runtime` | Minimal runtime compiled to WebAssembly, used when emitting `.wasm` witness generation targets. |

## Prerequisites

- **Rust** (latest stable, via [rustup](https://rustup.rs))
- **LLVM 18** — required by the `inkwell` LLVM bindings:
  ```bash
  # macOS
  brew install llvm@18
  ```
- **Graphviz** *(optional)* — only needed for `--draw-graphs` debug output:
  ```bash
  brew install graphviz
  ```

## Building the Binary

```bash
LLVM_SYS_180_PREFIX=/opt/homebrew/opt/llvm@18 cargo build --bin mavros --release
```

The compiled binary will be at `target/release/mavros`.

## Usage

Run `mavros compile` from the **root of a Noir project** (the directory containing `Nargo.toml`):

```bash
mavros compile
```

This produces two files inside the project's `target/` directory:

| File | Contents |
|---|---|
| `target/basic.json` | ABI, witness-generation binary (`witgen_binary`), and AD binary (`ad_binary`) |
| `target/r1cs.bin` | Serialised R1CS constraint system (bincode) |

### Options

```
mavros compile [PATH] [OPTIONS]

Arguments:
  [PATH]  Path to the Noir project root [default: current directory]

Options:
  --r1cs-output <PATH>    Output path for R1CS constraints  [default: target/r1cs.bin]
  --binary-output <PATH>  Output path for binaries and ABI  [default: target/basic.json]
  --draw-graphs           Generate SSA/CFG debug graphs into mavros_debug/
```

### Example

```bash
cd my-noir-project
mavros compile
# → target/basic.json
# → target/r1cs.bin
```

## WebAssembly Output

Mavros can emit the witness generation binary as a `.wasm` file in addition to (or instead of) running it natively. This is useful when the host environment — e.g. a browser or a WASM-capable verifier — prefers WebAssembly.

### How it works

1. The compiler lowers the witgen SSA to LLVM IR (via `inkwell`/LLVM 18), then compiles it to a `wasm32` target.
2. The generated WASM calls a small set of external functions (`__field_mul`, `__write_witness`, `__write_a/b/c`) that are implemented in the `mavros-wasm-runtime` crate. This runtime handles BN254 field arithmetic and writes outputs into memory.
3. A `.wasm.meta.json` sidecar file is written alongside the `.wasm` with ABI and R1CS layout info, so the host knows how to set up inputs and interpret outputs.

### Triggering WASM output

```bash
mavros --emit-wasm
```

This runs the full pipeline (compile + VM execution) and additionally emits `mavros_debug/witgen.wasm` and `mavros_debug/witgen.wasm.meta.json`.

`--emit-llvm` can be used alongside it to also save the intermediate LLVM IR (`.ll`) file.

> **Note:** WASM output is only available via the default command (no subcommand), not via `mavros compile`.

## Debugging

Passing `--draw-graphs` generates a `mavros_debug/` folder in the project root with SSA diagrams and CFG graphs at each compiler pass. Requires Graphviz.
