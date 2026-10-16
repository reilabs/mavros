# Mavros

Mavros is a toolchain for compiling [Noir](https://noir-lang.org) to R1CS with additional
functionality for use with [Spartan](https://github.com/microsoft/Spartan). It provides a complete
pipeline from Noir source to the output of the eventual circuit, and is designed with mobile proving
workflows in mind.

## Building

This project uses [Nix](https://lix.systems) [flakes](https://nixos.wiki/wiki/Flakes) to provide a
pinned build environment with all the tools you need to build the project. Assuming you have nix
installed, simply `make shell` to drop into your user shell inside the build environment, or run
`nix develop --command <your-command-here>` to run a specific command in the build environment.

You can use one of the two following methods to experiment with Mavros.

- Running `make install` will build and install Mavros to your cargo binary directory.
- Running `make release` will build Mavros to `./target/release/mavros`.

Run `make` to get an overview of the available utilities and targets. For complete guidance please
see our [contributing guidelines](./docs/CONTRIBUTING.md). See [usage](#usage) below for more
information on how to use Mavros.

## Usage

To use Mavros, navigate to the **root of a Noir project** (the directory containing `Nargo.toml`) in
your shell. Then run `mavros compile`.

This will produce the following two files inside the project's `target/` directory:

| File                | Contents                                                                        |
| ------------------- | ------------------------------------------------------------------------------- |
| `target/basic.json` | ABI and the program binary (`binary`) containing the witgen and AD entry points |
| `target/r1cs.bin`   | Serialised R1CS constraint system (bincode)                                     |

Pass `--include-debug-info` to additionally write `target/basic.debug.json`. This standalone,
versioned source map records VM bytecode word offsets, function names, and Noir source locations;
the executable bytecode in `basic.json` remains byte-for-byte unchanged. Source paths are relative
to the Noir package root by default; pass `--absolute-paths` to retain absolute paths.

### Verifier Mode

Compute the Spartan matrix contractions `(rAw, rBw, rCw)` with caller-supplied field vectors:

```bash
mavros --root ./my-noir-project --verify --r r.json --w w.json
```

Both files contain JSON arrays of decimal field strings, for example `["3", "17", "42"]`. `r` has
one entry per R1CS constraint. `w` covers the complete witness layout in order: algebraic witnesses
(including column zero), multiplicities, challenges, table data, and lookup data. Vector lengths
must match exactly. All entries may be arbitrary, including `w[0]`; the caller supplies any protocol
randomness. Paths to vector files are relative to the current working directory.

Each string must be canonical: unsigned decimal without leading zeros (except `"0"`), and strictly
less than the field modulus. Values are rejected rather than reduced modulo the field.

Verifier mode requires a single binary package. For a multi-binary workspace, select a member with
`--root` or configure `default-member` in `Nargo.toml`; otherwise the command fails before compiling
any member or writing a result.

The command compiles the project and prints a JSON object with `r_aw`, `r_bw`, and `r_cw` as decimal
strings. It needs no `Prover.toml` and does not check R1CS satisfaction. This operation is a
component of a verifier, not a complete proof verification protocol.

Prover artifacts contain only witgen and AD. To additionally emit a separate verifier artifact:

```bash
mavros compile ./my-noir-project --verifier-output target/verifier.json
```

`verifier.json` contains only verifier bytecode plus `witness_layout` and `constraints_layout`; it
contains no matrix rows or prover entry points. `--include-debug-info` also writes
`verifier.debug.json`. Default compilation emits no verifier code. `--verify` compiles and runs only
the verifier artifact.

Rust callers compile it with `mavros_compiler::api::compile_verifier_bytecode(&mut driver, options)`
and pass that binary to
`run_verifier_from_binary(&binary, &r, &w, witness_layout, constraints_layout, None)`. No R1CS
matrices are required at runtime. Both artifacts must come from the same circuit compilation;
applications must associate the verifier artifact with their trusted circuit configuration. Verifier
execution uses a dedicated forward bytecode entry point: witness writes read the next value from
`w`, linear combinations use field arithmetic, and constraints accumulate three scalars weighted by
`r`. It allocates no AD expression nodes or matrix output vectors. Lookup and table constraints are
evaluated from the same layouts. Prover bytecode remains compatible with existing witgen/AD callers.
LLVM and WASM artifacts still expose the existing witgen and AD entry points; the forward verifier
is currently available through the VM API and CLI.

### Per-Function FlameGraphs

Run the full compiler and VM pipeline with `--profile` to generate interactive per-function
FlameGraphs:

```bash
mavros --root . --profile
```

The command writes folded-stack inputs and SVGs under `mavros_debug/flamegraphs/` for four
dimensions: `constraint_size`, `witness_size`, `witgen_time`, and `ad_time`. The two time profiles
use the VM's deterministic simulated instruction count rather than wall-clock samples, so profiles
can be compared across machines and runs. Profiling automatically enables the VM debug mapping
needed to recover Noir function names. Shared lookup-table allocations appear under a separate
`<lookup tables>` branch; the cost of each lookup remains attributed to its callsite.

The command also writes `witgen_time.cpuprofile` and `ad_time.cpuprofile`. Load either file from the
Chrome DevTools Performance panel to inspect it with the DevTools flame chart and call tree. These
profiles preserve execution order and use deterministic instruction intervals rather than wall-clock
time. Runs of up to 100,000 instructions contain one sample per instruction; larger runs use
bounded, deterministic stratified sampling to avoid periodic sampling bias while keeping file size
practical.

SVG rendering uses Brendan Gregg's [FlameGraph](https://github.com/brendangregg/FlameGraph)
`flamegraph.pl`. It is pinned and available in `nix develop`; outside the Nix shell, install
FlameGraph and put `flamegraph.pl` on `PATH`. When the script is unavailable, Mavros still writes
the folded profiles and Chrome `.cpuprofile` files and skips only the SVGs.

For advanced usage and the CLI options, run `mavros --help`. To output the witness generation binary
for WASM-capable platforms, please see the [WASM output](./docs/CONTRIBUTING.md#WASM%20Output)
section in our contributing docs.

### Noir Workspaces

When pointed at a workspace, Mavros honors `default-member`. Without a default, the CLI runs or
compiles every binary member in manifest order; library members remain available as dependencies.
Each binary uses its own `Prover.toml` and `mavros_debug` directory. A failed member makes the
workspace run fail, while the remaining members are still checked.

When `mavros compile` resolves a package root different from the supplied path, relative output
paths use that package root, including for a single default member. The defaults then produce
`<member>/target/basic.json` and `<member>/target/r1cs.bin`. Absolute output paths and paths
containing `..` are rejected for multi-binary compilations. Point Mavros directly at a member
directory to compile just that package with the usual output-path behavior.

The functional test runner reports each selected binary workspace member as a separate test, with
its own checks, circuit size, and artifact sizes. Libraries are not separate executable tests.

### Usage Example

```bash
cd my-noir-project
mavros compile
# → target/basic.json
# → target/r1cs.bin
```

## Noir Feature Support and Completeness

For the current status of Mavros' support for various Noir constructs, see the [status](./STATUS.md)
doc that is generated by CI on every update. This provides information on how far the Mavros
pipeline gets for each Noir test.
