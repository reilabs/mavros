# Contributing

This document provides a brief introduction to how you can contribute to the Mavros project. We
welcome all kinds of contributions, whether they be code or otherwise. This repository is written in
[Rust](https://rust-lang.org), though there are components written in [Noir](https://noir-lang.org)
as well.

## Building

As Mavros relies on LLVM as a dependency, we use [Nix](https://lix.systems) to encapsulate the build
environment. You can build Mavros as follows:

1. Install Nix by following the [instructions](https://lix.systems/install/) for your system. Once
   done, check you have the `nix` command in your path.
2. Run `nix develop` to drop into a development shell for the project (devshell). This will download
   all of the dependencies onto your system without polluting your normal development environment
   and then drop you into a basic bash shell that has the necessary tools on the path.
3. Inside the devshell you can run your usual commands such as `cargo build` and so on.
4. You can also run commands directly using `nix develop --command <...>` and passing your command.
   This can also be used to launch the devshell using your shell of choice.

By default the environment in the devshell inherits from your system environment, appending it after
the environment-specific paths. To control what is available in the devshell, you can pass the
following flags to the invocation. `--ignore-environment` / `-i` clears the entire parent
environment and retains only the entries specified with `--keep` / `-k`. You can also use `--unset`
/ `-u` to unset specific environment variables in the environment inherited by the devshell.

### The Makefile

This repo includes a makefile with some useful shorthand commands. In particular, it can cope with
running its embedded commands that involve builds whether it is inside or outside of the nix
devshell (e.g. you can run `make build` in your normal shell and it will spawn a devshell to run the
build, or if you are already in the devshell it will not).

- Run `make` to see a list of the available commands with help.
- Run `make shell` to launch your user `$SHELL` inside the nix environment devshell.

### Common Issues

Below is a list of common issues that may be encountered when trying to build Mavros.

- On macOS, if you have Apple's SDK `clang` and `cc` in your path ahead of the one provided by the
  flake, you may get a linker error about not being able to find `libtinfo`. Hide these from your
  path (commonly located in `/usr/bin`) inside the devshell to fix it (e.g. by using `-u` as
  described above), or checking `$IN_NIX_SHELL` in your shell configuration.

## Testing

All of the tests in the repository can be run using `make test`. There are two main types of test
for Mavros:

- The **functional tests** are encapsulated in a special test driver included in the repository.
  They are most easily run using `make func-test`.
- The **unit tests** in the codebase can be run simply using `make unit-test`.

## Debugging

This section contains a few debugging tips for working on Mavros.

- **IR Graph Output:** Passing `--draw-graphs` to the `mavros` binary generates a `mavros_debug/`
  folder in the project root. This contains an SSA diagram and CFG graph for each compiler pass.
  Doing this requires [graphviz](https://graphviz.org), which is provided in the
  [self-contained development environment](#building).

## PRs

Mavros operates on a fork-and-PR [model](https://en.wikipedia.org/wiki/Fork_and_pull_model) for
contributing new code to `main`. The workflow is as follows:

1. Fork the repository if needed.
2. Create a new branch to contain your work with a descriptive name.
3. Perform your changes and commit them (in a single commit or across multiple commits) with a
   descriptive commit message.
4. Pull-request this branch against `main` in the Mavros repository. Make sure that your code passes
   `make lint` without any warnings or errors.

## Workspace Crates

The following is a summary of the crates in the workspace and their roles.

| Crate                 | Description                                                                                                        |
| --------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `mavros` (root)       | Main compiler crate. Contains the CLI binary, compiler pipeline (SSA → R1CS), driver, and public API.              |
| `mavros-artifacts`    | Shared data types: `R1CS`, `R1C`, `WitnessLayout`, `ConstraintsLayout`, `Field`. Used by both the compiler and VM. |
| `mavros-vm`           | Bytecode interpreter that runs witness generation and automatic differentiation programs.                          |
| `opcode-gen`          | Proc-macro crate that generates VM opcode definitions.                                                             |
| `ssa-builder`         | Proc-macro crate with helpers for constructing SSA IR.                                                             |
| `mavros-wasm-runtime` | Minimal runtime compiled to WebAssembly, used when emitting `.wasm` witness generation targets.                    |

## WASM Output

Mavros can emit the witness generation binary as a [WebAssembly](https://webassembly.org) `.wasm`
file in addition to (or instead of) executing it natively. This is designed for hosts—such as
browsers or a mobile device that can only load signed code—that prefer to execute using WASM.

```bash
mavros --emit-wasm
```

This works as follows:

1. The compiler lowers the witness generator SSA to LLVM IR (using
   [`inkwell`](https://github.com/TheDan64/inkwell)/LLVM 22), compiling it for the `wasm32` target.
   `wasm32` target.
2. The generated WASM calls a small set of external functions (`__field_mul`, `__write_witness`,
   `__write_a/b/c`) that are implemented in the [`mavros-wasm-runtime`](../wasm-runtime/) crate.
   This runtime handles BN254 field arithmetic and writes outputs into memory.
3. A `.wasm.meta.json` sidecar file is written alongside the `.wasm` with the ABI and R1CS layout
   info, so the host knows how to set up the witness generator's inputs and interpret its outputs.

Running with `--emit-wasm` will execute the full compilation pipeline, and _additionall_ emit the
`mavros_debug/witgen.wasm` and `mavros_debug/witgen.wasm.meta.json`. This can be combined with the
`--emit-llvm` flag to also save the intermediate LLVM IR as an `.ll` file.

> **Note:** WASM output is only currently available via the default command (no subcommand), not via
> `mavros compile`.
