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

