# LLVM Power Example

This directory shows a power-style program equivalent to `noir_tests/power` but in C + LLVM.

- `power.c`: C source with opaque `Field` handles and injected `__field_mul` / `__field_from_u64` / `__assert_eq` intrinsics.

## Importing this LLVM into mavros

The test-runner compiles `power.c` with `clang` to LLVM IR and imports the generated file.
The generated IR is written to `llvm_tests/power/mavros_debug/compiled_from_c.ll`.

For manual runs, compile C to LLVM IR first:

```bash
clang -S -emit-llvm -O0 -std=c11 llvm_tests/power/power.c -o /tmp/power.ll
cargo run -- --root noir_tests/power --llvm-ir /tmp/power.ll --skip-vm
```

The importer path currently supports a constrained subset intended for this class of examples.
