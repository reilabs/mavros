#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# WASM tests otherwise invoke Cargo repeatedly and contend on its build lock. Read the
# artifact path from Cargo so CARGO_TARGET_DIR and Cargo's target-dir config both work.
runtime_lib=$(
    cargo build --locked --package mavros-wasm-runtime \
        --target wasm32-unknown-unknown --release --message-format=json |
        jq --slurp --exit-status --raw-output '
            [.[] | select(.reason == "compiler-artifact")
             | select(.target.name == "mavros_wasm_runtime")
             | .filenames[] | select(endswith(".a"))] | last
        '
)
test -f "$runtime_lib"
export MAVROS_WASM_RUNTIME_LIB="$runtime_lib"

# Keep the larger stack required by the compiler's recursive tests.
export RUST_MIN_STACK=8388608
exec cargo nextest run --locked --workspace --all-targets --all-features "$@"
