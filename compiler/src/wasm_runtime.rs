//! Locating or building the wasm-runtime static library that WASM artifacts link against.
//!
//! This deliberately lives outside the compiler: codegen takes the library path as an input
//! ([`WasmCompileOpts::runtime_lib`][crate::compiler::codegen::llssa_to_llvm::WasmCompileOpts])
//! and never invokes cargo itself. Binaries (the CLI, the test runner) resolve the runtime up
//! front and thread the path through.

use std::path::PathBuf;
use std::process::Command;

/// Environment variable pointing at a pre-built wasm-runtime static library. When set,
/// [`locate_or_build`] uses it directly instead of invoking cargo.
pub const WASM_RUNTIME_LIB_ENV: &str = "MAVROS_WASM_RUNTIME_LIB";

/// Locate (via [`WASM_RUNTIME_LIB_ENV`]) or build the wasm-runtime static library.
///
/// Resolving the workspace makes cargo walk the noir git checkout, which races against parallel
/// test-runner children creating and deleting `mavros_debug` dirs for tests that live inside that
/// checkout ("failed to read directory ... mavros_debug/...: No such file or directory"). The
/// test runner therefore builds the runtime once up front and points children at the artifact
/// via [`WASM_RUNTIME_LIB_ENV`], so they never invoke cargo.
pub fn locate_or_build() -> PathBuf {
    if let Ok(lib_path) = std::env::var(WASM_RUNTIME_LIB_ENV) {
        let lib_path = PathBuf::from(lib_path);
        if lib_path.exists() {
            return lib_path;
        }
        panic!("{WASM_RUNTIME_LIB_ENV} is set but {lib_path:?} does not exist");
    }

    let metadata = cargo_metadata::MetadataCommand::new()
        .exec()
        .expect("Failed to get cargo metadata");

    let workspace_root = metadata.workspace_root.as_std_path();
    let wasm_runtime_dir = workspace_root.join("wasm-runtime");

    let output = Command::new("cargo")
        .current_dir(&wasm_runtime_dir)
        .args(["build", "--target", "wasm32-unknown-unknown", "--release"])
        .output()
        .expect("Failed to run cargo build for wasm-runtime");

    if !output.status.success() {
        eprintln!(
            "wasm-runtime build stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        panic!("Failed to build wasm-runtime for wasm32");
    }

    let lib_path = workspace_root
        .join("target")
        .join("wasm32-unknown-unknown")
        .join("release")
        .join("libmavros_wasm_runtime.a");

    if !lib_path.exists() {
        panic!("wasm-runtime library not found at {:?}", lib_path);
    }

    lib_path
}
