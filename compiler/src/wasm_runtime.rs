//! Locating or building the wasm-runtime static library that WASM artifacts link against.
//!
//! This deliberately lives outside the compiler: codegen takes the library path as an input
//! ([`WasmCompileOpts::runtime_lib`][crate::compiler::codegen::llssa_to_llvm::WasmCompileOpts])
//! and never invokes cargo itself. Binaries (the CLI, the test runner) resolve the runtime up
//! front and thread the path through.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Environment variable pointing at a pre-built wasm-runtime static library. When set,
/// [`locate_or_build`] uses it directly instead of invoking cargo.
pub const WASM_RUNTIME_LIB_ENV: &str = "MAVROS_WASM_RUNTIME_LIB";

/// Build a Wasmtime module from compact executable bytes and a separate DWARF companion.
///
/// Wasmtime 31 consumes embedded DWARF but does not follow the browser-oriented
/// `external_debug_info` section itself, so this reconstructs an in-memory module for compilation.
/// The executable file on disk remains stripped.
pub fn wasmtime_module_with_debug_info(
    engine: &wasmtime::Engine,
    executable: &[u8],
    debug: &[u8],
) -> wasmtime::Result<wasmtime::Module> {
    let merged =
        crate::wasm_debug::merge_debug_info(executable, debug).map_err(wasmtime::Error::msg)?;
    wasmtime::Module::new(engine, merged)
}

/// Load a Wasmtime module from explicitly supplied executable and debug-sidecar paths.
pub fn load_wasmtime_module_with_debug_info(
    engine: &wasmtime::Engine,
    wasm_path: impl AsRef<Path>,
    debug_path: impl AsRef<Path>,
) -> wasmtime::Result<wasmtime::Module> {
    let wasm_path = wasm_path.as_ref();
    let debug_path = debug_path.as_ref();
    let executable = std::fs::read(wasm_path).map_err(|error| {
        wasmtime::Error::msg(format!(
            "failed to read WASM executable {}: {error}",
            wasm_path.display()
        ))
    })?;
    let debug = std::fs::read(debug_path).map_err(|error| {
        wasmtime::Error::msg(format!(
            "failed to read WASM debug sidecar {}: {error}",
            debug_path.display()
        ))
    })?;
    wasmtime_module_with_debug_info(engine, &executable, &debug)
}

/// Load a Wasmtime module, automatically following a local `external_debug_info` sidecar when the
/// executable names one. Modules without an external-debug section load normally.
pub fn load_wasmtime_module(
    engine: &wasmtime::Engine,
    wasm_path: impl AsRef<Path>,
) -> wasmtime::Result<wasmtime::Module> {
    let wasm_path = wasm_path.as_ref();
    let executable = std::fs::read(wasm_path).map_err(|error| {
        wasmtime::Error::msg(format!(
            "failed to read WASM executable {}: {error}",
            wasm_path.display()
        ))
    })?;
    let Some(url) =
        crate::wasm_debug::external_debug_info_url(&executable).map_err(wasmtime::Error::msg)?
    else {
        return wasmtime::Module::new(engine, executable);
    };
    if url.contains("://") {
        return Err(wasmtime::Error::msg(format!(
            "Wasmtime loader cannot fetch remote external debug URL {url:?}"
        )));
    }
    let debug_path = wasm_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(url);
    let debug = std::fs::read(&debug_path).map_err(|error| {
        wasmtime::Error::msg(format!(
            "failed to read referenced WASM debug sidecar {}: {error}",
            debug_path.display()
        ))
    })?;
    wasmtime_module_with_debug_info(engine, &executable, &debug)
}

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

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_module_with_empty_dwarf() -> Vec<u8> {
        let mut module = b"\0asm\x01\0\0\0".to_vec();
        // () -> () type, one function of that type, and an empty function body.
        module.extend_from_slice(&[1, 4, 1, 0x60, 0, 0]);
        module.extend_from_slice(&[3, 2, 1, 0]);
        module.extend_from_slice(&[10, 4, 1, 2, 0, 0x0b]);
        module.extend_from_slice(&[0, 12, 11]);
        module.extend_from_slice(b".debug_info");
        module
    }

    #[test]
    fn wasmtime_loaders_accept_explicit_and_referenced_sidecars() {
        let dir = tempfile::tempdir().unwrap();
        let executable_path = dir.path().join("program.wasm");
        let debug_path = dir.path().join("program.debug.wasm");
        let (executable, debug) = crate::wasm_debug::split_debug_info(
            &minimal_module_with_empty_dwarf(),
            "program.debug.wasm",
        )
        .unwrap();
        std::fs::write(&executable_path, executable).unwrap();
        std::fs::write(&debug_path, debug).unwrap();

        let engine = wasmtime::Engine::default();
        load_wasmtime_module_with_debug_info(&engine, &executable_path, &debug_path).unwrap();
        load_wasmtime_module(&engine, &executable_path).unwrap();
    }
}
