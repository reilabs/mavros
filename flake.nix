{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    # Enumerated explicitly rather than via `eachDefaultSystem`, which also includes
    # `x86_64-darwin`. Nixpkgs 26.11 dropped support for that platform, so evaluating an
    # `x86_64-darwin` output throws and takes whole-flake evaluation (`nix flake show`,
    # `nix flake check`) down with it.
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ] (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };
        llvmPackages = pkgs.llvmPackages_22;
        noBrew = pkgs.writeShellScriptBin "brew" ''
          echo "brew is intentionally unavailable inside this Nix dev shell" >&2
          exit 127
        '';
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            noBrew

            # Rust Toolchain
            (rust-bin.nightly.latest.default.override {
              extensions = [ 
                "clippy" 
                "rust-analyzer" 
                "rust-src" 
                "rustfmt" 
              ];
              targets = [ "wasm32-unknown-unknown" ];
            })

            # Libraries
            fontconfig
            libffi
            libiconv
            libxml2
            ncurses
            openssl
            zlib

            # LLVM Dependencies
            llvmPackages.llvm
            llvmPackages.libclang
            llvmPackages.lld

            # Tools
            dprint
            git
            graphviz
            flamegraph
            nodejs_26
            pkg-config
          ];

          LLVM_SYS_221_PREFIX = "${pkgs.lib.getDev llvmPackages.libllvm}";
          # The WASM backend shells out to `wasm-ld`. Pin it to this shell's lld rather than
          # letting it resolve off `$PATH`, so the linker tracks the flake like everything else.
          # Note this cannot come from `LLVM_SYS_221_PREFIX`: that points at libllvm's `dev`
          # output, and lld is a separate derivation that ships no `bin/wasm-ld`.
          WASM_LD = "${llvmPackages.lld}/bin/wasm-ld";
          LIBCLANG_PATH = "${pkgs.lib.getLib llvmPackages.libclang}";
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.fontconfig.lib ];
        };
      }
    );
}
