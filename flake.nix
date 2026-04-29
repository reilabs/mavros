{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };
        llvmPackages = pkgs.llvmPackages_22;
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            (rust-bin.nightly.latest.default.override {
              extensions = [ 
                "clippy" 
                "rust-analyzer" 
                "rust-src" 
                "rustfmt" 
              ];
              targets = [ "wasm32-unknown-unknown" ];
            })
            llvmPackages.llvm
            llvmPackages.libclang
            llvmPackages.lld
            pkg-config
            openssl
            nodejs_25
            libffi
            libxml2
            ncurses
            zlib
            git
            fontconfig
            libiconv
            graphviz
          ];

          LLVM_SYS_221_PREFIX = "${pkgs.lib.getDev llvmPackages.libllvm}";
          LIBCLANG_PATH = "${pkgs.lib.getLib llvmPackages.libclang}";
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.fontconfig.lib ];
        };
      }
    );
}
