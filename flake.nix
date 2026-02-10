{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };
        llvmPackages = pkgs.llvmPackages_18;

        darwinDeps = pkgs.lib.optionals pkgs.stdenv.isDarwin (with pkgs.darwin.apple_sdk.frameworks; [
          CoreText
          CoreGraphics
          CoreFoundation
          Security
          SystemConfiguration
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            (rust-bin.nightly.latest.default.override {
              extensions = [ "rust-src" "rust-analyzer" ];
              targets = [ "wasm32-unknown-unknown" ];
            })
            llvmPackages.llvm
            llvmPackages.libclang
            llvmPackages.lld
            pkg-config
            openssl
            nodejs_20
            libffi
            libxml2
            ncurses
            zlib
            git
            fontconfig
            libiconv
          ] ++ darwinDeps;

          LLVM_SYS_180_PREFIX = "${llvmPackages.llvm.dev}";
          LIBCLANG_PATH = "${llvmPackages.libclang.lib}/lib";
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.fontconfig.lib ];
        };
      }
    );
}
