.DEFAULT_GOAL := help

# This is set if inside the devshell, so any command that MUST run in the devshell should have
# $(SHELL_WRAPPER) prefixed.
ifeq ($(IN_NIX_SHELL),)
		SHELL_WRAPPER := nix develop --command
else
		SHELL_WRAPPER :=
endif

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# -- Building -------------------------------------------------------------------------------------

.PHONY: build
build: ## Build Mavros for testing
	$(SHELL_WRAPPER) cargo build --all-targets --all-features

.PHONY: build-test-runner
build-test-runner: ## Builds the test runner in release mode
	$(SHELL_WRAPPER) cargo build --release --bin test-runner

.PHONY: release
release: ## Build Mavros in release mode
	$(SHELL_WRAPPER) cargo build --release --all-targets --all-features

# -- Testing --------------------------------------------------------------------------------------

.PHONY: unit-test
unit-test: ## Run the unit tests
	$(SHELL_WRAPPER) cargo test --all-targets --all-features

.PHONY: func-test
func-test: ## Run the functional test harness
	$(SHELL_WRAPPER) cargo run --release --bin test-runner -- --output STATUS.md

.PHONY: test
test: unit-test func-test ## Run all the tests

.PHONY: install
install: ## Builds the Mavros CLI and installs it to your cargo binary path
	cargo install --locked --bin mavros --path .

# -- Linting --------------------------------------------------------------------------------------

.PHONY: clippy
clippy: ## Lint the codebase with clippy
	$(SHELL_WRAPPER) cargo clippy --all-targets --all-features

.PHONY: format-check-rust
format-check-rust: ## Check rust formatting without changing files (reformat with `make format-rust`)
	$(SHELL_WRAPPER) cargo fmt --all --check

.PHONY: format-check-docs
format-check-docs: ## Check docs and config formatting without changing files (reformat with `make format-docs`)
	$(SHELL_WRAPPER) dprint check

.PHONY: format-check
format-check: format-check-rust format-check-docs ## Check code, docs, and config formatting without changing files (reformat with `make format`)

.PHONY: lint
lint: format-check clippy ## Run all the linting tasks

# -- Utility --------------------------------------------------------------------------------------

.PHONY: format-rust
format-rust: ## Format the codebase with rustfmt
	$(SHELL_WRAPPER) cargo fmt --all

.PHONY: format-docs
format-docs: ## Format the docs and configs
	$(SHELL_WRAPPER) dprint fmt

.PHONY: format
format: format-docs format-rust ## Formats code, documentation, and configuration files.

.PHONY: clean
clean: ## Clean all build artifacts in the project.
	cargo clean

.PHONY: shell
shell: ## Launch the user's `$SHELL` inside the devshell
	nix develop --command $(shell echo $$SHELL)

.PHONY: editor
editor: ## Launch the user's `$EDITOR` inside the devshell
	nix develop --command $(EDITOR)

