#!/usr/bin/env python3
"""Detect test-status regressions between two STATUS.md files.

A regression is any tracked cell that was a checkmark (✅) in the baseline but
is no longer a checkmark in the current run, for a test present in both. Exits
1 if any regression is found, 0 otherwise.

This is a faithful port of the `--check-regression` mode that used to live in
the Rust test-runner; keeping it as a standalone script lets the CI merge job
run it without the Nix/LLVM toolchain the compiled binary needs.

Usage: check_regression.py <baseline-STATUS.md> <current-STATUS.md>
"""

import sys

CHECK = "✅"

# (column index, human-readable name) for every cell a regression can occur in.
# Indices are into the cell list AFTER empty cells are dropped, so cell 0 is the
# test name. Mirrors REGRESSION_COLS in the Rust test-runner.
REGRESSION_COLS = [
    (1, "Compiled"),
    (2, "R1CS"),
    (7, "Witgen Compile"),
    (8, "Witgen Run VM"),
    (9, "Witgen Correct"),
    (10, "Witgen No Leak"),
    (11, "AD Compile"),
    (12, "AD Run VM"),
    (13, "AD Correct"),
    (14, "AD No Leak"),
    (15, "Witgen WASM Compile"),
    (16, "Witgen WASM Run"),
    (17, "Witgen WASM Correct"),
    (18, "Witgen WASM No Leak"),
    (19, "AD WASM Compile"),
    (20, "AD WASM Run"),
    (21, "AD WASM Correct"),
    (22, "AD WASM No Leak"),
]


def parse_status_rows(path):
    """Parse a STATUS.md table into {test_name: [cells]}.

    Skips the two-line markdown header, splits each row on '|', trims cells and
    drops empties (so a `| a | b |` row yields ['a', 'b']), and ignores rows
    with fewer than 23 cells.
    """
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    rows = {}
    for line in lines[2:]:
        cells = [c.strip() for c in line.split("|")]
        cells = [c for c in cells if c]
        if len(cells) < 23:
            continue
        rows[cells[0]] = cells
    return rows


def main(argv):
    if len(argv) != 3:
        print(
            "usage: check_regression.py <baseline-STATUS.md> <current-STATUS.md>",
            file=sys.stderr,
        )
        return 2

    baseline = parse_status_rows(argv[1])
    current = parse_status_rows(argv[2])

    regressions = []
    for name, cur_cells in current.items():
        base_cells = baseline.get(name)
        if base_cells is None:
            continue
        for col, col_name in REGRESSION_COLS:
            base_val = base_cells[col]
            cur_val = cur_cells[col]
            if base_val == CHECK and cur_val != CHECK:
                regressions.append(f"  {name} / {col_name}: {CHECK} → {cur_val}")

    if not regressions:
        print("No regressions detected.", file=sys.stderr)
        return 0

    print("REGRESSIONS DETECTED:", file=sys.stderr)
    for r in regressions:
        print(r, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
