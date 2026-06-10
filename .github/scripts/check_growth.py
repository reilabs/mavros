#!/usr/bin/env python3
"""Report test-pass-rate and R1CS/bytecode size changes between two STATUS.md
files, as a markdown comment.

Prints (to stdout) the overall success rate, a "Positive Changes" section
(newly-passing cells, pass-rate delta, size decreases) and a "Warnings" section
(size increases). Always exits 0 — this is informational, not a gate.

This is a faithful port of the `--check-growth` mode that used to live in the
Rust test-runner; keeping it as a standalone script lets the CI merge job run
it without the Nix/LLVM toolchain the compiled binary needs.

Usage: check_growth.py <baseline-STATUS.md> <current-STATUS.md>
"""

import sys

CHECK = "✅"

# Columns (post-empty-strip cell index) that count toward pass rates. Mirrors
# REGRESSION_COLS in the Rust test-runner.
REGRESSION_COLS = [
    1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
]


class Row:
    """One parsed STATUS.md test row."""

    __slots__ = ("name", "cells", "rows", "cols", "witgen_bytes", "ad_bytes")

    def __init__(self, cells):
        self.name = cells[0]
        self.cells = cells
        self.rows = _parse_int(cells[3])
        self.cols = _parse_int(cells[4])
        self.witgen_bytes = _parse_int(cells[5])
        self.ad_bytes = _parse_int(cells[6])


def _parse_int(s):
    """Parse an int cell, or None if it isn't one (e.g. the '-' placeholder)."""
    try:
        return int(s)
    except ValueError:
        return None


def parse_status_rows(path):
    """Parse a STATUS.md table into a list of Row, preserving file order.

    Skips the two-line markdown header, splits each row on '|', trims cells and
    drops empties, and ignores rows with fewer than 23 cells.
    """
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    result = []
    for line in lines[2:]:
        cells = [c.strip() for c in line.split("|")]
        cells = [c for c in cells if c]
        if len(cells) < 23:
            continue
        result.append(Row(cells))
    return result


def _pct(num, den):
    return num / den * 100.0 if den > 0 else 0.0


def _size_lines(name, metric, before, after, warnings, improvements):
    """Append a growth (warning) or shrink (improvement) row for one metric.

    `+{:.1}%` for growth always shows the sign; the shrink delta is already
    negative so it carries its own '-'.
    """
    if before is None or after is None:
        return
    if after > before:
        pct = (after - before) / before * 100.0
        warnings.append(
            f"| {name} | {metric} | {before} | {after} | +{after - before} ({pct:+.1f}%) |"
        )
    elif after < before:
        pct = (after - before) / before * 100.0
        improvements.append(
            f"| {name} | {metric} | {before} | {after} | {after - before} ({pct:.1f}%) |"
        )


def main(argv):
    if len(argv) != 3:
        print(
            "usage: check_growth.py <baseline-STATUS.md> <current-STATUS.md>",
            file=sys.stderr,
        )
        return 2

    baseline = parse_status_rows(argv[1])
    current = parse_status_rows(argv[2])
    base_map = {r.name: r for r in baseline}

    new_checkmarks = []  # (test_name, col_name placeholder) — col name is the index here
    existing_baseline_checkmarks = 0
    existing_current_checkmarks = 0
    existing_total = 0

    total_current_checkmarks = 0
    total_current_cells = 0

    improvements = []
    warnings = []

    # The Rust version labels new-checkmark columns by the same human names used
    # for regressions. Reconstruct that name lookup.
    col_names = {
        1: "Compiled", 2: "R1CS", 7: "Witgen Compile", 8: "Witgen Run VM",
        9: "Witgen Correct", 10: "Witgen No Leak", 11: "AD Compile",
        12: "AD Run VM", 13: "AD Correct", 14: "AD No Leak",
        15: "Witgen WASM Compile", 16: "Witgen WASM Run",
        17: "Witgen WASM Correct", 18: "Witgen WASM No Leak",
        19: "AD WASM Compile", 20: "AD WASM Run", 21: "AD WASM Correct",
        22: "AD WASM No Leak",
    }

    for cur in current:
        for col in REGRESSION_COLS:
            total_current_cells += 1
            if cur.cells[col] == CHECK:
                total_current_checkmarks += 1

        base = base_map.get(cur.name)
        if base is None:
            continue

        for col in REGRESSION_COLS:
            existing_total += 1
            base_pass = base.cells[col] == CHECK
            cur_pass = cur.cells[col] == CHECK
            if base_pass:
                existing_baseline_checkmarks += 1
            if cur_pass:
                existing_current_checkmarks += 1
            if not base_pass and cur_pass:
                new_checkmarks.append((cur.name, col_names[col]))

        _size_lines(cur.name, "Constraints", base.rows, cur.rows, warnings, improvements)
        _size_lines(cur.name, "Witnesses", base.cols, cur.cols, warnings, improvements)
        _size_lines(cur.name, "Witgen Size (bytes)", base.witgen_bytes, cur.witgen_bytes, warnings, improvements)
        _size_lines(cur.name, "AD Size (bytes)", base.ad_bytes, cur.ad_bytes, warnings, improvements)

    existing_baseline_pct = _pct(existing_baseline_checkmarks, existing_total)
    existing_current_pct = _pct(existing_current_checkmarks, existing_total)
    existing_pct_change = existing_current_pct - existing_baseline_pct
    total_current_pct = _pct(total_current_checkmarks, total_current_cells)

    out = []
    out.append(f"**Overall success rate on test cases: {total_current_pct:.1f}%**\n")

    has_positive_news = (
        bool(new_checkmarks) or existing_pct_change > 0.0 or bool(improvements)
    )
    if has_positive_news:
        out.append("### Positive Changes\n")

        if new_checkmarks or abs(existing_pct_change) > 0.001:
            if new_checkmarks:
                out.append(
                    f"<details>\n<summary>{len(new_checkmarks)} cell(s) turned into checkmarks ✅</summary>\n"
                )
                for test, col in new_checkmarks:
                    out.append(f"- **{test}** / {col}")
                out.append("\n</details>\n")
            if abs(existing_pct_change) > 0.001:
                out.append(
                    f"- Existing tests: {existing_baseline_pct:.1f}% → "
                    f"{existing_current_pct:.1f}% ({existing_pct_change:+.1f}%)"
                )
            out.append("")

        if improvements:
            out.append("<details>")
            out.append("<summary><b>R1CS/bytecode size decreased</b></summary>\n")
            out.append("| Test | Metric | Before | After | Change |")
            out.append("|------|--------|--------|-------|--------|")
            out.extend(improvements)
            out.append("\n</details>\n")

    if not warnings:
        if not has_positive_news:
            out.append("No test improvements or R1CS/bytecode size changes detected.")
        else:
            out.append("No R1CS/bytecode size growth detected.")
        print("\n".join(out))
        return 0

    out.append("### Warnings\n")
    out.append("<details>")
    out.append("<summary><b>R1CS/bytecode size growth detected</b></summary>\n")
    out.append("| Test | Metric | Before | After | Change |")
    out.append("|------|--------|--------|-------|--------|")
    out.extend(warnings)
    out.append("\n</details>")

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
