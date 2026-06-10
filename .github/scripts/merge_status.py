#!/usr/bin/env python3
"""Merge several partial STATUS.md files (one per CI shard) into one table.

The two-line markdown header is taken from the first input; every data row from
every input is collected and re-sorted by test name (the first table cell) so
the merged output matches the ordering an unsharded run would produce.

This is a faithful port of the test-runner's `--merge` mode; keeping it as a
standalone script lets the CI merge job run without the Nix/LLVM toolchain.

Usage: merge_status.py --output <out> <part1> <part2> [...]
"""

import sys


def name_of(row):
    """The test name is the first non-empty pipe-delimited cell."""
    parts = row.split("|")
    return parts[1].strip() if len(parts) > 1 else ""


def main(argv):
    if "--output" not in argv:
        print("usage: merge_status.py --output <out> <part1> <part2> [...]", file=sys.stderr)
        return 2
    oi = argv.index("--output")
    output_path = argv[oi + 1]
    parts = [a for i, a in enumerate(argv[1:], start=1) if i not in (oi, oi + 1)]
    if not parts:
        print("merge_status.py requires at least one input file", file=sys.stderr)
        return 2

    header = None
    rows = []
    for part in parts:
        with open(part, encoding="utf-8") as f:
            lines = f.read().splitlines()
        if len(lines) < 2:
            print(f"{part} is missing its header", file=sys.stderr)
            return 1
        if header is None:
            header = (lines[0], lines[1])
        rows.extend(l for l in lines[2:] if l.strip())

    rows.sort(key=name_of)

    h0, h1 = header
    out = [h0, h1] + rows
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")

    print(
        f"Merged {len(parts)} shard file(s) into {output_path} ({len(rows)} test rows)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
