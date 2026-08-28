#!/usr/bin/env python3
"""Invariant checker for the narrative editing pass.

Compares a working-tree chapter against its committed baseline
(`git show HEAD:<path>`) and reports anything the edit is not allowed to
change: citation tags, numbers, headings, code fences, table rows, and
[unverified] markers. Prose may move and reflow freely; evidence may not
be lost, altered, or invented.

    python3 scripts/narrative_invariants.py src/chapters/24-kvpack-the-format.md
Exit 0 = clean, 1 = violations (listed on stdout).
"""

import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CITE = re.compile(
    r"\[(?:receipt |claims #|ledger|docs/|crates/|scripts/|src/|third_party/|measured-numbers)[^\]]*\]"
)
NUM = re.compile(r"(?<![\w./-])\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?(?:x|×|%|ms|s|GB|GiB|MB|MiB|KB|KiB|B|Gbps|Gb/s|tok/s)?")
HEADING = re.compile(r"^#{1,6} .*$", re.M)
UNVERIFIED = re.compile(r"\[unverified\]")


def features(text: str):
    fence = text.count("```")
    tables = len([l for l in text.splitlines() if l.lstrip().startswith("|")])
    return {
        "citation tags": Counter(CITE.findall(text)),
        "numbers": Counter(NUM.findall(text)),
        "headings": Counter(HEADING.findall(text)),
        "[unverified] markers": Counter(UNVERIFIED.findall(text)),
        "code-fence delimiters": Counter({"```": fence}),
        "table rows": Counter({"|row|": tables}),
    }


def main() -> int:
    rel = sys.argv[1]
    new = (ROOT / rel).read_text()
    old = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"HEAD:{rel}"],
        capture_output=True, text=True, check=True,
    ).stdout
    ok = True
    for name, before in features(old).items():
        after = features(new)[name]
        lost = before - after
        gained = after - before
        for item, n in lost.items():
            print(f"LOST {name}: {item!r} x{n}")
            ok = False
        for item, n in gained.items():
            print(f"ADDED {name}: {item!r} x{n}")
            ok = False
    print("CLEAN" if ok else "VIOLATIONS — fix before finishing")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
