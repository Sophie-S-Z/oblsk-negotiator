"""
Regenerate the behavior-tree characterization fixture.

Run this only on purpose, when a deliberate change to the tree's behavior means
the recorded decisions should move. It overwrites golden_decisions.json with the
current tree's output over the parity battery. After running, eyeball the git
diff: every change should be one you intended.

    python tests/generate_golden.py
"""

from __future__ import annotations
import os, sys, json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.parity_harness import run_battery

OUT = os.path.join(os.path.dirname(__file__), "golden_decisions.json")


def main() -> None:
    records = run_battery()
    with open(OUT, "w") as f:
        json.dump(records, f, indent=1, sort_keys=True)
    steps = sum(len(r["steps"]) for r in records)
    print(f"wrote {OUT}: {len(records)} scenarios, {steps} steps")


if __name__ == "__main__":
    main()
