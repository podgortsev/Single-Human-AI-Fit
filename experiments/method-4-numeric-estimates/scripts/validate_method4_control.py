#!/usr/bin/env python3
"""
validate_method4_control.py — offline check of the control analysis. No GPU,
no model, no network.

Builds synthetic CSVs in the shape the two runs produce, with answers known in
advance, and asserts that analyse_method4_control.py reads them the way it is
supposed to. Three cases, each the kind of mistake that would otherwise be
found only after an hour of Colab.

  A. REFERENCE DECIDES IT. The disclosure sits exactly between the two neutral
     clauses: higher than commuting, no different from cycling. The analysis
     must call the cell REFERENCE DECIDES IT, because the original result came
     from the clause being compared against.

  B. REAL EFFECT. The disclosure is higher than both neutral clauses by the
     same amount. The analysis must call it holds.

  C. NULL. The disclosure matches both neutral clauses. Nothing may be called
     significant.

Run:
    python validate_method4_control.py
"""

import csv
import os
import random
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSE = os.path.join(HERE, "analyse_method4_control.py")
N = 100
MEASURES = [("SCORE", "assessment"), ("CHURN", "probability")]
DISCLOSURES = ["SCREEN", "AGE", "DEAF", "ADHD"]


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "measure", "kind", "signal", "case", "value",
                    "status", "raw_output"])
        w.writerows(rows)


def build(tmp, case):
    """case: 'reference', 'real' or 'null'."""
    rng = random.Random(7)
    main_rows, ctrl_rows = [], []
    for mid, kind in MEASURES:
        for c in range(N):
            base = 50 + rng.gauss(0, 3)
            # the two neutral clauses differ by 6 in 'reference', else not
            commute = base
            cycling = base + (6 if case == "reference" else 0)
            if case == "reference":
                disclosed = cycling          # level with cycling, above commute
            elif case == "real":
                disclosed = base + 6         # above both
            else:
                disclosed = base             # level with both
            main_rows.append(["m", mid, kind, "NONE", c, round(commute, 2),
                              "ok", ""])
            ctrl_rows.append(["m", mid, kind, "CTRL_CYCLE", c,
                              round(cycling, 2), "ok", ""])
            for s in DISCLOSURES:
                main_rows.append(["m", mid, kind, s, c, round(disclosed, 2),
                                  "ok", ""])
    write(os.path.join(tmp, "method4_qwen.csv"), main_rows)
    write(os.path.join(tmp, "method4_control_qwen.csv"), ctrl_rows)


def run(tmp):
    out = subprocess.run([sys.executable, ANALYSE, "--dir", tmp,
                          "--model", "qwen"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout)
        print(out.stderr)
        sys.exit("analysis script failed")
    return out.stdout


def verdicts(text):
    """Just the verdict column of section 2, so prose cannot be mistaken for a
    result."""
    out = []
    for line in text.splitlines():
        for v in ("REFERENCE DECIDES IT", "same way, not significant",
                  "no movement either way", "holds"):
            if line.rstrip().endswith(v) and line.startswith("   ")                     and any(ch.isdigit() for ch in line):
                out.append(v)
                break
    return out


def check(name, text, must, must_not):
    bad = [m for m in must if m not in text]
    worse = [m for m in must_not if m in text]
    if bad or worse:
        print(text)
        if bad:
            print(f"FAIL {name}: expected but missing: {bad}")
        if worse:
            print(f"FAIL {name}: present but should not be: {worse}")
        sys.exit(1)
    print(f"  pass: {name}")


def main():
    print("validating analyse_method4_control.py on synthetic data\n")

    with tempfile.TemporaryDirectory() as tmp:
        build(tmp, "reference")
        text = run(tmp)
        v = verdicts(text)
        if set(v) != {"REFERENCE DECIDES IT"} or not v:
            print(text)
            sys.exit(f"FAIL: expected every verdict to be "
                     f"REFERENCE DECIDES IT, got {sorted(set(v))}")
        print(f"  pass: all {len(v)} cells called REFERENCE DECIDES IT")
        check("the two neutral clauses are reported as different",
              text, ["move between two clauses"], [])
        if len(v) != len(MEASURES) * len(DISCLOSURES):
            print(text)
            sys.exit(f"FAIL: expected {len(MEASURES) * len(DISCLOSURES)} "
                     f"cells, got {len(v)}")

    with tempfile.TemporaryDirectory() as tmp:
        build(tmp, "real")
        text = run(tmp)
        v = verdicts(text)
        if set(v) != {"holds"} or not v:
            print(text)
            sys.exit(f"FAIL: expected every verdict to be holds, "
                     f"got {sorted(set(v))}")
        print(f"  pass: all {len(v)} cells hold under both references")

    with tempfile.TemporaryDirectory() as tmp:
        build(tmp, "null")
        text = run(tmp)
        v = verdicts(text)
        if set(v) - {"no movement either way"}:
            print(text)
            sys.exit(f"FAIL: a null produced verdicts {sorted(set(v))}")
        print(f"  pass: null produced no finding ({sorted(set(v)) or 'no cells'})")

    print("\nall checks passed")


if __name__ == "__main__":
    main()
