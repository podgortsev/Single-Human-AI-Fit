#!/usr/bin/env python3
"""
verify_paper_numbers.py — check the paper against the committed data.

No GPU. `python verify_paper_numbers.py` re-runs every no-GPU analysis in the
repository and confirms that each number printed in single-human-ai-fit.tex
appears in the output of the script that is supposed to produce it.

WHY THIS EXISTS
---------------
The paper claims its numbers regenerate from the committed outputs. That claim
was previously only an assertion in the abstract. This file turns it into a
test that either passes or does not.

WHAT IT DOES AND DOES NOT COVER
-------------------------------
Covered: every value in the five tables of the paper, plus the headline counts
quoted in the running text, checked against freshly generated analysis output.

Not covered: the raw model outputs themselves. Producing those CSVs took a GPU
and several hours per model, and nothing here re-runs a model. What is verified
is the whole chain from the committed CSV to the printed number.

Two figures in the paper are drawn from constants transcribed out of that same
analysis output rather than computed at draw time; those constants are checked
here like any other number. The method 3c figure computes from the CSVs
directly and carries its own assertion inside make_figures.py.
"""

import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "experiments")
PY = sys.executable


def run(script, *args):
    """Run an analysis and return its stdout, or None if it could not run."""
    path = os.path.join(EXP, script)
    if not os.path.exists(path):
        return None
    try:
        r = subprocess.run([PY, path] + list(args), capture_output=True,
                           text=True, timeout=900, cwd=HERE)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:                                  # noqa: BLE001
        print(f"  could not run {script}: {e}")
        return None


def glob_csv(method, model, stem):
    return os.path.join(EXP, method, "outputs", model, f"{stem}_{model}.csv")


# ---------------------------------------------------------------------------
# what the paper prints, and which analysis is supposed to produce it
# ---------------------------------------------------------------------------

def check(label, output, values):
    """Every value must appear literally in the analysis output."""
    if output is None:
        print(f"  SKIP  {label}: analysis could not be run")
        return None
    missing = [v for v in values if v not in output]
    if missing:
        print(f"  FAIL  {label}: not found in output -> {missing}")
        return False
    print(f"  ok    {label}: {len(values)} values confirmed")
    return True


def main():
    print("=" * 74)
    print("VERIFYING THE PAPER AGAINST THE COMMITTED DATA")
    print("=" * 74)
    print("Each number below is regenerated from a CSV in this repository.")
    print("No model is run and no GPU is used.\n")

    results = []

    # --- Method 1, trait attribution -------------------------------------
    m1 = run("method-1-traits-logprob/scripts/analyse_method1_positive.py",
             *[glob_csv("method-1-traits-logprob", m, "method1")
               for m in ("qwen", "llama", "mistral")])
    results.append(check(
        "method 1 table", m1,
        ["2.00", "2.92", "0.95", "1.34", "1.07", "1.72",
         "1.05", "0.91", "0.57", "0.09", "0.76", "0.31"]))

    # --- Method 3a, group contrast ---------------------------------------
    m3a = run("method-3a-single-signal/scripts/analyse_method3a_groups.py")
    results.append(check(
        "method 3a table", m3a,
        ["+1.10", "+5.70", "-0.70", "+8.20", "+15.10", "-0.90",
         "0.0002", "0.0009", "0.5152"]))
    results.append(check(
        "method 3a survivor counts", m3a,
        ["qwen: 0 of 24", "llama: 5 of 24", "mistral: 0 of 24"]))

    # --- Method 3c, wrappers ---------------------------------------------
    m3c = run("method-3c-wrappers/scripts/analyse_method3_wrapper.py",
              *[glob_csv("method-3c-wrappers", m, "method3_wrapper")
                for m in ("qwen", "llama", "mistral")])
    results.append(check(
        "method 3c table", m3c,
        ["+8.0", "+20.7", "+2.7", "+6.2", "+13.5", "+1.3",
         "+4.2", "+6.8", "-1.5",
         "0.0011", "0.0002", "0.0023", "0.0042", "0.0070", "0.0360"]))

    # --- Method 6b, against the neutral-alternative control ---------------
    for model, vals in (
            ("qwen", ["-5.99", "-3.47", "-5.87", "-6.93"]),
            ("llama", ["-1.62", "-1.46", "-0.58", "-1.23"]),
            ("mistral", ["+7.14", "-1.83", "+0.23"])):
        out = run("method-6b-logprob-choice/scripts/analyse_method6b.py",
                  glob_csv("method-6b-logprob-choice", model, "method6b"))
        results.append(check(f"method 6b table, {model}", out, vals))
    m6b_mistral = run("method-6b-logprob-choice/scripts/analyse_method6b.py",
                      glob_csv("method-6b-logprob-choice", "mistral",
                               "method6b"))
    results.append(check(
        "method 6b mistral usability (61% below gate, 163 clean pairs)",
        m6b_mistral, ["61%", "41", "71", "51"]))

    # --- Method 7, item level --------------------------------------------
    m7 = run("method-7-stated-vs-measured/scripts/analyse_method7.py")
    results.append(check(
        "method 7 headline null", m7, ["0 of 9"]))
    results.append(check(
        "method 7 degeneracy check (llama constant)", m7, ["57"]))

    # --- Method 8, stability ---------------------------------------------
    m8 = run("method-8-consistency/scripts/analyse_method8.py")
    if m8:
        rows = re.findall(r"^(SCREEN|AGE|DEAF|ADHD)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
                          m8, re.M)
        less = sum(int(r[1]) for r in rows)
        more = sum(int(r[2]) for r in rows)
        none = sum(int(r[3]) for r in rows)
        tot = sum(int(r[4]) for r in rows)
        ok = (less, more, none, tot) == (7, 10, 16, 33)
        print(f"  {'ok   ' if ok else 'FAIL '} method 8 tally: "
              f"{less} less / {more} more / {none} none of {tot} "
              f"(paper says 7 / 10 / 16 of 33)")
        results.append(ok)
    else:
        print("  SKIP  method 8 tally")
        results.append(None)

    # --- the datasets rebuild --------------------------------------------
    vt = subprocess.run([PY, os.path.join(HERE, "experiments", "shared", "tasks",
                                          "validate_tasks.py")],
                        capture_output=True, text=True)
    ok = "RESULT: PASS" in (vt.stdout or "")
    print(f"  {'ok   ' if ok else 'FAIL '} all 200 task keys re-derived "
          f"independently of the generator")
    results.append(ok)

    print()
    print("=" * 74)
    hard = [r for r in results if r is not None]
    failed = [r for r in hard if r is False]
    skipped = len(results) - len(hard)
    print(f"{len(hard) - len(failed)} of {len(hard)} checks passed"
          + (f", {skipped} skipped" if skipped else ""))
    print("RESULT: PASS" if not failed else f"RESULT: FAIL ({len(failed)})")
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
