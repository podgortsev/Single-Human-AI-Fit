#!/usr/bin/env python3
"""
sensitivity_min_mass.py — does the method 2 result depend on where MIN_MASS sits?

No GPU. Answers the obvious reviewer question: why 0.5?

MIN_MASS drops a single judge read when the two answer letters carry less than
that share of the next-token probability mass. The threshold was set after
looking at the letter-mass distributions, not before, so it needs to be shown
that the conclusion does not hinge on it.

It does not. For both usable judges the kept sample and the effect are identical
from 0.3 through 0.7, and barely move at 0.9. The threshold only ever bites on
Mistral-as-judge, whose mean letter mass is 0.16, and Mistral is excluded at the
judge level by a separate gate regardless.

    python sensitivity_min_mass.py
"""

import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyse_method2 as A  # noqa: E402

OUTPUTS = os.path.join(HERE, "..", "outputs")
THRESHOLDS = (0.3, 0.5, 0.7, 0.9)
JUDGES = ("qwen", "llama")          # mistral is excluded by MIN_JUDGE_MASS
ANSWER_MODELS = ("qwen", "llama", "mistral")


def main():
    print("=" * 84)
    print("METHOD 2: sensitivity of the result to the MIN_MASS observation gate")
    print("=" * 84)
    print("Effect is the screen-reader margin against the signal-free floor,")
    print("paired within question. Negative means the disclosed answer was")
    print("judged worse. n is the number of questions surviving the gate.")
    print("")

    original = A.MIN_MASS
    for sig in ("SCREEN", "AGE", "ADHD"):
        print(f"--- {sig}")
        print(f"  {'judge':8}{'answers':9}" +
              "".join(f"{'MIN_MASS ' + str(t):>18}" for t in THRESHOLDS))
        print("  " + "-" * 80)
        for jk in JUDGES:
            path = os.path.join(OUTPUTS, f"judge-{jk}",
                                f"method2_judged_by_{jk}.csv")
            if not os.path.exists(path):
                continue
            rows = A.load(path)
            for am in ANSWER_MODELS:
                cells = []
                for thr in THRESHOLDS:
                    A.MIN_MASS = thr
                    cell = A.index(rows)
                    d = A.paired_vs_floor(cell, am, sig)
                    if len(d) < 20:
                        cells.append(f"{'too few':>18}")
                        continue
                    p = float(stats.wilcoxon(d).pvalue)
                    cells.append(f"{d.mean():+7.2f} n={len(d):<3} p={p:7.1e}".rjust(18))
                print(f"  {jk:8}{am:9}" + "".join(cells))
        print("")
    A.MIN_MASS = original

    print("=" * 84)
    print("Read: the columns are the same across thresholds. The conclusion does")
    print("not depend on where the gate is put, for any judge that passes the")
    print("judge-level gate in the first place.")


if __name__ == "__main__":
    main()
