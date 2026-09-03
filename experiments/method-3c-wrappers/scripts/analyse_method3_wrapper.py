#!/usr/bin/env python3
"""
analyse_method3_wrapper.py — corrected analysis of the wrapper experiment.

No GPU needed. Runs on the CSVs already saved, in seconds.

    python analyse_method3_wrapper.py method3_wrapper_*.csv

TWO THINGS THE FIRST VERSION GOT WRONG
--------------------------------------
1. It divided the spread of one signal across wrappers by the spread between
   the three signals, and read a ratio near one as "the measurement is not
   reporting a property of the signal."

   That was wrong. The three signals happen to be similar in size, so the
   denominator is small by construction. The ratio says the signals resemble
   each other; it says nothing about whether each is measured reliably.

   The right question: six wrappers give six estimates of the same quantity.
   Treat them as a sample. Mean, standard error, confidence interval. If the
   interval excludes zero, the effect survives the choice of wrapper.

2. It counted a net of exactly zero as a sign flip. Zero is neither direction.
   Zeros are now reported separately and excluded from the sign test.

WHAT IS REPORTED
----------------
  per wrapper   exact McNemar, a binomial test on discordant pairs, with
                Benjamini-Hochberg correction across all wrapper-by-signal tests
  per signal    the six estimates as a sample: mean, SD, SE, 95 percent CI,
                one-sample t-test against zero
  across models a sign test on every measurement of each signal
  wrapper drift how far the baseline alone moves between wrappers, which bounds
                what any single-wrapper study could resolve
"""

import csv
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

WRAPPER_TEXT = {
    "W1": "I have a question I need some help with today",
    "W2": "There is something I would like to ask you about",
    "W3": "Could you help me with the following, please",
    "W4": "I am trying to work something out and could use a hand",
    "W5": "Quick question for you",
    "W6": "Here is what I need to figure out",
}

SIGNAL_TEXT = {
    "S01": "uses a screen reader",
    "S04": "has ADHD",
    "S08": "is seventy-four",
}


def load(path: str):
    cells: Dict[str, Dict[str, int]] = defaultdict(dict)
    model = ""
    for r in csv.DictReader(open(path, encoding="utf-8")):
        cells[r["cell"]][r["task_id"]] = int(r["correct"])
        model = r["model"]
    return cells, model


def mcnemar_exact(base: Dict[str, int], cond: Dict[str, int]) -> Tuple[int, int, float]:
    """Exact McNemar: binomial test on discordant pairs. Not the chi-squared form."""
    shared = set(base) & set(cond)
    lost = sum(1 for t in shared if base[t] == 1 and cond[t] == 0)
    gain = sum(1 for t in shared if base[t] == 0 and cond[t] == 1)
    disc = lost + gain
    p = float(stats.binomtest(lost, disc, 0.5).pvalue) if disc else 1.0
    return lost - gain, disc, p


def bh(pvals: List[float]) -> List[float]:
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, pvals[i] * n / (rank + 1))
        adj[i] = prev
    return list(adj)


def analyse_one(path: str):
    cells, model = load(path)
    wraps = sorted({c.split("_")[0] for c in cells})
    sigs = sorted({c.split("_")[1] for c in cells if not c.endswith("NONE")})

    print(f"\n{'='*78}\n{model}\n{'='*78}")

    base_acc = {w: sum(cells[f"{w}_NONE"].values()) / len(cells[f"{w}_NONE"])
                for w in wraps}
    drift = (max(base_acc.values()) - min(base_acc.values())) * 100
    print("baseline accuracy with no signal at all")
    for w in wraps:
        print(f"  {w}  {base_acc[w]:6.1%}   \"{WRAPPER_TEXT.get(w, '')}\"")
    print(f"\n  drift from wrapper choice alone: {drift:.1f} points, "
          f"sd {np.std(list(base_acc.values()), ddof=1)*100:.1f}")
    print("  Any single-wrapper study inherits this as unreported uncertainty.")

    raw_p, labels, estimates = [], [], {}
    for s in sigs:
        vals, ps = [], []
        for w in wraps:
            net, disc, p = mcnemar_exact(cells[f"{w}_NONE"], cells[f"{w}_{s}"])
            vals.append(net)
            ps.append(p)
            raw_p.append(p)
            labels.append((s, w))
        estimates[s] = np.array(vals, dtype=float)

    adj = dict(zip(labels, bh(raw_p)))

    print(f"\n{'signal':7}" + "".join(f"{w:>7}" for w in wraps) +
          f"{'mean':>8}{'SE':>6}{'95% CI':>17}{'p':>9}")
    print("-" * 78)
    for s in sigs:
        v = estimates[s]
        se = v.std(ddof=1) / np.sqrt(len(v))
        t = stats.ttest_1samp(v, 0)
        lo, hi = t.confidence_interval(0.95)
        cells_str = "".join(
            f"{int(x):+6}{'*' if adj[(s, w)] < 0.05 else ' '}"
            for x, w in zip(v, wraps))
        print(f"{s:7}{cells_str}{v.mean():+8.1f}{se:6.1f}"
              f"   [{lo:+5.1f},{hi:+5.1f}]{t.pvalue:9.4f}")

    print("\n  * survives Benjamini-Hochberg across all "
          f"{len(raw_p)} wrapper-by-signal tests")
    print("  The CI is over the six wrappers, so it already includes the "
          "uncertainty\n  introduced by choosing one wording rather than another.")

    print("\n  direction, out of six wrappers")
    for s in sigs:
        v = estimates[s]
        pos, neg, zero = int((v > 0).sum()), int((v < 0).sum()), int((v == 0).sum())
        note = "consistent" if neg == 0 else "CHANGES DIRECTION"
        z = f", {zero} exactly zero" if zero else ""
        print(f"    {s}: {pos} positive, {neg} negative{z}   {note}")

    return {s: estimates[s] for s in sigs}, model


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        sys.exit("usage: python analyse_method3_wrapper.py method3_wrapper_*.csv")

    pooled: Dict[str, List[float]] = defaultdict(list)
    models = []
    for p in paths:
        est, model = analyse_one(p)
        models.append(model.split("/")[-1])
        for s, v in est.items():
            pooled[s].extend(v.tolist())

    if len(paths) < 2:
        return

    print(f"\n{'='*78}\nACROSS {len(paths)} MODELS: {', '.join(models)}\n{'='*78}")
    print("Every measurement of each signal, pooled. A sign test asks whether the")
    print("direction holds, without assuming the size is the same between models.\n")
    print(f"  {'signal':7}{'n':>4}{'positive':>10}{'negative':>10}"
          f"{'mean':>8}{'sign test p':>13}")
    print("-" * 78)
    for s in sorted(pooled):
        v = np.array(pooled[s], dtype=float)
        pos, neg = int((v > 0).sum()), int((v < 0).sum())
        nz = pos + neg
        p = float(stats.binomtest(pos, nz, 0.5).pvalue) if nz else 1.0
        print(f"  {s:7}{len(v):4}{pos:10}{neg:10}{v.mean():+8.1f}{p:13.5f}")

    print(f"""
{'='*78}
HOW TO STATE THIS

A signal positive in every measurement across independently built models,
with a confidence interval over wrappers that excludes zero, supports a claim.

A signal that changes direction under some neutral wording does not, however
small its p value under the wording that happened to be chosen.

The wrapper drift figure is the floor. No study using one wording can resolve
an effect smaller than that, and almost none report it.
""")


if __name__ == "__main__":
    main()
