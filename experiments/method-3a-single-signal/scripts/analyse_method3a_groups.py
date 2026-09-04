#!/usr/bin/env python3
"""
analyse_method3a_groups.py — form-of-writing conditions against stated-about-self
conditions.

No GPU. Runs on the CSVs already saved.

WHY THIS EXISTS SEPARATELY
--------------------------
run_method3_single.py tests each condition against the baseline, one exact
McNemar per condition, with a Bonferroni threshold for 24 comparisons. That
answers "does this one condition cost accuracy".

It does not answer the question the method was built for: does STATING something
about yourself cost more than WRITING differently. That is a comparison between
two groups of conditions, and it needs its own named test. Earlier versions of
the result document quoted p values for this contrast that no committed script
produced; this file replaces them with numbers anyone can regenerate.

THE TEST
--------
Ten form conditions (F01-F10) against ten stated conditions (S01-S10). The unit
is the CONDITION, and the value is its net loss against the baseline, so n is
10 against 10, not 200 against 200. Mann-Whitney U, one-sided, asking whether
stated conditions lose more than form conditions.

The two groups are separate sets of conditions facing the same 200 tasks, so
they are treated as independent samples rather than paired. Two designed pairs
do exist (F01 with S09, non-native shown against declared; F04 with S07,
imprecise typing shown against declared) and are printed separately, but with
one pair each they support no test.

    python analyse_method3a_groups.py
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["qwen", "llama", "mistral"]
DESIGNED_PAIRS = [("F01", "S09", "non-native: shown by writing / declared"),
                  ("F04", "S07", "imprecise typing: shown by typos / declared")]


def cond_table(path):
    """condition -> {task: correct}."""
    by = defaultdict(dict)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        by[r["condition"]][r["task_id"]] = int(r["correct"])
    return by


def mcnemar(a, b):
    """Exact McNemar of condition a against condition b, on shared tasks.

    Returns (net, discordant, p). Positive net means a lost more than b.
    """
    shared = set(a) & set(b)
    a_lost = sum(1 for t in shared if b[t] == 1 and a[t] == 0)
    b_lost = sum(1 for t in shared if a[t] == 1 and b[t] == 0)
    disc = a_lost + b_lost
    p = float(stats.binomtest(a_lost, disc, 0.5).pvalue) if disc else 1.0
    return a_lost - b_lost, disc, p


def nets(path):
    by = defaultdict(dict)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        by[r["condition"]][r["task_id"]] = int(r["correct"])
    base = by["BASE"]

    def net(c):
        shared = set(base) & set(by[c])
        lost = sum(1 for t in shared if base[t] == 1 and by[c][t] == 0)
        gained = sum(1 for t in shared if base[t] == 0 and by[c][t] == 1)
        return lost - gained

    return {c: net(c) for c in by if c != "BASE"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=os.path.join(HERE, "..", "outputs"))
    a = ap.parse_args()

    print("=" * 76)
    print("METHOD 3a: writing differently against stating something about yourself")
    print("=" * 76)
    print("Unit is the condition; the value is its net loss against the baseline.")
    print("Mann-Whitney U, one-sided, ten form conditions against ten stated.")
    print("")
    print(f"{'model':9}{'form mean':>11}{'stated mean':>13}{'U p':>10}"
          f"{'control net':>13}")
    print("-" * 76)

    all_f, all_s = [], []
    per_model = {}
    for m in MODELS:
        path = os.path.join(a.outputs, m, f"method3_single_{m}.csv")
        if not os.path.exists(path):
            continue
        n = nets(path)
        f = [v for c, v in sorted(n.items()) if c.startswith("F")]
        s = [v for c, v in sorted(n.items()) if c.startswith("S")]
        ctrl = n.get("CTRL", float("nan"))
        p = float(stats.mannwhitneyu(s, f, alternative="greater").pvalue)
        per_model[m] = n
        print(f"{m:9}{np.mean(f):+11.2f}{np.mean(s):+13.2f}{p:10.4f}"
              f"{ctrl:+13.0f}")
        all_f += f
        all_s += s

    if all_f:
        p = float(stats.mannwhitneyu(all_s, all_f, alternative="greater").pvalue)
        print(f"{'pooled':9}{np.mean(all_f):+11.2f}{np.mean(all_s):+13.2f}"
              f"{p:10.4f}")
        print("")
        print("  The pooled row is DESCRIPTIVE only. It stacks the same ten")
        print("  conditions from three models as though they were thirty")
        print("  independent observations, which they are not.")

    print("")
    print("=" * 76)
    print("THE TWO DESIGNED PAIRS: matched hypotheses, shown or declared")
    print("=" * 76)
    print("One pair each, so these are reported as numbers, not as a test.")
    print("")
    print(f"{'model':9}{'pair':46}{'shown':>8}{'declared':>10}")
    print("-" * 76)
    for m, n in per_model.items():
        for shown, declared, label in DESIGNED_PAIRS:
            if shown in n and declared in n:
                print(f"{m:9}{label:46}{n[shown]:+8}{n[declared]:+10}")


def signal_vs_control(outputs):
    """Each signal condition tested DIRECTLY against the control condition.

    The built-in analysis in run_method3_single.py reports conditions that
    (a) beat the baseline at p<0.05 and (b) have a numerically larger net than
    the control. That is a filter, not a test: it never compares a signal with
    the control statistically. Both faced the same 200 tasks, so the comparison
    is available, and this is it.
    """
    print("")
    print("=" * 76)
    print("EACH SIGNAL AGAINST THE CONTROL, DIRECTLY")
    print("=" * 76)
    print("Exact McNemar of the condition against CTRL on the same 200 tasks.")
    print("Positive net means the condition lost more than the control did.")
    print("Bonferroni threshold for 24 comparisons is 0.0021.")
    print("")
    for m in MODELS:
        path = os.path.join(outputs, m, f"method3_single_{m}.csv")
        if not os.path.exists(path):
            continue
        by = cond_table(path)
        if "CTRL" not in by:
            continue
        rows = []
        for c in sorted(by):
            if c in ("BASE", "CTRL"):
                continue
            net, disc, p = mcnemar(by[c], by["CTRL"])
            rows.append((c, net, disc, p))
        surv = [r for r in rows if r[3] < 0.05 / len(rows)]
        print(f"--- {m}: {len(surv)} of {len(rows)} beat the control after "
              f"correction")
        for c, net, disc, p in sorted(surv, key=lambda r: r[3]):
            print(f"      {c}  net {net:+4}  discordant {disc:3}  p={p:.4f}")
        if not surv:
            nominal = [r for r in rows if r[3] < 0.05]
            print(f"      none. {len(nominal)} reach p<0.05 uncorrected: "
                  + ", ".join(r[0] for r in nominal))
        print("")


if __name__ == "__main__":
    main()
    signal_vs_control(os.path.join(HERE, "..", "outputs"))
