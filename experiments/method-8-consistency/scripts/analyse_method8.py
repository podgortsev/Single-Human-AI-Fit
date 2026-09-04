#!/usr/bin/env python3
"""
analyse_method8.py — consistency / stability, from data already on disk.

No GPU, no new runs. Reads the method 1, method 3c and method 6b CSVs and asks a
question none of them asked directly: is the model LESS STABLE for a person who
has disclosed something. Not "does it judge them lower" but "does its output
scatter more when nothing that should matter is changed".

Three independent probes, one per source, each paired within item against a
signal-free baseline, each a Wilcoxon signed-rank test.

MULTIPLICITY FAMILY. Benjamini-Hochberg is applied at two levels and they answer
different questions. Within a probe, across that probe's model x signal cells,
which is what each probe table reports. Then across all 33 probe x model x
signal cells, which is what the agreement table reports and which is stricter.
Where they disagree the agreement table governs.

WHAT n IS. The n column counts paired observations, not independent subjects.
Probe A's 600 is 100 profiles crossed with six positive traits, and the six
traits on one profile are correlated, so probe A's p values are optimistic.

  A  method 1     spread of a trait log probability across the 5 prompt phrasings
  B  method 3c    disagreement of the correct-answer outcome across the 6 wrappers
  C  method 6b    spread of the choice margin across the 3 decision questions

A signal "destabilises" a model on a probe if the paired instability difference
against the baseline is significantly positive. A significantly negative
difference means the disclosed condition is MORE stable, which is also a result.

USAGE
-----
    python analyse_method8.py
    python analyse_method8.py --m1 PATH --m3c PATH --m6b PATH

Paths default to the sibling experiment output folders.
"""

import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np
from scipy import stats

MODELS = ["qwen", "llama", "mistral"]
POS_TRAITS = {"intelligent", "hardworking", "trustworthy", "educated",
              "polite", "calm"}
M1_SIGNALS = ["SCREEN", "AGE", "DEAF", "ADHD"]
M3C_SIGNALS = {"S01": "SCREEN", "S04": "ADHD", "S08": "AGE"}
M6B_SIGNALS = ["SCREEN", "AGE", "DEAF", "ADHD"]
MIN_MASS = 0.5
HERE = os.path.dirname(os.path.abspath(__file__))


def bh(pvals):
    n = len(pvals)
    if not n:
        return []
    order = np.argsort(pvals)
    adj, prev = np.empty(n), 1.0
    for r in range(n - 1, -1, -1):
        i = order[r]
        prev = min(prev, pvals[i] * n / (r + 1))
        adj[i] = prev
    return list(adj)


def rd(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wilcox(diffs):
    d = np.asarray([x for x in diffs if x == x])
    if len(d) < 10 or np.allclose(d, 0):
        return len(d), float("nan"), float("nan"), float("nan")
    med = float(np.median(d))
    try:
        p = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        p = 1.0
    return len(d), med, float(d.mean()), p


# ---------------------------------------------------------------- probe A: m1
def probe_a(root):
    out = {}
    for m in MODELS:
        path = os.path.join(root, m, f"method1_{m}.csv")
        if not os.path.exists(path):
            continue
        g = defaultdict(list)      # (signal, profile, trait) -> [logprob per phrasing]
        for r in rd(path):
            if r["trait"] in POS_TRAITS:
                g[(r["signal"], r["profile"], r["trait"])].append(
                    float(r["logprob"]))
        sd = {k: np.std(v, ddof=1) for k, v in g.items() if len(v) >= 3}
        base = {(p, t): s for (sig, p, t), s in sd.items() if sig == "NONE"}
        for sig in M1_SIGNALS:
            diffs, rb, rs = [], [], []
            for (s, p, t), v in sd.items():
                if s != sig or (p, t) not in base:
                    continue
                diffs.append(v - base[(p, t)])
                rb.append(base[(p, t)])
                rs.append(v)
            n, med, mean, pval = wilcox(diffs)
            ratio = (np.mean(rs) / np.mean(rb)) if rb else float("nan")
            out[(m, sig)] = dict(n=n, med=med, mean=mean, p=pval, ratio=ratio)
    return out


# --------------------------------------------------------------- probe B: m3c
def probe_b(root):
    out = {}
    for m in MODELS:
        path = os.path.join(root, m, f"method3_wrapper_{m}.csv")
        if not os.path.exists(path):
            continue
        g = defaultdict(list)      # (signal, task) -> [correct 0/1 per wrapper]
        for r in rd(path):
            g[(r["signal"], r["task_id"])].append(int(r["correct"]))

        def instab(v):
            k = sum(v)
            return k * (len(v) - k)          # 0 when unanimous, max near half

        inst = {k: instab(v) for k, v in g.items() if len(v) >= 4}
        base = {t: s for (sig, t), s in inst.items() if sig == "NONE"}
        for code, sig in M3C_SIGNALS.items():
            diffs = [v - base[t] for (s, t), v in inst.items()
                     if s == code and t in base]
            n, med, mean, pval = wilcox(diffs)
            out[(m, sig)] = dict(n=n, med=med, mean=mean, p=pval,
                                 ratio=float("nan"))
    return out


# --------------------------------------------------------------- probe C: m6b
def probe_c(root):
    out = {}
    for m in MODELS:
        path = os.path.join(root, m, f"method6b_{m}.csv")
        if not os.path.exists(path):
            continue
        cell = defaultdict(dict)
        for r in rd(path):
            la, lb = float(r["lp_A"]), float(r["lp_B"])
            cell[(r["signal"], r["profile"], r["question"])][r["order"]] = (
                la, lb, math.exp(la) + math.exp(lb))
        margin = {}
        for (sig, p, q), d in cell.items():
            if "sig_A" not in d or "sig_B" not in d:
                continue
            a1, b1, t1 = d["sig_A"]
            a2, b2, t2 = d["sig_B"]
            if t1 < MIN_MASS or t2 < MIN_MASS:
                continue
            margin[(sig, p, q)] = ((a1 - b1) + (b2 - a2)) / 2
        byp = defaultdict(dict)
        for (sig, p, q), v in margin.items():
            byp[(sig, p)][q] = v
        sd = {k: np.std(list(v.values()), ddof=1)
              for k, v in byp.items() if len(v) == 3}
        base = {p: s for (sig, p), s in sd.items() if sig == "CONTROL_PARA"}
        for sig in M6B_SIGNALS:
            diffs = [v - base[p] for (s, p), v in sd.items()
                     if s == sig and p in base]
            n, med, mean, pval = wilcox(diffs)
            out[(m, sig)] = dict(n=n, med=med, mean=mean, p=pval,
                                 ratio=float("nan"))
    return out


def show(title, res, unit):
    print(f"\n{'='*90}\n{title}\n{'='*90}")
    print(f"{'model':9}{'signal':8}{'n':>6}{'median d':>11}{'mean d':>10}"
          f"{'ratio':>8}{'p':>10}{'BH':>10}   verdict")
    print(f"{'':52}({unit})")
    print("-" * 92)
    keys = [k for k in res if res[k]["p"] == res[k]["p"]]
    adj = dict(zip(keys, bh([res[k]["p"] for k in keys])))
    for m in MODELS:
        for k in [x for x in res if x[0] == m]:
            r = res[k]
            a = adj.get(k, float("nan"))
            if a == a and a < 0.05:
                v = "LESS stable" if r["med"] > 0 else "MORE stable"
            else:
                v = "no effect"
            rr = f"{r['ratio']:8.2f}" if r["ratio"] == r["ratio"] else f"{'-':>8}"
            ap = f"{a:10.4f}" if a == a else f"{'-':>10}"
            pp = f"{r['p']:10.4f}" if r["p"] == r["p"] else f"{'-':>10}"
            print(f"{m:9}{k[1]:8}{r['n']:6}{r['med']:+11.3f}{r['mean']:+10.3f}"
                  f"{rr}{pp}{ap}   {v}")
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1", default=os.path.join(HERE, "..", "..",
                    "method-1-traits-logprob", "outputs"))
    ap.add_argument("--m3c", default=os.path.join(HERE, "..", "..",
                    "method-3c-wrappers", "outputs"))
    ap.add_argument("--m6b", default=os.path.join(HERE, "..", "..",
                    "method-6b-logprob-choice", "outputs"))
    a = ap.parse_args()

    print("METHOD 8, CONSISTENCY / STABILITY")
    print("Is the model LESS stable for a person who disclosed something?")
    print("Positive median = the disclosed condition scatters more than the "
          "signal-free baseline.")

    ra = probe_a(a.m1)
    rb = probe_b(a.m3c)
    rc = probe_c(a.m6b)
    show("PROBE A  method 1: trait log prob across 5 phrasings   "
         "(baseline NONE)", ra, "log prob SD")
    show("PROBE B  method 3c: correct-answer outcome across 6 wrappers   "
         "(baseline NONE)", rb, "k*(6-k) instability")
    show("PROBE C  method 6b: choice margin across 3 decision questions   "
         "(baseline CONTROL_PARA)", rc, "margin SD, logits")

    print(f"\n{'='*90}\nAGREEMENT\n{'='*90}")
    print(f"{'signal':8}{'LESS stable':>13}{'MORE stable':>13}"
          f"{'no effect':>11}{'of cells':>10}")
    print("-" * 60)
    allres = {("A",) + k: v for k, v in ra.items()}
    allres.update({("B",) + k: v for k, v in rb.items()})
    allres.update({("C",) + k: v for k, v in rc.items()})
    keys = [k for k in allres if allres[k]["p"] == allres[k]["p"]]
    adj = dict(zip(keys, bh([allres[k]["p"] for k in keys])))
    for sig in ["SCREEN", "AGE", "DEAF", "ADHD"]:
        less = more = none = 0
        for k, r in allres.items():
            if k[2] != sig:
                continue
            a_ = adj.get(k, float("nan"))
            if a_ == a_ and a_ < 0.05:
                if r["med"] > 0:
                    less += 1
                else:
                    more += 1
            else:
                none += 1
        print(f"{sig:8}{less:>13}{more:>13}{none:>11}{less+more+none:>10}")
    print("\nOne cell is one probe x one model. DEAF has no method 3c probe, so "
          "it has fewer cells.")


if __name__ == "__main__":
    main()
