#!/usr/bin/env python3
"""
analyse_method2.py — cross-judge read of method 2.

No GPU. Runs on the judged CSVs already saved.

    python analyse_method2.py method2_judged_by_qwen.csv method2_judged_by_llama.csv ...

WHY A SEPARATE ANALYSIS
-----------------------
run_method2_judge.py analyses one judge at a time. Three judges were run, which
is better than the design asked for: a result that holds whoever judges is worth
much more than one judge's opinion. This script reads them together.

It also fixes two things the built-in read gets wrong.

1. THE ADDITIVITY WARNING WAS INFLATED. The built-in check compares the slot
   bias across all comparisons including IDENTITY. IDENTITY shows the SAME
   answer twice, so the judge has nothing but position to go on and its slot
   term is necessarily larger. Excluding that degenerate case, the spread across
   the four real comparisons is 0.5 to 1.4 logits rather than 2.4 to 5.2. Half
   that spread is the residual left in the order-averaged margin, and it is
   reported here as an uncertainty floor an effect has to clear.

2. A JUDGE THAT WILL NOT ANSWER WITH A LETTER IS NOT A JUDGE. Letter mass
   P(A)+P(B) is checked per judge, and a judge below the threshold is excluded
   from every conclusion rather than quietly averaged in.

WHAT IS REPORTED
----------------
  per judge    letter mass, slot bias, additivity residual
  per cell     judge x answer model x signal: the order-averaged margin against
               the length-matched CONTROL lead-in, Wilcoxon, BH within judge
  agreement    for each signal, in how many usable cells it is negative and
               significant, and whether it clears the additivity residual
  self-judged  cells where the judge scored its own answers are marked
"""

import csv
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

ANSWER_MODELS = ["qwen", "llama", "mistral"]
SIGNALS = ["SCREEN", "AGE", "ADHD"]
FLOOR = "NONE_vs_CONTROL"
DEGENERATE = "IDENTITY"
MIN_MASS = 0.5
MIN_JUDGE_MASS = 0.8


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bh(pvals):
    n = len(pvals)
    if not n:
        return []
    order = np.argsort(pvals)
    adj, prev = np.empty(n), 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, pvals[i] * n / (rank + 1))
        adj[i] = prev
    return list(adj)


def index(rows):
    """(answer_model, question, comparison) -> {order: (lpA, lpB, mass, dwords)}"""
    cell = defaultdict(dict)
    for r in rows:
        cell[(r["answer_model"], r["question_id"], r["comparison"])][r["order"]] = (
            float(r["lp_A"]), float(r["lp_B"]), float(r["letter_mass"]),
            int(r["words_changed"]) - int(r["words_reference"]))
    return cell


def series(cell, model, comparison, keyed=False):
    """Order-averaged margin, slot term and length gap, one entry per question.

    keyed=True returns {question: margin} so a signal can be paired with the
    signal-free floor on the same question.
    """
    comb, pos, dw, by_q = [], [], [], {}
    for (am, q, comp), d in cell.items():
        if am != model or comp != comparison:
            continue
        if "chg_A" not in d or "chg_B" not in d:
            continue
        a1, b1, t1, g = d["chg_A"]
        a2, b2, t2, _ = d["chg_B"]
        if t1 < MIN_MASS or t2 < MIN_MASS:
            continue
        m1, m2 = a1 - b1, b2 - a2
        comb.append((m1 + m2) / 2)
        pos.append((m1 - m2) / 2)
        dw.append(g)
        by_q[q] = (m1 + m2) / 2
    if keyed:
        return by_q
    return np.array(comb), np.array(pos), np.array(dw)


def paired_vs_floor(cell, model, signal):
    """Per question: signal margin minus the signal-free floor margin.

    Both are measured against the same CONTROL answer on the same question, so
    this contrast removes the reference and asks the only question that matters:
    is a disclosure treated differently from having no lead-in at all. The test
    and the verdict then use the same quantity, which testing against zero did
    not.
    """
    sig = series(cell, model, signal, keyed=True)
    flo = series(cell, model, FLOOR, keyed=True)
    shared = sorted(set(sig) & set(flo))
    return np.array([sig[q] - flo[q] for q in shared])


def judge_stats(cell):
    """Slot bias on the degenerate case, and the residual from the real ones."""
    ident = [series(cell, m, DEGENERATE)[1].mean()
             for m in ANSWER_MODELS if len(series(cell, m, DEGENERATE)[1])]
    spreads = []
    for m in ANSWER_MODELS:
        terms = [series(cell, m, c)[1].mean()
                 for c in [FLOOR] + SIGNALS if len(series(cell, m, c)[1])]
        if len(terms) >= 2:
            spreads.append(max(terms) - min(terms))
    slot = float(np.mean(ident)) if ident else float("nan")
    resid = max(spreads) / 2 if spreads else float("nan")
    return slot, resid


def main():
    paths = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not paths:
        sys.exit("usage: python analyse_method2.py method2_judged_by_*.csv")

    judges, usable = {}, []
    print("=" * 100)
    print("METHOD 2, ANSWER QUALITY. SUBJECTIVE: REPORT SEPARATELY FROM 1, 3, 4, 6.")
    print("=" * 100)
    print("\nJUDGES\n" + "-" * 100)
    for p in paths:
        rows = load(p)
        key, name = rows[0]["judge_key"], rows[0]["judge"]
        cell = index(rows)
        mass = np.array([float(r["letter_mass"]) for r in rows])
        slot, resid = judge_stats(cell)
        ok = mass.mean() >= MIN_JUDGE_MASS
        judges[key] = dict(name=name, cell=cell, resid=resid, ok=ok)
        if ok:
            usable.append(key)
        note = "" if ok else "   EXCLUDED: not answering with a letter"
        print(f"{key:9} letter mass {mass.mean():.2f}   below {MIN_MASS}: "
              f"{np.mean(mass < MIN_MASS):>4.0%}   slot bias on identical "
              f"answers {slot:5.2f}   additivity residual +/-{resid:.2f}{note}")

    if not usable:
        sys.exit("\nno usable judge, nothing can be concluded.")
    print(f"\nusable judges: {', '.join(usable)}")

    print("\n" + "=" * 100)
    print("EFFECT OF DISCLOSING, paired against the signal-free floor per question")
    print("=" * 100)
    print("negative = the answer to the person who disclosed was judged worse\n")

    verdicts = defaultdict(list)
    for jk in usable:
        J = judges[jk]
        print(f"--- judge {jk}   (additivity residual +/-{J['resid']:.2f} logits)")
        print(f"  {'answers by':13}{'signal':8}{'n':>4}{'vs floor':>9}{'95% CI':>18}"
              f"{'Wilcoxon':>10}{'BH':>9}{'len gap':>9}   verdict")
        print("  " + "-" * 96)
        raw_p, keys = [], []
        for am in ANSWER_MODELS:
            for s in SIGNALS:
                d = paired_vs_floor(J["cell"], am, s)
                if len(d) >= 20:
                    raw_p.append(float(stats.wilcoxon(d).pvalue))
                    keys.append((am, s))
        adj = dict(zip(keys, bh(raw_p)))
        for am in ANSWER_MODELS:
            for s in SIGNALS:
                d = paired_vs_floor(J["cell"], am, s)
                _, _, dw = series(J["cell"], am, s)
                label = am + (" (self)" if jk == am else "")
                if len(d) < 20:
                    print(f"  {label:13}{s:8}{len(d):4}   too few clean reads")
                    continue
                m = d.mean()
                se = d.std(ddof=1) / np.sqrt(len(d))
                lo, hi = stats.t.interval(0.95, len(d) - 1, loc=m, scale=se or 1e-12)
                p = float(stats.wilcoxon(d).pvalue)
                a = adj[(am, s)]
                if a < 0.05 and m < 0 and abs(m) > J["resid"]:
                    v = "worse"
                elif a < 0.05 and m < 0:
                    v = "worse, inside residual"
                elif a < 0.05 and m > 0:
                    v = "better"
                else:
                    v = "no effect"
                verdicts[s].append((jk, am, v, m))
                print(f"  {label:13}{s:8}{len(d):4}{m:+9.2f}"
                      f"   [{lo:+6.2f},{hi:+6.2f}]{p:>10.4f}{a:>9.4f}"
                      f"{dw.mean() if len(dw) else 0:+9.0f}   {v}")
        print()

    print("=" * 100)
    print("AGREEMENT ACROSS JUDGES AND ANSWER MODELS")
    print("=" * 100)
    print(f"{'signal':8}{'worse':>8}{'inside resid':>14}{'no effect':>11}"
          f"{'better':>8}{'of':>5}   mean gap vs floor")
    print("-" * 100)
    for s in SIGNALS:
        v = verdicts[s]
        w = sum(1 for x in v if x[2] == "worse")
        ins = sum(1 for x in v if x[2] == "worse, inside residual")
        ne = sum(1 for x in v if x[2] == "no effect")
        bt = sum(1 for x in v if x[2] == "better")
        gap = np.mean([x[3] for x in v]) if v else float("nan")
        print(f"{s:8}{w:>8}{ins:>14}{ne:>11}{bt:>8}{len(v):>5}   {gap:+.2f}")

    print("""
A cell is one judge crossed with one answer model. "worse" means significant
after correction, in the negative direction, and further from the signal-free
floor than the additivity residual. "worse, inside residual" means the direction
is right and the p value small, but the size is within what the imperfect order
correction could produce on its own: suggestive and no more.

len gap is the mean word-count difference between the two answers. Small values
mean the judge was not simply rewarding length.

Cells marked (self) are a model judging its own answers. That bias is constant
within an answer model, so it cannot create a difference between conditions of
that same model, but the cells are marked so the reader can check.""")


if __name__ == "__main__":
    main()
