#!/usr/bin/env python3
"""
analyse_method4_control.py — does the rating result survive a change of
reference condition? No GPU.

THE QUESTION
------------
Method 4 compared every disclosure against one clause, "commutes from a nearby
town". Method 6b showed that clause is not neutral to these models. So the
rating result has never been read against a second neutral detail, and "the
rating goes up when a person discloses" could be, in part, "the rating goes
down when a person commutes".

This script reads the original run and the control run together and reports
each disclosure twice: against the commute clause and against the cycling
clause. Three things decide the answer.

1. THE REFERENCE ITSELF. Cycling against commuting, paired within profile. If
   the two neutral clauses already differ, the size of that gap is the amount
   of the original result that was never about disclosure.

2. DIRECTION UNDER BOTH REFERENCES. A disclosure that raises the number
   against both references is a finding about the disclosure. One that raises
   it against commuting only is a finding about the commute clause.

3. HOW MUCH SURVIVES. The median shift against cycling, next to the median
   shift against commuting, measure by measure.

USAGE
-----
    python analyse_method4_control.py                 # all three models
    python analyse_method4_control.py --model qwen
    python analyse_method4_control.py --dir /path/to/csvs
"""

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

MODELS = ["qwen", "llama", "mistral"]
DISCLOSURES = ["SCREEN", "AGE", "DEAF", "ADHD"]
BASE = "NONE"            # commutes from a nearby town
ALT = "CTRL_CYCLE"       # cycles to work most days
USABLE = ("ok", "hedged")
MIN_PAIRS = 10

# Higher is better for the person, so a positive shift is favourable. Taken
# from run_method4.py's Measure.good_when_high; None where it carries no
# valence. EXPERIENCE is excluded everywhere: its answer is stated in the
# profile, so it measures reading, not judgement.
VALENCE = {
    "SALARY": True, "CREDIT": True, "RAISE": True, "SUCCESS": True,
    "SCORE": True, "RATING": True,
    "CHURN": False, "DAYS": False, "SUPPORT": False,
}
SKIP_MEASURES = {"EXPERIENCE"}


def bh(pvals: List[float]) -> List[float]:
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(pvals)
    adj, prev = np.empty(n), 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, pvals[i] * n / (rank + 1))
        adj[i] = prev
    return list(adj)


def read(path: str) -> Dict[Tuple[str, str], Dict[str, float]]:
    """(measure, signal) -> {case: value}, usable rows only."""
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["status"] not in USABLE or not r["value"]:
                continue
            if r["measure"] in SKIP_MEASURES:
                continue
            out.setdefault((r["measure"], r["signal"]), {})[r["case"]] = float(
                r["value"])
    return out


def paired(a: Dict[str, float], b: Dict[str, float]):
    """b minus a, over the cases both conditions answered."""
    common = sorted(set(a) & set(b))
    d = np.array([b[c] - a[c] for c in common], dtype=float)
    if len(d) < MIN_PAIRS:
        return None
    # All differences zero is an answer, not a missing cell: the two conditions
    # are indistinguishable. Wilcoxon cannot be run on it, so p is 1 by
    # definition. Dropping the row here would hide exactly the case this
    # script exists to catch.
    p = 1.0 if not np.any(d != 0) else float(stats.wilcoxon(d).pvalue)
    return dict(n=len(d), med=float(np.median(d)),
                up=int((d > 0).sum()), down=int((d < 0).sum()),
                base_med=float(np.median([a[c] for c in common])), p=p)


def arrow(med: float, up: int, down: int) -> str:
    if up > down:
        return "up"
    if down > up:
        return "down"
    return "flat"


def run_model(model: str, d: str) -> bool:
    main_path = os.path.join(d, f"method4_{model}.csv")
    ctrl_path = os.path.join(d, f"method4_control_{model}.csv")
    if not (os.path.exists(main_path) and os.path.exists(ctrl_path)):
        print(f"\n{model}: missing {os.path.basename(main_path)} or "
              f"{os.path.basename(ctrl_path)}, skipped")
        return False

    vals = read(main_path)
    vals.update(read(ctrl_path))
    measures = sorted({m for (m, s) in vals if s == BASE})

    print("\n" + "=" * 92)
    print(f"{model}: THE RATING CHANNEL AGAINST TWO NEUTRAL REFERENCES")
    print("=" * 92)

    # ---------------------------------------------------------------- 1
    print("\n1. THE TWO REFERENCES AGAINST EACH OTHER")
    print("   Cycling minus commuting, paired within profile. Anything here is")
    print("   movement with no disclosure in it at all.\n")
    print(f"   {'measure':11}{'n':>5}{'median diff':>12}{'diff/base':>10}"
          f"{'up':>5}{'down':>6}{'p':>9}")
    print("   " + "-" * 60)
    ref_rows, ref_p = [], []
    for m in measures:
        r = paired(vals.get((m, BASE), {}), vals.get((m, ALT), {}))
        if not r:
            continue
        ref_rows.append((m, r))
        ref_p.append(r["p"])
    ref_adj = dict(zip([m for m, _ in ref_rows], bh(ref_p)))
    for m, r in ref_rows:
        star = "*" if ref_adj[m] < 0.05 else " "
        pct = r["med"] / max(abs(r["base_med"]), 1e-9) * 100
        print(f"   {m:11}{r['n']:5}{r['med']:+12,.1f}{pct:+9.1f}%"
              f"{r['up']:5}{r['down']:6}{r['p']:9.4f}{star}")
    moved = sum(1 for m, _ in ref_rows if ref_adj[m] < 0.05)
    print(f"\n   {moved} of {len(ref_rows)} measures move between two clauses "
          f"that say nothing about the person.")
    if moved:
        print("   That movement was inside every number the original run "
              "reported.")

    # ---------------------------------------------------------------- 2
    print("\n2. EACH DISCLOSURE AGAINST BOTH REFERENCES")
    print("   Same profiles, same questions. Only the clause compared against")
    print("   changes.\n")
    print(f"   {'measure':11}{'signal':8}"
          f"{'vs commute':>22}{'vs cycling':>22}   verdict")
    print("   " + "-" * 86)

    rows, pv = [], []
    for m in measures:
        for s in DISCLOSURES:
            a = paired(vals.get((m, BASE), {}), vals.get((m, s), {}))
            b = paired(vals.get((m, ALT), {}), vals.get((m, s), {}))
            if not a or not b:
                continue
            rows.append((m, s, a, b))
            pv.extend([a["p"], b["p"]])
    adj = dict(zip([(m, s, k) for (m, s, _, _) in rows for k in ("a", "b")],
                   bh(pv)))

    survived = agreed = 0
    for m, s, a, b in rows:
        sa = "*" if adj[(m, s, "a")] < 0.05 else " "
        sb = "*" if adj[(m, s, "b")] < 0.05 else " "
        da, db = arrow(a["med"], a["up"], a["down"]), arrow(
            b["med"], b["up"], b["down"])
        same = da == db and da != "flat"
        agreed += same
        if same and adj[(m, s, "b")] < 0.05:
            survived += 1
        if da == "flat" and db == "flat":
            verdict = "no movement either way"
        elif same:
            verdict = ("holds" if adj[(m, s, "b")] < 0.05
                       else "same way, not significant")
        else:
            verdict = "REFERENCE DECIDES IT"
        print(f"   {m:11}{s:8}"
              f"{a['med']:+11,.1f} {da:<5}{sa}"
              f"{b['med']:+11,.1f} {db:<5}{sb}   {verdict}")

    print(f"\n   {agreed} of {len(rows)} cells point the same way under both "
          f"references.")
    print(f"   {survived} of {len(rows)} also survive correction against "
          f"cycling.")
    print("   Cells marked REFERENCE DECIDES IT are the ones where the answer "
          "came from")
    print("   the clause being compared against, not from the disclosure.")

    # ---------------------------------------------------------------- 3
    print("\n3. BY DISCLOSURE, POOLED OVER MEASURES")
    print(f"   {'signal':8}{'cells':>6}{'up vs commute':>16}"
          f"{'up vs cycling':>16}{'agree':>8}")
    print("   " + "-" * 56)
    for s in DISCLOSURES:
        rs = [(m, a, b) for (m, sig, a, b) in rows if sig == s]
        if not rs:
            continue
        ua = sum(1 for _, a, _ in rs if a["up"] > a["down"])
        ub = sum(1 for _, _, b in rs if b["up"] > b["down"])
        ag = sum(1 for _, a, b in rs
                 if arrow(a["med"], a["up"], a["down"]) ==
                 arrow(b["med"], b["up"], b["down"]) != "flat")
        print(f"   {s:8}{len(rs):6}{ua:16}{ub:16}{ag:8}")

    print("\n   A disclosure whose column shrinks from left to right was "
          "partly the")
    print("   commute clause. A disclosure that keeps its direction under both")
    print("   references is a property of the disclosure itself.")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--dir", default=os.path.join(here, "..", "outputs"),
                    help="folder holding the CSVs, or a parent of <model>/ "
                         "folders")
    ap.add_argument("--model", default=None, choices=MODELS)
    a = ap.parse_args()

    models = [a.model] if a.model else MODELS
    done = 0
    for m in models:
        d = os.path.join(a.dir, m)
        if not os.path.isdir(d):
            d = a.dir
        done += run_model(m, d)
    if not done:
        sys.exit("no model had both CSVs present")


if __name__ == "__main__":
    main()
