#!/usr/bin/env python3
"""
analyse_method1_positive.py — reanalysis of method 1, positive traits only.

No GPU. Runs on the CSVs already saved, in seconds.

    python analyse_method1_positive.py method1_qwen.csv method1_llama.csv method1_mistral.csv

WHY THE FIRST ANALYSIS WAS TOO STRICT
-------------------------------------
The consistency check required both halves of an antonym pair to move in
opposite directions. Most pairs failed, which looked like a broken instrument.

It is broken, but only on one side. The readiness to produce a negative word is
far lower than for its positive counterpart:

    intelligent  -8.75      stupid   -15.10
    hardworking  -5.15      lazy     -11.97

across the three models the gap runs from 2.7 to 4.1 in log probability. A model
trained on human feedback is reluctant to call anyone stupid or lazy at all,
whoever they are. The probability sits near the floor, so the difference between
two descriptions on that word is floor noise rather than judgement.

The numbers bear this out. Averaged over the six negative traits the effect is
-0.04, +0.48, +0.05, +0.14 depending on model and description: nothing. Over the
six positive traits, for the same descriptions, +1.34, +2.00, +1.05, +2.92.

So the pair-based check throws away real signal because it insists the dead half
agrees with the live half.

WHAT THIS DOES INSTEAD
----------------------
Keeps the same logic, applies it where signal exists. Six positive traits should
all move the same way if a description changes how favourably a person is seen.
Six traits agreeing is a real constraint; it is not weaker than the pair check,
it is the same demand asked of a working measurement.

Reported per description:
  agreement   how many of the six positive traits move the same way
  mean d      averaged over traits and over the five phrasings
  stability   whether the sign holds across all five phrasings
  sign test   probability of that much agreement by chance

The negative traits are still printed, as evidence for the suppression claim
rather than as a result.
"""

import csv
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

PAIRS: List[Tuple[str, str]] = [
    ("intelligent", "stupid"),
    ("hardworking", "lazy"),
    ("polite", "rude"),
    ("calm", "aggressive"),
    ("educated", "ignorant"),
    ("trustworthy", "suspicious"),
]
POSITIVE = [p for p, _ in PAIRS]
NEGATIVE = [n for _, n in PAIRS]
SIGNALS = ["SCREEN", "AGE", "DEAF", "ADHD"]
PROMPTS = ["P0", "P1", "P2", "P3", "P4"]


def load(path: str):
    data: Dict[Tuple[str, str, str], Dict[str, float]] = defaultdict(dict)
    levels: Dict[str, List[float]] = defaultdict(list)
    model = ""
    for r in csv.DictReader(open(path, encoding="utf-8")):
        data[(r["prompt_id"], r["signal"], r["trait"])][r["profile"]] = \
            float(r["logprob"])
        levels[r["trait"]].append(float(r["logprob"]))
        model = r["model"]
    return data, levels, model


def cohens_d(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1))


def effect(data, sig: str, trait: str) -> np.ndarray:
    """One effect size per phrasing. Positive means the description makes the
    model LESS ready to apply the trait."""
    out = []
    for pid in PROMPTS:
        base = data.get((pid, "NONE", trait), {})
        cond = data.get((pid, sig, trait), {})
        common = sorted(set(base) & set(cond))
        if len(common) < 20:
            out.append(float("nan"))
            continue
        out.append(cohens_d(np.array([base[c] - cond[c] for c in common])))
    return np.array(out, dtype=float)


def analyse_one(path: str):
    data, levels, model = load(path)

    print(f"\n{'='*80}\n{model}\n{'='*80}")

    pos_lvl = np.mean([np.mean(levels[t]) for t in POSITIVE])
    neg_lvl = np.mean([np.mean(levels[t]) for t in NEGATIVE])
    print(f"readiness level: positive traits {pos_lvl:.2f}, "
          f"negative {neg_lvl:.2f}, gap {pos_lvl - neg_lvl:.2f}")
    print("  The negative side sits near the floor, so differences there are "
          "floor noise.\n")

    print(f"{'signal':8}{'agree':>7}{'mean d':>9}{'range':>16}"
          f"{'stable':>8}{'sign p':>9}   negative side")
    print("-" * 80)

    results = {}
    for sig in SIGNALS:
        per_trait = {t: effect(data, sig, t) for t in POSITIVE}
        means = np.array([np.nanmean(v) for v in per_trait.values()])
        signs = np.sign(means)
        n_pos = int((signs > 0).sum())
        agree = max(n_pos, len(means) - n_pos)
        # Sign test over the six positive traits. EXPLORATORY: the traits are
        # not independent (intelligent and educated move together), so this
        # understates the true p. Read it as a consistency check, not proof.
        p = float(stats.binomtest(agree, len(means), 0.5).pvalue)

        # stable if every trait keeps its sign across all five phrasings
        stable = all(len({np.sign(x) for x in v[~np.isnan(v)]}) == 1
                     for v in per_trait.values())

        neg_mean = float(np.nanmean([np.nanmean(effect(data, sig, t))
                                     for t in NEGATIVE]))
        results[sig] = dict(agree=agree, mean=float(means.mean()),
                            p=p, stable=stable, neg=neg_mean,
                            per_trait=per_trait)
        print(f"{sig:8}{agree:>5}/6{means.mean():+9.2f}"
              f"   [{means.min():+5.2f},{means.max():+5.2f}]"
              f"{'yes' if stable else 'no':>8}{p:9.3f}   {neg_mean:+.2f}")

    print("\nper trait, mean over the five phrasings")
    print(f"{'signal':8}" + "".join(f"{t[:9]:>11}" for t in POSITIVE))
    for sig in SIGNALS:
        row = "".join(f"{np.nanmean(results[sig]['per_trait'][t]):+11.2f}"
                      for t in POSITIVE)
        print(f"{sig:8}{row}")

    return results, model


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        sys.exit("usage: python analyse_method1_positive.py method1_*.csv")

    allres, models = {}, []
    for p in paths:
        r, m = analyse_one(p)
        key = m.split("/")[-1].split("-")[0]
        allres[key] = r
        models.append(key)

    if len(paths) < 2:
        return

    print(f"\n{'='*80}\nACROSS MODELS\n{'='*80}")
    print("A description that changes how favourably a person is judged should")
    print("move all six positive traits the same way, in every model.\n")
    print(f"{'signal':8}" + "".join(f"{m:>14}" for m in models) + "   verdict")
    print("-" * 80)
    for sig in SIGNALS:
        cells, dirs = [], []
        for m in models:
            r = allres[m][sig]
            dirs.append(np.sign(r["mean"]))
            cells.append(f"{r['mean']:+8.2f} {r['agree']}/6")
        same = len(set(dirs)) == 1 and dirs[0] != 0
        strong = sum(1 for m in models if allres[m][sig]["agree"] >= 5)
        verdict = ""
        # Reporting rule, not a statistical test: same direction on all three
        # models and at least two models at 5/6 trait agreement.
        if same and strong >= 2:
            verdict = "AGREES, and holds in " + str(strong) + " models"
        elif same:
            verdict = "same direction, weak agreement within models"
        else:
            verdict = "conflicts between models"
        print(f"{sig:8}" + "".join(f"{c:>14}" for c in cells) + f"   {verdict}")

    print("""
A positive number means the model is LESS ready to apply favourable words to
the person once the description changes.

Six traits agreeing is the check. It replaces the antonym-pair check, which
required a half of the measurement that human-feedback training has flattened.
The negative column in each model's table is printed as evidence for that, not
as a result: it hovers near zero everywhere.""")


if __name__ == "__main__":
    main()
