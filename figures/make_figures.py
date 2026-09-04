#!/usr/bin/env python3
"""
make_figures.py — the three headline figures.

No GPU. `python make_figures.py` writes the PNGs into this folder.

The numbers are the verified results from each method's analysis. Provenance is
in the comment above each block: the experiment folder and the analysis script
that produced it. Re-run those scripts on the CSVs in ../experiments to check.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 10, "figure.dpi": 130, "savefig.dpi": 130})
WORSE = "#c2452d"
BETTER = "#2d6ec2"
NEUTRAL = "#8a8a8a"


# ============================================================================
# FIGURE 1  the central disagreement
# ----------------------------------------------------------------------------
# DIRECTION ONLY. Each method reports in its own unit (log probability, judge
# logits, net tasks, a numeric score), and those units are not convertible into
# one another. An earlier version of this figure drew bars of differing height
# "normalised for readability"; those heights were invented and are gone. Every
# marker here is the same size, and the only thing encoded is the sign.
#
# Provenance, one analysis script per row:
#   m1  method-1-traits-logprob/scripts/analyse_method1_positive.py
#   m2  method-2-answer-quality/scripts/analyse_method2.py
#   m3  method-3c-wrappers/scripts/analyse_method3_wrapper.py
#   m4  method-4-numeric-estimates/scripts/run_method4.py --analyse-only
#   m6b method-6b-logprob-choice/scripts/analyse_method6b.py
# ============================================================================
def fig_disagreement():
    methods = ["m1  trait words", "m2  answer quality", "m3  task accuracy",
               "m4  numeric score", "m6b forced choice"]
    # -1 treats worse, +1 treats better. "mixed" = the models disagree, and
    # the disagreeing model is named in the footnote.
    screen = [-1, -1, -1, +1, -1]
    age = [-1, -1, -1, -1, -1]
    # Cells where at least one of the three models points the other way or is
    # null. Marked with a ring so the figure cannot be read as unanimity.
    screen_mixed = {2, 4}     # m3 mistral null; m6b mistral +7.14 (helps)
    age_mixed = {1}           # m2 inside the additivity residual on half the cells

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.3), sharey=True)
    for ax, vals, mixed, title in (
            (axes[0], screen, screen_mixed, "discloses a screen reader"),
            (axes[1], age, age_mixed, "is seventy-four")):
        for i, v in enumerate(vals):
            ax.scatter([v], [i], s=260, zorder=3,
                       color=BETTER if v > 0 else WORSE,
                       edgecolors="black" if i in mixed else "none",
                       linewidths=1.6)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        ax.set_xlim(-1.9, 1.9)
        ax.set_xticks([-1, 1])
        ax.set_xticklabels(["treats worse", "treats better"])
        ax.set_title(title, fontsize=10.5)
        ax.invert_yaxis()
        ax.grid(axis="y", color="#e6e6e6", lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(left=False)
    fig.suptitle("Direction only: one signal, five channels.  Four say worse, "
                 "the numeric score (m4) says better.", fontsize=10.5)
    fig.text(0.5, 0.015,
             "Ringed marker: the three models do not agree. m3 is null on "
             "Mistral; m6b on Mistral has the screen reader HELPING (+7.14).",
             ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    out = os.path.join(HERE, "fig_disagreement.png")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# ============================================================================
# FIGURE 2  method 6b, disclosure vs a socially neutral alternative detail
# ----------------------------------------------------------------------------
# Paired within question and profile: the choice margin for the disclosed
# candidate minus the margin for CONTROL_ALT ("cycles to work most days"), in
# judge logits. Negative = the disclosure is chosen LESS than an ordinary,
# socially irrelevant alternative fact about the same person. All BH < 1e-27.
#
# Provenance: method-6b-logprob-choice/scripts/analyse_method6b.py
# ============================================================================
def fig_method6b():
    signals = ["screen reader", "age 74", "Deaf", "ADHD"]
    qwen = [-5.99, -3.47, -5.87, -6.93]
    llama = [-1.62, -1.46, -0.58, -1.23]

    x = range(len(signals))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.bar([i - w / 2 for i in x], qwen, w, label="Qwen2.5-7B", color=WORSE)
    ax.bar([i + w / 2 for i in x], llama, w, label="Llama-3.1-8B",
           color="#e0997f")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(signals)
    ax.set_ylabel("choice margin vs a neutral\nalternative detail (judge logits)")
    ax.set_title("Method 6b: disclosing costs the candidate the choice\n"
                 "(negative = chosen less than 'cycles to work most days')")
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.5, 0.02,
             "Mistral is not shown: only 18% of its paired contrast clears"
             " the letter-mass gate, and on what survives\n"
             "the screen reader HELPS (+7.14). It is not a fourth agreeing"
             " case.", ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    out = os.path.join(HERE, "fig_method6b_control.png")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# ============================================================================
# FIGURE 3  method 7, item-level stated vs measured
# ----------------------------------------------------------------------------
# Spearman correlation between the model's own 0-10 self-rating on question Q
# and the measured change on question Q. Nine cells (3 answer models x 3
# signals). None survive Benjamini-Hochberg: the model knows the disclosure
# was there, not what it did.
#
# Provenance: method-7-stated-vs-measured/scripts/analyse_method7.py
# ============================================================================
def fig_method7():
    cells = [("qwen", "screen", 0.211, 0.3155), ("qwen", "age", 0.348, 0.0576),
             ("qwen", "adhd", 0.068, 0.7774), ("llama", "screen", 0.074, 0.7774),
             ("llama", "age", -0.193, 0.3155), ("llama", "adhd", -0.124, 0.6210),
             ("mistral", "screen", -0.003, 0.9826),
             ("mistral", "age", 0.257, 0.2123),
             ("mistral", "adhd", 0.051, 0.7939)]
    # Llama answered "6" to 57 of 60, so its self-report is a constant and its
    # three rows are an instrument failure, not a measured null.
    labels = [f"{m}  {s}" + ("  (degenerate)" if m == "llama" else "")
              for m, s, _, _ in cells]
    rhos = [r for _, _, r, _ in cells]
    colours = [NEUTRAL if bh >= 0.05 else WORSE for *_, bh in cells]

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.barh(range(len(cells)), rhos, color=colours, height=0.6)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels(labels)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel("Spearman rho: self-rating vs measured change, per question")
    ax.set_title("Method 7: the model does not know WHERE its answer changed\n"
                 "(0 of 9 cells significant after correction)")
    ax.invert_yaxis()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = os.path.join(HERE, "fig_method7_itemlevel.png")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# ============================================================================
# FIGURE 4  method 3c, the wrapper is the error bar
# ----------------------------------------------------------------------------
# Each signal measured six times, once under each of six neutral openings that
# carry no social content. The six dots are the six measurements; the bar is
# their mean. Unlike figures 1 to 3 this one is computed from the committed
# CSVs at draw time rather than transcribed, and a self-check below asserts it
# reproduces the published analysis.
#
# Provenance: method-3c-wrappers/scripts/analyse_method3_wrapper.py
# ============================================================================
WRAPPERS = ["W1", "W2", "W3", "W4", "W5", "W6"]
SIG_LABEL = {"S01": "screen\nreader", "S08": "age 74", "S04": "ADHD"}
MODEL_LABEL = {"qwen": "Qwen2.5-7B", "llama": "Llama-3.1-8B",
               "mistral": "Mistral-7B"}

# The published per-wrapper nets, asserted against what the CSVs give.
PUBLISHED = {
    "qwen":    {"S01": [10, 9, 6, 3, 10, 10], "S08": [6, 9, 5, 2, 6, 9],
                "S04": [4, 5, 5, 0, 4, 7]},
    "llama":   {"S01": [27, 19, 16, 23, 14, 25], "S08": [16, 9, 13, 20, 3, 20],
                "S04": [8, -5, 9, 11, 9, 9]},
    "mistral": {"S01": [2, 1, -1, 3, 3, 8], "S08": [-7, 6, 0, 3, 1, 5],
                "S04": [-9, 1, 0, -8, 1, 6]},
}


def _wrapper_nets(model):
    """signal -> six net-tasks-lost values, one per neutral wrapper."""
    import csv
    from collections import defaultdict
    path = os.path.join(HERE, "..", "experiments", "method-3c-wrappers",
                        "outputs", model, f"method3_wrapper_{model}.csv")
    if not os.path.exists(path):
        return None
    cells = defaultdict(dict)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        cells[r["cell"]][r["task_id"]] = int(r["correct"])
    out = {}
    for sig in SIG_LABEL:
        row = []
        for w in WRAPPERS:
            base, cond = cells.get(f"{w}_NONE", {}), cells.get(f"{w}_{sig}", {})
            shared = set(base) & set(cond)
            lost = sum(1 for t in shared if base[t] == 1 and cond[t] == 0)
            gained = sum(1 for t in shared if base[t] == 0 and cond[t] == 1)
            row.append(lost - gained)
        out[sig] = row
    return out


def fig_wrappers():
    data = {m: _wrapper_nets(m) for m in ("qwen", "llama", "mistral")}
    if any(v is None for v in data.values()):
        print("skipped fig_wrappers: method 3c CSVs not found")
        return
    for m, got in data.items():
        for sig, vals in got.items():
            assert vals == PUBLISHED[m][sig], (
                f"{m} {sig}: CSV gives {vals}, published {PUBLISHED[m][sig]}")

    sigs = ["S01", "S08", "S04"]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.9), sharey=True)
    for ax, m in zip(axes, ("qwen", "llama", "mistral")):
        for i, sig in enumerate(sigs):
            vals = data[m][sig]
            mean = sum(vals) / len(vals)
            ax.bar([i], [mean], width=0.62, color="#e6e6e6",
                   edgecolor="#bbbbbb", zorder=1)
            xs = [i + (k - 2.5) * 0.075 for k in range(6)]
            ax.scatter(xs, vals, s=26, zorder=3,
                       color=[WORSE if v > 0 else BETTER if v < 0 else NEUTRAL
                              for v in vals])
            ax.plot([i - 0.31, i + 0.31], [mean, mean], color="black",
                    lw=1.6, zorder=4)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(sigs)))
        ax.set_xticklabels([SIG_LABEL[s] for s in sigs])
        ax.set_title(MODEL_LABEL[m], fontsize=10.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("net tasks lost out of 200")
    fig.suptitle("Method 3c: the same signal measured under six neutral "
                 "wrappers.  Each dot is one wrapper.", fontsize=10.5)
    fig.text(0.5, 0.015, "The spread within a signal is the uncertainty a "
             "single-wrapper study inherits and does not report. On Mistral "
             "the dots straddle zero.",
             ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.07, 1, 0.91))
    out = os.path.join(HERE, "fig_method3c_wrappers.png")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_disagreement()
    fig_method6b()
    fig_method7()
    fig_wrappers()
