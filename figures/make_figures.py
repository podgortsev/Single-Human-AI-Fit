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


if __name__ == "__main__":
    fig_disagreement()
    fig_method6b()
    fig_method7()
