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
# Direction each method points for a person who discloses a screen reader or
# age 74. Negative = the model treats that person WORSE on that channel;
# positive = better. Values are normalised to [-1, 1] within a method for
# readability; the sign and the significance are what matter, not the height.
#
# Provenance:
#   m1  method-1-traits-logprob/scripts/analyse_method1_positive.py
#   m2  method-2-answer-quality/scripts/analyse_method2.py
#   m3  method-3c-wrappers/scripts/analyse_method3_wrapper.py
#   m4  method-4-numeric-estimates/scripts/run_method4.py --analyse-only
#   m6b method-6b-logprob-choice/scripts/analyse_method6b.py
# ============================================================================
def fig_disagreement():
    methods = ["m1\ntrait words", "m2\nanswer quality", "m3\ntask accuracy",
               "m4\nnumeric score", "m6b\nforced choice"]
    # -1 worse, +1 better, 0 not applicable / null
    screen = [-0.75, -0.90, -0.85, +0.80, -0.80]
    age = [-0.90, -0.55, -0.80, -0.85, -0.55]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4), sharey=True)
    for ax, vals, title in ((axes[0], screen, "discloses a screen reader"),
                            (axes[1], age, "is seventy-four")):
        colours = [BETTER if v > 0 else WORSE for v in vals]
        ax.barh(range(len(methods)), vals, color=colours, height=0.62)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        ax.set_xlim(-1.15, 1.15)
        ax.set_xticks([-1, 1])
        ax.set_xticklabels(["treats worse", "treats better"])
        ax.set_title(title, fontsize=10.5)
        ax.invert_yaxis()
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(left=False)
    fig.suptitle("The central disagreement: one signal, five channels.  "
                 "Four say worse, the numeric score (m4) says better.",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
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
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
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
    fig.tight_layout()
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
    labels = [f"{m}  {s}" for m, s, _, _ in cells]
    rhos = [r for _, _, r, _ in cells]
    colours = [NEUTRAL if bh >= 0.05 else WORSE for *_, bh in cells]

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
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
