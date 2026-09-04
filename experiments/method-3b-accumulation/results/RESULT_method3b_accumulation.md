# Method 3b. Does the penalty grow when signals stack?

Four runs: two models, two lengths of neutral filler. 20,800 generations.

This experiment belongs conceptually to the signal-stacking work rather than to
the single-signal audit. It is reported here because its null, and the
reproducibility failure underneath it, bear directly on how every other number
in this study should be read.

---

## Design

**Six signals**, selected because they showed notable effects in the
single-signal run. That is selection on the outcome, so this experiment is a
follow-up rather than an independent confirmatory test. They are: screen reader,
ADHD, dyslexia, age, non-native language, first time here.

**Eight routes.** Each adds signals one at a time in a fixed order, giving four
points: zero, one, two, three. The orders differ so the shape of the curve can
be separated from which signals happened to come first.

**The task question and its answer key are identical** in every condition. Only
the opening sentence changes, so the full prompt is not byte-identical.

**Opening-sentence length is targeted, not matched exactly.** Forty words in the
first pass; whole neutral clauses targeting twenty-six in the second, which in
practice gives 23 to 26 words. Because signals differ in length, lead-in length
still varies a little with depth, so message length is not fully eliminated as
an explanation. The second pass was needed because the forty-word neutral
preamble materially changed Llama's accuracy: its control scored 6 percentage
points **above** the baseline, a paired net of -12 at p=0.008. The preamble did
not cost Llama accuracy, it moved it, and that is worse for a baseline.

**Which pass is primary.** The 26-word pass is the canonical one. The 40-word
pass is a preliminary run, excluded from the primary read because its neutral
control altered task accuracy. It is kept because the pair of runs is the
evidence for the reproducibility problem below.

**Control:** the same length, neutral words, no signal.

---

## Numbers

| Run | Baseline | Control | Depth 1 | Depth 2 | Depth 3 | Shape |
|---|---|---|---|---|---|---|
| Llama, 40 words | 34.5% | **-12** | +3.8 | +6.5 | +9.0 | rising |
| Llama, 26 words | 36.5% | +1 | +8.1 | +3.2 | -1.1 | falling |
| Qwen, 40 words | 60.5% | -1 | +4.2 | +5.6 | +5.6 | flat |
| Qwen, 26 words | 57.5% | +2 | +2.4 | +2.8 | +4.6 | weakly rising |

The Llama forty-word run cannot be read: the control beat the baseline by 12 at
p=0.008, so the baseline is not a stable reference point.

---

## There is no accumulation

In none of the four runs does the penalty grow faster than linearly.

**Llama at the corrected length gives a falling curve.** On the route means,
three signals cost less than one. One three-signal condition beats the baseline
at a nominal, uncorrected p=0.038; no multiplicity correction was applied at
this depth, so it is not a finding on its own.

**Qwen gives a weak rise that does not hold up.** The originally reported
p=0.031 came from a Mann-Whitney U across routes, which treats depth 1 and
depth 3 as independent groups. They are not: the same eight routes carry both.
Re-tested as a paired Wilcoxon signed-rank on the route-level difference
D3 - D1, it is **p=0.0625**, and a sign test gives the same. The comparison in
the shipped console logs is the old unpaired one; the code has since been
changed to the paired test.

| run | D1 mean | D3 mean | Mann-Whitney (old) | Wilcoxon paired (correct) |
|---|---|---|---|---|
| Qwen 26w | +2.4 | +4.6 | 0.0306 | **0.0625** |
| Qwen 40w | +4.2 | +5.6 | 0.1561 | 0.0547 |
| Llama 40w | +3.8 | +9.0 | 0.0629 | 0.0430 |
| Llama 26w | +8.1 | -1.1 | 0.9714 | 0.9961 |

Every depth also lies within a few net tasks of the control, which is a display
heuristic with an arbitrary margin, not a test.

The claim "the further from average, the disproportionately worse" is not
supported by these data.

---

## The larger finding: the measurement is highly sensitive to the wrapper

Same 24 conditions, same 200 tasks, same model, same code. The difference
between runs is the neutral wrapper: forty words of padding against
whole-clause padding targeting twenty-six.

**Correlation between runs: r = 0.13 on Llama, r = 0.22 on Qwen.** Both
indistinguishable from zero.

Individual conditions change sign. S04+S09+S01 was +10, became -9.
S01+S05+S08 was +3, became -14.

The shape of Llama's curve inverted from rising to falling.

## This reaches back to the single-signal result

Six signals measured three times, in three different surrounding texts.

| Signal | Short lead-ins, 9-15 words | 40 words | 26 words | Spread |
|---|---|---|---|---|
| S01 screen reader | +24 | +8 | +16 | 16 |
| S04 ADHD | +25 | +2 | -2 | **27** |
| S05 dyslexia | +14 | +5 | +3 | 11 |
| S08 age | +22 | +8 | +15 | 14 |
| S09 non-native | +12 | 0 | +9 | 12 |
| S10 first time | +12 | -1 | -1 | 13 |

Llama data. Correlation between runs r=0.31 and r=0.38, neither significant.

**The single-signal conclusion has to be qualified.** The effect exists in a
particular presentation and does not survive a change in the surrounding
neutral text.

---

## What follows from this

Not the claim "models serve people who disclose worse". It did not hold up
across wrappers.

But a claim about the measurement itself:

**The measured size of a bias is set by the design of the measurement as much as
by what is being measured. Changing the neutral wrapper inverts the sign of the
conclusion.**

Stated more carefully than "the measurement does not reproduce": the measurement
is highly sensitive to the surrounding neutral text. Whether that counts as
non-reproducibility or as an uncontrolled moderator is a matter of framing, and
either way it is the reason method 3c exists.

Method 3c takes this seriously and measures it deliberately, across six neutral
wrappers.

---

## Methodological caveats

**The test.** Exact McNemar, a binomial test on discordant pairs. The asymptotic
chi-squared form was not used.

**The shape of the curve was assigned by a heuristic**, not by comparing fitted
models. The thresholds were chosen for convenience, and small changes in the
increments can flip the label between COMPOUNDING and MIXED. A claim about shape
needs a second difference with a confidence interval, or a comparison of linear
against quadratic fits. Read the labels as descriptions of the printed numbers,
nothing more.

**Depth 1 and depth 3 are paired within route**, which the original Mann-Whitney
comparison ignored. Corrected above.

**The eight routes are not eight independent signal sets.** Routes that reach
the same combination of signals by a different order are marked with a star, and
some single signals appear at depth 1 in more than one route. Treating the eight
as independent observations overstates the evidence, and it is why the depth
comparison is reported as a paired test over routes rather than as n=8
independent measurements.

**Multiplicity correction was applied in the single-signal run and not here.**
Twenty-four comparisons per depth; none of the starred cells has been checked
against a correction.

---

## Files

`outputs/<model>/method3_stack_<model>_40w.csv` - the forty-word pass.
`outputs/<model>/method3_stack_<model>_26w.csv` - the twenty-six-word pass.

`scripts/run_method3_stack.py` - the twenty-six-word run.
`scripts/run_method3_stack_40w.py` - the forty-word run.
