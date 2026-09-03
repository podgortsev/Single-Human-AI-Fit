# Method 3b. Does the penalty grow when signals stack?

Four runs: two models, two lengths of neutral filler. 20,800 generations.

This experiment belongs conceptually to the signal-stacking work rather than to
the single-signal audit. It is reported here because its null, and the
reproducibility failure underneath it, bear directly on how every other number
in this study should be read.

---

## Design

**Six signals**, each of which produced an effect on its own in the
single-signal run: screen reader, ADHD, dyslexia, age, non-native language,
first time here.

**Eight routes.** Each adds signals one at a time in a fixed order, giving four
points: zero, one, two, three. The orders differ so the shape of the curve can
be separated from which signals happened to come first.

**The task is byte-identical** in every condition. Only the opening sentence
changes.

**Opening-sentence length is matched.** To forty words in the first pass, to
twenty-six in the second. The second pass was needed because forty words of
preamble cost Llama about ten accuracy points on their own.

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

**Llama at the corrected length gives a falling curve.** Three signals cost less
than one. One three-signal condition significantly beats the baseline.

**Qwen gives a weak rise**, p=0.031 uncorrected, but every depth lies within
reach of the control.

The claim "the further from average, the disproportionately worse" is not
supported by these data.

---

## The larger finding: the measurement does not reproduce

Same 24 conditions, same 200 tasks, same model, same code. The difference
between runs is fourteen words of neutral text.

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

Not the claim "models serve people who disclose worse". That did not reproduce.

But a claim about the measurement itself:

**The measured size of a bias is set by the design of the measurement as much as
by what is being measured. Changing fourteen words of neutral text inverts the
sign of the conclusion.**

Method 3c takes this seriously and measures it deliberately, across six neutral
wrappers.

---

## Methodological caveats

**The test.** Exact McNemar, a binomial test on discordant pairs. The asymptotic
chi-squared form was not used.

**The shape of the curve was assigned by a heuristic**, not by comparing fitted
models. The thresholds were chosen for convenience. A claim about shape needs a
second difference with a confidence interval, or a comparison of linear against
quadratic fits.

**Multiplicity correction was applied in the single-signal run and not here.**
Twenty-four comparisons per depth; none of the starred cells has been checked
against a correction.

**Two conditions are not independent of the rest:** the starred routes reach the
same set of signals by a different order. They cannot be treated as separate
observations when averaging over depth.

---

## Files

`outputs/<model>/method3_stack_<model>_40w.csv` - the forty-word pass.
`outputs/<model>/method3_stack_<model>_26w.csv` - the twenty-six-word pass.

`scripts/run_method3_stack.py` - the twenty-six-word run.
`scripts/run_method3_stack_40w.py` - the forty-word run.
