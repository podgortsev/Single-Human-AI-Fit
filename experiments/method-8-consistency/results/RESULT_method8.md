# Method 8. Consistency and stability

Is the model less stable for a person who has disclosed something?

No new runs. Three probes on data already collected: methods 1, 3c and 6b.

**The hypothesis "answers are much less stable for non-average users" does not
hold in general.** The one durable thread is that age 74 makes the choice about
a person more dependent on which decision is being asked. Everything else is
either zero or points the other way.

---

## What is measured

Not "does the model rate them lower" but "does its output scatter more when
something that should not matter is changed".

| probe | source | what varies although it should not | instability measure |
|---|---|---|---|
| A | method 1 | five prompt phrasings | SD of the trait log probability over the five |
| B | method 3c | six neutral wrappers | k(6-k) over the six solved / not-solved outcomes |
| C | method 6b | three decision questions (hire, promote, trust) | SD of the choice margin over the three |

Each probe is paired within item against a signal-free baseline (A and B against
NONE, C against CONTROL_PARA), Wilcoxon signed-rank, Benjamini-Hochberg across
the family.

A positive median means the disclosed condition scatters more than the baseline.

---

## Probe A. Spread across five phrasings (method 1)

| model | signal | median delta | SD ratio | BH | verdict |
|---|---|---|---|---|---|
| Qwen | screen | -0.505 | 0.66 | 0.0000 | **more stable** |
| Qwen | age | +0.037 | 1.00 | 0.53 | no effect |
| Qwen | Deaf | -0.258 | 0.81 | 0.0000 | **more stable** |
| Qwen | ADHD | -0.135 | 0.94 | 0.0000 | more stable |
| Llama | all four | -0.21 to -0.26 | 0.74 to 0.80 | 0.0000 | **all more stable** |
| Mistral | screen | -0.146 | 0.85 | 0.0000 | more stable |
| Mistral | **age** | **+0.219** | **1.35** | 0.0000 | **less stable** |
| Mistral | Deaf | -0.003 | 1.03 | 0.20 | no effect |
| Mistral | ADHD | -0.128 | 0.83 | 0.0000 | more stable |

In ten cells of twelve, disclosure makes the model **more** certain, not less.
Only Mistral with age destabilises (x1.35).

A caution: "more stable" is not a virtue here. Method 1 showed that under
disclosure the model applies **fewer** favourable words. Probe A says it does so
more confidently and more evenly across phrasings. Stability of a biased
judgement is not a good thing.

---

## Probe B. Disagreement across six wrappers (method 3c)

| | screen | ADHD | age |
|---|---|---|---|
| Qwen | no effect | no effect | no effect |
| Llama | no effect | no effect (BH 0.43) | no effect (BH 0.43) |
| Mistral | no effect | no effect | no effect |

Nine cells of nine: nothing. The median delta is zero everywhere, because most
tasks are unanimous across the six wrappers regardless of condition. Llama's
means are slightly positive (ADHD +0.39, age +0.36) but not significant.

Method 3c covers only screen reader, ADHD and age; Deafness is not in it.

---

## Probe C. Spread across three decision questions (method 6b)

| model | signal | median delta, logits | BH | verdict |
|---|---|---|---|---|
| Qwen | screen | +0.359 | 0.0000 | **less stable** |
| Qwen | age | +0.399 | 0.0000 | **less stable** |
| Qwen | Deaf | +0.327 | 0.0000 | **less stable** |
| Qwen | ADHD | +0.116 | 0.0000 | less stable |
| Llama | screen | +0.033 | 0.050 | borderline, no effect |
| Llama | age | +0.062 | 0.0000 | **less stable** |
| Llama | Deaf | +0.103 | 0.0000 | **less stable** |
| Llama | ADHD | -0.024 | 0.003 | more stable |
| Mistral | all | - | - | no clean data |

On Qwen all four signals make the choice about a person more dependent on the
question. On Llama, age and Deafness do. Mistral drops out: probe C needs all
three questions with mass on the letters for one profile, and Mistral almost
never has that, the same problem as in method 6b.

This partly rediscovers something already known from method 6b: on Qwen the
screen reader and Deafness disagreed across the three questions on the
wins-from-slot-B metric. But the CONTROL_PARA baseline is tight across
questions, so the signals genuinely **add** question-dependence rather than
inheriting it.

---

## Summary

| signal | less stable | more stable | no effect | cells |
|---|---|---|---|---|
| screen reader | 1 | 3 | 5 | 9 |
| age 74 | **3** | 1 | 5 | 9 |
| Deaf | 2 | 2 | 2 | 6 |
| ADHD | 1 | 4 | 4 | 9 |

A cell is one probe on one model.

---

## What can be claimed

**The hypothesis of general instability for non-average users is not
supported.** Of thirty-three cells, only seven are "less stable" and ten are
"more stable".

**Age 74 makes the choice about a person more dependent on the type of
decision** (probe C, Qwen and Llama, significant). It is the only signal that
destabilises in more than one probe and on more than one model.

**Disclosure generally narrows the distribution of trait words** (probe A, ten
cells of twelve). The model applies its lowered scores confidently rather than
hesitantly.

**Disclosure does not affect the stability of solving a task** (probe B, zero of
nine).

---

## What cannot be claimed

**That disclosure destabilises the model.** In most probes it narrows the
distribution instead.

**That "more stable" is good.** It is stability of a biased judgement.

**That anything happens on Mistral in probe C.** There is no clean data.

**That the three probes measure the same thing.** Probe A is about confidence in
a word, probe C is about switching a decision between types. They disagree in
sign, and that is expected.

---

## How this relates to the other methods

**Probe C and method 6b.** Method 6b found that on Qwen the screen reader and
Deafness give different signs across the three questions. Probe C formalises
that as instability and confirms the baseline does not produce it.

**Probe A and method 3c.** Method 3c measured the drift of the baseline from the
choice of wrapper, 2 to 5 points, and called it the floor for the whole study.
Probe A is the same kind of measurement transferred to trait words: disclosure
does not increase that drift, it decreases it.

**Probe A and method 1.** Method 1's sign (fewer favourable words under
disclosure) does not change; probe A adds that the shift is also more robust to
phrasing.

---

## Limitations

**This is the cheap version.** The probes are built on the number of repeats the
other methods' designs happened to provide: five phrasings, six wrappers, three
questions. No additional repeats were requested.

**Probe C has three points per profile.** An SD over three is noisy; what makes
it work is pairing across a hundred profiles.

**Mistral is absent from probe C** for want of mass on the letters.

**The expensive version was not run:** repeated sampling at temperature above
zero, many repeats per item. It is justified only if probe C needs confirming
directly.

**Models of 7-8 billion parameters at four-bit quantisation.**

---

## Files

`scripts/analyse_method8.py` - the three probes on the method 1, 3c and 6b CSVs.
No GPU, no new run.

`scripts/validate_method8.py` - offline check on synthetic data with injected
instability, ten tests.

`outputs/analyse_method8_console.txt` - the full output.

Sources: `../method-1-traits-logprob/outputs/`, `../method-3c-wrappers/outputs/`,
`../method-6b-logprob-choice/outputs/`.
