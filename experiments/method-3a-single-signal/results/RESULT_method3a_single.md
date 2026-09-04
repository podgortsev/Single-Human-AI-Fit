# Method 3a. One signal at a time

Does the model solve the same task less often depending on who appears to be
asking?

Three models, 26 conditions, 200 tasks, 15,600 generations.

---

## Design

**The task text is byte-identical** in every condition. Only a short opening
sentence changes.

Two hundred tasks with a checkable answer: percentages, multi-step arithmetic,
unit conversion, dates, table lookup, logic, speed. The key is computed by the
same code that writes the question, so the key is correct by construction.

**Twenty-six conditions.** A baseline, a control with no signal, ten ways of
writing, ten statements about oneself, four behavioural, and unfamiliarity with
terminology.

**Three models:** Qwen2.5-7B, Llama-3.1-8B, Mistral-7B-v0.3. Different
developers, different countries, different training data.

**Paired test.** Not a comparison of average accuracy but a count of which tasks
were lost. Only discordant pairs carry information: tasks the baseline solved
and the condition did not, and the reverse. The rest are silent.

**Control.** A condition with different wording and no signal at all. Whatever
it shows is the cost of rephrasing on its own. A condition has to clearly exceed
it to mean anything.

**Multiplicity.** Twenty-four comparisons, threshold 0.0021.

---

## Numbers

| | Qwen | Llama | Mistral |
|---|---|---|---|
| Baseline accuracy | 60.0% | 44.0% | 50.0% |
| Control, net loss | +4 | 0 | -4 |
| Way of writing, mean of 10 | +1.10 | +5.70 | -0.70 |
| Stated about self, mean of 10 | +8.20 | +15.10 | -0.90 |
| Difference, Mann-Whitney U | p=0.0002 | p=0.0009 | p=0.52 |

Pooled across models, descriptive only: form +2.03, stated +7.47, p=0.0018.

**What that test is.** Ten form conditions against ten stated conditions,
Mann-Whitney U, one-sided. The unit is the condition and the value is its net
loss against the baseline, so n is 10 against 10, not 200 against 200. The
pooled row stacks the same ten conditions from three models as though they were
thirty independent observations, which they are not; it is reported as a
description, not as a test. Regenerate all of it with
`scripts/analyse_method3a_groups.py`.

An earlier version of this document quoted p values of 0.0005, 0.0014 and 0.004
for this contrast. No committed script produced them. They have been replaced by
the numbers above, which any reader can reproduce.

### Per condition, almost nothing survives correction

The group contrast above and the per-condition tests answer different questions,
and they do not agree. With a Bonferroni threshold of 0.0021 for 24 comparisons:

| model | conditions surviving correction |
|---|---|
| Qwen | **0 of 24** |
| Llama | 5 of 24 |
| Mistral | **0 of 24** |

So no individual Qwen or Mistral condition is a finding. The group contrast has
more power than 24 separate tests because it asks one question of twenty
conditions instead of twenty questions of one condition each, and that is why it
reaches significance on Qwen where no single condition does. Both facts belong
in any honest summary.

---

## The main result

**On Qwen and Llama, stating something about yourself cost more accuracy than
writing differently. Mistral showed no comparable effect.**

On the two models where anything happens, a person who mentions ADHD, a screen
reader, their age, or that this is their first time here solves the same task
less often than the baseline, and by more than a person writing in a dialect,
with typos, or informally. On Mistral neither group differs from the baseline
and the two groups do not differ from each other.

The strength of the claim sits in the group contrast, not in any single
condition: on Qwen no individual condition survives correction.

**Survived correction on Llama, five conditions:**

ADHD +25, wrong terminology +25, screen reader +24, age 74 +22, first time here
+12.

Four of those five are statements about the person: ADHD, screen reader, age,
first time here. The fifth, **wrong terminology, is a different kind of signal**
and should not be read alongside them. It reports something about the user's
command of the vocabulary, not about who they are, and it plausibly makes the
question genuinely harder to interpret. Grouping it with accessibility and age
disclosures in one headline would overstate the case.

**Positive on all three models at once, five conditions:**

wrong terminology, dyslexia, non-native language stated, Deafness, first time
here. Positive on all three, but on Mistral none of these is distinguishable
from noise, so this is a direction and not a demonstration.

---

## Mistral shows nothing

Every condition sits within noise, and the control is -4.

This is not a failed run. It is the third case of the same thing. The model
appears simply not to attend to the opening sentence.

So the effect is not universal. It is a property of particular models, not of
language models in general.

---

## Against the published literature

Existing work disagrees about whether naming a trait explicitly helps or makes
matters worse.

These data speak to it directly, on one body of material, through two designed
pairs where the same circumstance is presented two ways: non-native English
shown by the writing (F01) or declared (S09), and imprecise typing shown by
typos (F04) or declared as a motor impairment (S07).

| model | non-native: shown / declared | typing: shown / declared |
|---|---|---|
| Qwen | +0 / +7 | +2 / +12 |
| Llama | +6 / +12 | +10 / +9 |
| Mistral | +2 / +3 | -5 / -4 |

**This does not establish that explicit mention hurts and the implicit form does
not.** Declaring costs more in four of the six comparisons, but on Llama the
typing pair goes the other way (+10 shown against +9 declared), and Mistral's
numbers are noise. With one pair per comparison there is no test to run here,
only six numbers that mostly lean one way. The group contrast above is the
stronger evidence for the same idea, and it is the one to cite.

---

## Practical consequence

People are taught to state their needs in order to get better service.

These data raise a concern about that advice without settling it: stating an
accessibility need or a personal characteristic did not improve the model's
performance on the same task, and on two of three models it was associated with
lower accuracy. That is a finding about three open models on 200 arithmetic
tasks, not a general claim that people are worse off disclosing to an AI
system.

---

## Limitations

**The signal lives only in the opening sentence.** In real use the whole message
is in one register, and the effect of writing style may be underestimated for
exactly that reason. The rigour of the design was bought with sensitivity.

**The tasks are arithmetic.** On tasks with no single right answer the picture
may differ. That is method 2.

**Models of 7-8 billion parameters at four-bit quantisation.**

**Two hundred tasks.** Between eight and forty-eight discordant pairs per
condition. Weak effects are not resolvable at this size.

**Two conditions, F05 and F06, are not comparable with the rest**, because there
the length of the opening sentence is itself the property being measured.

**Measured at one wrapper.** Method 3c repeats this across six neutral wrappers
and shows how much of it survives. Read the two together.

---

## Files

`outputs/<model>/method3_single_<model>.csv` - raw output, one row per
generation, 5,200 rows each.

`../../shared/tasks/tasks.json` - 200 tasks with keys.

`scripts/run_method3_single.py` - the run and its built-in per-condition
analysis. Note that its printed summary says "twenty-five conditions" where the
correction it applies uses 24; the arithmetic is right and the prose is off by
one. The source string has been corrected for future runs, and the shipped log
is left as it was produced.

`scripts/analyse_method3a_groups.py` - the form-against-stated contrast and the
two designed pairs. No GPU.
