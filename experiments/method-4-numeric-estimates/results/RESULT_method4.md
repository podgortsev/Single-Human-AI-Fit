# Method 4. A number with no right answer

Does the model name a different number about a person when one detail in their
description changes?

Three models, ten measures, five descriptions, 100 profiles. 15,000 generations.

---

## How this differs from method 3

Method 3 used arithmetic with a single correct answer. There is nowhere to
drift: the answer does not depend on who is asking.

Here there is no right answer. The model is estimating, so it has latitude. That
latitude is what is being measured.

**The signal sits in the description of the person being judged**, not in the
voice of the person asking. That is how decisions about hiring, credit and
insurance are actually structured.

**The task stays the same.** One subordinate clause changes: "commutes from a
nearby town" against "uses a screen reader at work".

---

## How it is measured

No subjectivity enters. We are not judging whether a number is fair. We are
comparing two numbers named about the same profile under two descriptions.

**Paired Wilcoxon** on within-profile differences. The pilot showed why this is
critical: comparing medians by group found nothing, while the paired test on the
same data found eleven significant shifts out of twenty. A coarse scale is not a
dead scale when the comparison is within-profile.

**Benjamini-Hochberg** across all comparisons.

**Four response statuses.** A number without hedging, a number with hedging, a
refusal with no number, and a number copied out of the prompt. A refusal is not
a small number; it is a different event.

---

## The ten measures

Allocation: salary, credit limit, percentage raise.
Prediction: probability of success, probability of leaving within two years.
Judgement: suitability score, rating out of ten.
Properties: apparent years of experience, days for a project, times per month
they need help.

Each is a number **about the person**. Temperature, area and latency were not
used: a description of a person cannot affect them by construction.

---

# What all three models agree on

## Age seventy-four pulls everything down

Four measures agree in direction on all three models: percentage raise,
suitability score, rating, need for help.

The sizes on Llama and Mistral are large:

| Measure | Llama | Mistral |
|---|---|---|
| Percentage raise | **-74%** | - |
| Suitability score | **-52%** | **-99%** |
| Rating | -31% | -77% |
| Probability of success | -99% | - |

On Llama's score, 80 profiles of 96 move down and 7 move up.

## Deafness and screen reader use raise the scores

Score and rating go up on all three models, significant everywhere.

Rating with Deafness: Qwen +13%, Llama +9%, Mistral +13%. On Qwen 80 profiles of
100 go up, zero go down.

---

# What they do not agree on

**Thirteen direct conflicts between models.**

ADHD is a complete divergence. Qwen raises salary, credit, raise, score and
rating. Llama lowers all of them, significantly.

The same description, opposite decisions.

Size agrees nowhere. On the score with age: Qwen zero, Llama minus 52 percent,
Mistral minus 99 percent.

---

# The coherence check fails

Probability of success and probability of leaving ask opposite questions. They
are obliged to move in opposite directions.

**Qwen: all four descriptions are incoherent, each on two measures of three.**

The model simultaneously asserts that the person is more likely to do well, more
likely to leave, and slower on a project.

Llama is better: two of four descriptions are coherent.

So some of Qwen's significant shifts cannot be read as a judgement about the
person. It is an incoherent set of numbers, and small p values do not rescue it.

---

# Three findings that method 3 could not produce

## Refusals depend on the description

On Llama, mentioning a screen reader gives **6.2% refusals against 1.7% at
baseline**. Three and a half times more often. With Deafness, 4.8%.

The model refuses to name a number about a person with a disability more often.

This is a distinct kind of exclusion: not a lower estimate but **no estimate**.
Method 3 had no refusals at all, so this is specific to evaluative questions.

Mistral refuses 19 to 28 percent under every description, so its refusal is
about the question rather than about the person.

## Fact extraction breaks when age is mentioned

The years-of-experience measure works as a correctness control: the answer is
**stated in the prompt**, and the model takes it from there. That proves the
profile is being read rather than ignored.

Correct extractions out of 100:

| | Baseline | Screen | Age | Deaf | ADHD |
|---|---|---|---|---|---|
| Qwen | 95 | 93 | **53** | 88 | **41** |
| Llama | 93 | 88 | 89 | 92 | 89 |
| Mistral | 95 | 100 | 99 | 100 | 100 |

**On Qwen, mentioning age drops extraction from 95 to 53**, and in 27 cases it
answers "74", substituting the age for the tenure.

With ADHD it falls to 41, and 74 never appears; the model names some third
number.

This is not bias in an estimate. It is **loss of information that was given
explicitly in the text**, caused by a mention about the person.

## Mistral is close to unusable for evaluative questions

Nineteen to twenty-eight percent refusals and twenty-two to thirty-seven percent
unreadable answers. Of ten measures, two had enough data.

---

# What can be claimed

**Age seventy-four lowers estimates on all three models**, across four measures,
significantly, in places by tens of percent.

**Mentioning Deafness or a screen reader raises the score and the rating** on
all three models.

**Mentioning a disability raises the chance of refusing to answer** on Llama, by
a factor of three.

**Mentioning age breaks the extraction of a directly stated fact** on Qwen.

---

# What cannot be claimed

**That the direction is the same for every signal.** ADHD gives opposite answers
on different models.

**That the size is known.** The spread between models runs from zero to minus
ninety-nine percent on the same measure.

**That Qwen's numbers reflect a judgement about the person.** The coherence
check fails on all four descriptions.

---

# How this relates to method 3

In method 3, mentioning a screen reader **lowered** the share of correct answers
on the same task, robustly, 17 measurements of 18.

Here the same mention **raises** the estimates of the person.

There is no contradiction: these are different things. The model solves the
person's task worse and rates the person higher. Both are bad in their own way,
and the second can mask the first.

---

# Limitations

**The years-of-experience measure contains its answer in the prompt.** Used as a
correctness control, not as a measure of judgement.

**Money measures are sticky.** Salary and credit on Qwen give a median shift of
zero alongside a significant paired test: the scale is coarse, the direction is
visible, the size is not.

**Models of 7-8 billion parameters at four-bit quantisation.**

**Profiles are generated programmatically** and uniform in style. Real CVs are
more varied.

**One run per model.** Robustness to the choice of phrasing, as in method 3c,
was not checked here.

---

# Files

`outputs/<model>/method4_<model>.csv` - raw output, 5,000 rows each.

`scripts/run_method4.py` - the run and its analysis, profiles included.

`pilot/pilot_method4.py` - the pilot on twenty profiles, and its output. What it
taught is written into the runner's docstring.
