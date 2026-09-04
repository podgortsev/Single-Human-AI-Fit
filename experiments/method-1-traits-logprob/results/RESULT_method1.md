# Method 1. What the model thinks of a person

Measured by the model's own probabilities rather than by anything it says.

Three models, 100 profiles, five descriptions, six trait pairs, five prompt
phrasings. 30,000 scored sequences per model, 90,000 across the three.

---

## Design

The model is given an unfinished sentence, "This candidate is", and we measure
how readily it would continue with "intelligent" against "stupid".

Nothing is judged by a human or by another model. The readiness figure is
computed by the model itself.

**The same 100 profiles and five descriptions as methods 3 and 4.** Three
instruments measuring the same people.

**Five prompt phrasings kept as five separate estimates**, not averaged, so the
spread between them is visible rather than hidden.

### What the statistical unit is, and why

This matters enough to state plainly, because it is unusual.

Each phrasing gives one paired Cohen's d computed over roughly 100 profiles,
comparing the same profile under two descriptions. That part is a
within-profile paired comparison with n around 100.

The test reported below is then run **on those five numbers**. So the unit of
the test is the phrasing, n=5, df=4. Benjamini-Hochberg is applied across the
48 resulting tests (4 signals x 12 traits), not across profile-level tests.

That is deliberate, and it is the conservative choice, not a convenient one. The
alternative, testing the roughly 100 paired profiles directly, was computed for
comparison and is almost uninformative here:

| | phrasing-level t, n=5 | profile-level Wilcoxon, per trait-phrasing cell |
|---|---|---|
| Qwen age | p = 0.0008 | 29 of 30 cells significant at 0.05 |
| Qwen ADHD | p = 0.85 | 29 of 30 cells significant at 0.05 |

With around 100 paired profiles, nearly every cell is significantly non-zero,
including the ones where the direction disagrees between traits and phrasings.
The profile-level test answers "is this shift non-zero", which at this sample
size is almost always yes. The phrasing-level test answers "does this shift
point the same way whichever wording is used", which is the question worth
asking.

The cost is real: n=5 gives 4 degrees of freedom and very little power, and the
five phrasings are five hand-written openings, not a random sample from the
population of phrasings. Generalising beyond them is not warranted.

---

## Half the measure carries almost no signal

The first consistency check required both halves of an antonym pair to move in
opposite directions. Most pairs failed: 0 to 4 out of 6.

The cause is numeric. The readiness to produce a negative word is far lower:

| | intelligent | stupid |
|---|---|---|
| Qwen | -8.75 | **-15.10** |
| | hardworking | lazy |
| Qwen | -5.15 | **-11.97** |

The gap between positive and negative traits: Qwen 2.66, Llama 3.44,
Mistral 4.14.

Negative-trait log probabilities are substantially lower across all three
models. That is consistent with a floor effect that reduces the sensitivity of
the negative half of the measure. It is descriptive evidence for that reading,
not a demonstration of it: showing the negative side is genuinely at a floor
would need the distribution, not just the mean.

What the numbers do show directly is that the negative side carries no signal
here. Averaged over the six negative traits the effect is -0.04, +0.48, +0.05,
+0.14. Over the six positive traits, for the same descriptions, +0.57 to +2.92.

**The original check was rejecting the positive-side signal because it demanded
agreement from a half of the measure that moves almost not at all.**

## The corrected check

The same logic applied where the signal actually is. If a description changes
how favourably the person is judged, **all six positive traits must move the
same way**.

That is not a weaker requirement. It is the same requirement, put to the part of
the measurement that moves.

**This is a reanalysis, and it should be read as one.** The antonym-pair check
came first, it failed, and the criterion was then changed after seeing that
result. Nothing here was pre-registered. The reason for the change is
independently checkable, in that the negative side carries no signal on any
model or description, but the sequence was: criterion, failure, new criterion.

Two further caveats on the reported statistics:

- The agreement figure is a **sign test over six positive traits**, treated as
  six observations. They are not independent; "intelligent" and "educated" move
  together. Read it as an exploratory consistency check, not as independent
  confirmation.
- The verdict column applies a **reporting rule of our own**: same direction on
  all three models, with at least two models at 5/6 trait agreement. That is a
  stated criterion, not a statistical test, and it carries no p value.

---

## Result

| Description | Qwen | Llama | Mistral | Verdict |
|---|---|---|---|---|
| Age 74 | +2.00, 6/6 | **+2.92, 6/6** | +0.95, 5/6 | agrees on all three |
| Screen reader | +1.34, 6/6 | +1.07, 5/6 | +1.72, 5/6 | agrees on all three |
| Deaf | +1.05, 6/6 | +0.91, 4/6 | +0.57, 4/6 | same direction, weak agreement within models |
| ADHD | -0.09, 4/6 | +0.76, 4/6 | +0.31, 4/6 | models conflict |

A positive number means the model is **less** ready to apply the favourable word
after the description changes.

**Age.** All six traits move down on Qwen and Llama, five of six on Mistral. On
Llama the sign does not flip in any of the five phrasings.

**Screen reader.** Six of six on Qwen, five of six on the other two.

---

## Against the other methods, on the same 100 people

**Age is confirmed by three independent instruments.**

- Method 1: less ready to apply favourable words, three models.
- Method 3: significant, six wrappers of six, 12 measurements of 12.
- Method 4: lower score, rating and raise on all three models; on Llama by
  52 to 74 percent.

**The screen reader splits the instruments.**

- Method 1: less ready to apply favourable words, three models.
- Method 3: lower accuracy on the same task, 17 measurements of 18.
- Method 4: **higher** score and rating.

The model applies fewer good words to that person, solves their task less often,
and gives them a higher score.

The third does not cancel the first two. It covers them.

---

## Limitations

**Half the measure carries no signal.** The negative traits do not move, which
is consistent with training suppressing them, though that mechanism is inferred
rather than shown. Either way the sensitivity of the instrument is lower than
intended, and the positive side is doing all the work.

**Sign stability across phrasings holds only for age**, on Qwen and Llama. For
the other descriptions individual traits change sign between prompts.

**The five phrasings are not an error bar in the statistical sense.** They show
that estimates move with wording; they are too few, and too deliberately chosen,
to quantify how much. Method 3c measures wording sensitivity properly, on six
neutral wrappers, and finds a drift of 2 to 5 accuracy points. That figure, not
this one, is the study's estimate of what phrasing costs.

**"hardworking" is three tokens on Llama and Mistral**, and so is "trustworthy"
on Mistral. Those positions are noisier than the rest.

**The statistical unit of the reported test is the phrasing, n=5, df=4.** See
the design section. The within-phrasing comparison is over roughly 100 paired
profiles, but that is not the n of the test.

**Models of 7-8 billion parameters at four-bit quantisation.**

---

## Files

`outputs/<model>/method1_<model>.csv` - raw output, 30,000 rows each.

`scripts/run_method1.py` - the run.

`scripts/analyse_method1_positive.py` - the corrected analysis, no GPU.
