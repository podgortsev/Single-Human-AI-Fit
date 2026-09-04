# Limitations

What cannot be claimed from this work, and the design decisions that determine
how far each number reaches.

Most items here were learned by getting something wrong first. Where that is the
case it is said so, because a caveat with a history is easier to trust than one
asserted in the abstract.

---

## 1. Scope

**Three models, 7 to 8 billion parameters, 4-bit quantisation.** Qwen2.5-7B,
Llama-3.1-8B, Mistral-7B-v0.3. Nothing here establishes anything about the
large commercial systems most people use. Quantisation itself is uncontrolled:
no full-precision comparison was run.

**Repetition, and what counts as a run.** A "run" here means one pass of one
method over one model, producing one output CSV. On that definition:

- Methods 1, 2, 4, 6a, 7 and 8 were each run **once per model**. Nothing was
  repeated under identical settings, so run-to-run sampling variability is
  unmeasured for them.
- Methods 3a, 3b and 3c are **separate experiments, not repeats of one**. They
  share the 200 tasks but ask different questions (single signal; accumulation;
  robustness across six wrappers). Only 3c repeats the same measurement
  deliberately, six times over, and that repetition is the error bar reported
  there.
- Method 6b was run twice per model: once before a socially neutral alternative
  control existed and once after. The second run supersedes the first, which is
  kept in the repository because the two together are the evidence that the
  control mattered.

So no number in this study carries a same-settings replication except through
method 3c's six wrappers.

**Synthetic material throughout.** The 100 profiles are generated and uniform in
style; real CVs vary far more. The 200 arithmetic tasks are generated from seven
templates. The 60 open questions are hand-written to a fixed set of criteria.

**Four signals carry most of the weight**: screen reader, Deafness, ADHD, age
74. They are not a sample of anything. Conclusions are about these four
disclosures.

**The signal usually sits in one clause, often the opening sentence.** In real
use a whole message is written in one register. The rigour of the design was
bought with sensitivity, and the effect of writing style in particular is
probably underestimated for that reason.

---

## 2. What the effect sizes will and will not support

**The direction of the main effects is more robust than their magnitude.** This
is the single most important limitation in the study. It is a statement about
relative confidence, not a claim that the directions are settled: several are
model-specific, and method 6b's screen-reader result reverses on Mistral.

- Method 2's effect size varies **fivefold depending on which model judges**, so
  its logits are not comparable across judges.
- Method 4's spread on the same measure runs from zero to -99 percent between
  models.
- Method 3c shows the measured size moving 2 to 5 accuracy points with the
  choice of neutral wrapper.

Read every number as "this moved, in this direction, on this model". Not as a
quantity that would survive a change of setup.

**Units are not comparable across methods.** Log probability, judge logits, net
tasks and a 0-10 rating do not convert into one another. `fig_disagreement.png`
therefore encodes direction only; an earlier version of that figure drew bars of
differing height "normalised for readability", and those heights were invented.

---

## 3. Statistical decisions that could go the other way

**Method 1's statistical unit is the phrasing, not the profile.** That gives n=5
and df=4 for the headline test. It is the conservative choice: testing at
profile level flags 29 of 30 cells, including cells where the models disagree in
direction. A reader who prefers the profile-level unit gets far more
significance, and should be suspicious of it.

**Method 3a's group contrast and its per-condition tests disagree, and both are
reported.** The contrast between ten "form" and ten "stated" conditions is
significant on Qwen (p=0.0002), while **zero of 24 individual Qwen conditions**
survive Bonferroni correction. The group test has more power because it asks one
question of twenty conditions instead of twenty questions of one. Neither
result is the whole picture.

**Method 2's additivity residual is our criterion, not a standard procedure.**
Half the age and ADHD cells fall inside it. Those are a direction, not a size.

**Multiplicity correction was applied within each method's family of tests, not
across the study.** With seven methods there is no single family, and no
study-wide correction is claimed.

**Method 8's one positive thread rests on a standard deviation over three points
per profile.** It is a lead, not a finding.

---

## 4. Where a control changed the answer

**Method 6b's headline inverted when a control was added.** The first run
compared four disclosures against a single dull commute clause and concluded
that disclosure was *preferred*. It had no control carrying a different but
socially neutral detail. On Qwen, such a detail wins from the disfavoured slot in
96 percent of profiles, more than any disclosure does. Measured against it,
every disclosure loses.

The lesson generalises: **a control must vary the thing under test, not a
different thing.** The identical-detail control that the first run did have
returns zero by construction and is uninformative as a check on the correction:
it cannot fail, so passing it demonstrates nothing. Worse, in method 2 that
degenerate control inflated the non-additivity estimate roughly fivefold,
because when a judge is shown two identical answers it has nothing but position
to go on. The residual is therefore computed from the real comparisons, and the
degenerate control is reported but excluded from it.

**Rephrasing alone moves results.** Six arbitrary but defensible neutral
openings moved baseline accuracy by 2 to 5 points. What a signal-free control
shows is therefore the reference level for that measurement, and we do not
interpret differences at or below it as evidence of a signal effect.

---

## 5. Whether the model was answering at all

**Switching from parsing text to reading log probabilities does not remove
non-answers. It makes them harder to see.**

When a model declines to answer in the requested format, that shows up as **low
probability mass on the required answer tokens** rather than as a visible
refusal string. Scoring logP("A") against logP("B") is only a judgement if the
model was about to emit one of those letters; otherwise the ratio is computed
over two tokens the model was not going to produce, and it still yields a
number. P(A)+P(B) is therefore recorded on every row as a validity gate, and the
consequences are large:

- **Method 2**: Mistral carries no mass on the letters in 94 percent of reads.
  Excluded as a judge entirely.
- **Method 6b**: Mistral's mean letter mass is 0.40, with 61 percent of reads
  below half. Because the paired contrast needs all four reads clean, only
  **18 percent** of its design survives: 163 pairs of 900. Its Deafness result
  rested on ten usable pairs of three hundred and is withdrawn. Its PROMOTE
  question yields no clean pair at all.
- **Method 4**: Mistral refuses 19 to 28 percent of questions and returns 22 to
  37 percent unreadable answers. Two of ten measures had enough data.

**The letter-mass threshold of 0.5 was chosen after inspecting the
distributions, not before.** It is not pre-specified. `sensitivity_min_mass.py`
re-runs method 2 at 0.3, 0.5, 0.7 and 0.9; the conclusion is unchanged for both
usable judges, which is reassurance rather than proof.

**Refusal is counted separately from a low number.** On Llama, mentioning a
screen reader triples the refusal rate in method 4 (6.2 percent against 1.7).
That is its own finding, and averaging it into the scores would hide it.

---

## 6. Method-specific limits

**Method 2 is the only subjective method.** A model judges the answers. Its
result is reported separately and labelled. Method 7 inherits this entirely,
because method 2 supplies its measured side.

**Method 4's coherence check fails on Qwen for all four descriptions.** The same
person is judged more likely to succeed *and* more likely to leave. Those cells
are difficult to read as an evaluation whatever their p values. Incoherence does
not invalidate the individual tests; it makes them hard to interpret.

**Method 4 anchors.** Told a person is 74, models answer "74" to a question
about years of experience. On Qwen, mentioning age drops correct extraction from
95 to 53 of 100. A number copied from the prompt is not an estimate and is never
averaged in. The years-of-experience measure states its own answer in the
prompt, so it survives only as a control showing the profile is read.

**Method 6a failed and is published as a negative result.** Position bias does
not cancel: first-slot win rates of 67 / 99 / 92 percent, a parsed choice at a
ceiling, and a control detail that is not neutral. Its "won both / lost both"
counts assume the slot bias is constant in size; method 6b measured it varying
from 3.5 to 7.7 logits across signals on the same design, so they are not
position-proof either.

**Method 7 measures post-hoc attribution, not introspection.** The model is
asked after it has produced the answer and seen the disclosure, and is never
shown the counterfactual answer it would have given. A null therefore rules out
accurate after-the-fact attribution, not introspective access in general. Only
one operationalisation was tested: a 0-10 rating.

**Llama's method 7 self-report is degenerate.** It answers "6" to 57 of 60
questions. Its rows are an instrument failure and must not be read as denial.

**Method 3b is not a result of this paper**, and its accumulation findings are
reported in a separate study on signal stacking. One by-product of it is
retained here because it constrains everything else: its two runs of the same
design, differing only in the neutral filler, correlate at **r=0.13 on Llama and
r=0.22 on Qwen**. Individual conditions changed sign between them. Method 3c
exists because of this, and it limits how much any single-wrapper measurement in
this study can bear.

**Method 5, interaction cost, was never run.** It needs multi-turn dialogue.
Where the axes catalogue lists method 5, that is planned scope, not work done.

---

## 7. Known properties of the datasets

Found by `shared/tasks/validate_tasks.py`, which re-derives all 200 arithmetic
answers from the question text independently of the generator. All 200 keys are
correct. Three things the generator does not guarantee:

**200 rows over 191 distinct questions.** Nine rows repeat a question that
already appears, with consistent keys. Repeats are weighted twice. Five of the
nine are in the rate family.

**The date family is ambiguous.** "runs for N days" has an inclusive and an
exclusive reading; the key uses the exclusive one. A model using the other is
marked wrong on all 25 date tasks, in every condition alike, so it moves
baseline accuracy and cancels in the paired comparison.

**The table generator does not exclude ties.** A tie at the extreme a question
asks about would give two correct answers. The shipped set contains none; the
validator now asserts this so a regenerated set cannot ship one silently.

**None of the datasets is regenerated to tidy any of this up.** Every published
number was measured on these exact rows.

**Profile factors are not balanced.** Field, years, scope and achievement are
drawn independently: fields appear 2 to 9 times, scopes 6 to 17. Because every
condition faces all 100 profiles and each profile serves as its own control,
this imbalance primarily affects precision rather than the within-profile paired
comparison itself. It does limit any attempt to read effects *by* field or
seniority, which this study does not attempt. The achievement and scope clauses
carry prestige of their own and are not neutral background.

**Question domains are balanced in count, not in content.** Ten each across six
domains, but mean length runs from 12.6 words (people) to 15.2 (money), and a
good answer has a different shape in each.

**A signal can interact with a domain rather than with the person.** A screen
reader paired with a tech question may change the answer because it changes what
a helpful answer *is*, not because the model thinks less of the asker. Method 2
pairs within question, so the comparison is sound; the interpretation is not
settled by it.

**That answer quality "genuinely varies" across the 60 questions is an
assumption**, not a verified property. No independent rater study was run.
Method 2's result rests on it.

---

## 8. On the framing of the contribution

**Task-dependent bias is not claimed as a novel observation.** That measurements
of bias vary by task is established: see the survey by Gallegos et al.
(*Computational Linguistics*, 2024), the task-dependent stereotyping result in
arXiv:2604.02669, and FairFund-Bench (arXiv:2607.28934), which reports models
advantaging a group when rating individually and penalising it when ranking side
by side. That last is structurally the same pattern as our method 4 against
method 6b, on different material.

What this study adds is the unit of analysis: one person, one characteristic
changed, seven forms of interaction compared. Any claim that the sign-reversal
itself is new would be wrong, and is not made here or in `RESULTS.md`.

**The proposed reading, that bias is a property of person, task and model
together rather than of the model alone, is an interpretation.** It is
consistent with these data and with the prior work above. It is not established
by them, and this study was not designed to test it against competing
explanations.

---

## 9. Things that would strengthen this

In rough order of value per unit of effort.

1. **A second run of every method**, to put a sampling error bar on effects that
   currently have none.
2. **A set of neutral alternative controls in method 6b**, rather than the
   single "cycles to work most days". One neutral detail is one data point about
   what neutral means.
3. **Human raters on a subsample of method 2**, to test the assumption that
   quality varies and to calibrate the model judges against something outside
   the family.
4. **A full-precision run**, to separate quantisation effects from model
   effects.
5. **Larger models**, since the 7-8B range may not represent the systems in use.
6. **Pre-registration.** Expectations were written down for later methods but
   not for the earliest ones. With hundreds of numbers, some combination always
   supports something.
