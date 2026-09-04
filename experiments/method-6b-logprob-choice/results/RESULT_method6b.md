# Method 6b. Head to head, choice read as a log probability

The rebuild of method 6a. No letter is parsed; logP("A") and logP("B") are read
at the answer position.

Three models, three questions, seven comparisons, 100 profiles, two orders.
12,600 forward passes.

**Disclosing costs the candidate the choice on Qwen and Llama.** Against a
socially neutral alternative detail, all four disclosures lose, on both models,
significantly. Mistral is a separate case.

---

## Why the first version of this document was wrong

The first 6b run, without `CONTROL_ALT`, said the opposite: that Qwen **prefers**
the candidate who disclosed, and that age 74 wins from the disfavoured slot in
82% of profiles. That was an artefact of the control.

Every signal was being compared against one and the same clause, "commutes from
a nearby town". When a control carrying **different content that is still
socially neutral** was added ("cycles to work most days"), it beat every signal:

| Qwen, wins from slot B | share of profiles | mean m2 |
|---|---|---|
| **CONTROL_ALT, cycling** | **96%** | **+6.36** |
| age 74 | 82% | +2.89 |
| screen reader | 56% | +0.37 |
| Deaf | 52% | +0.48 |
| ADHD | 40% | -0.57 |
| CONTROL_PARA, paraphrase | 0% | -4.55 |
| CONTROL_ID, same detail | 0% | -10.18 |

The model does not prefer disclosure. It prefers **any alternative to a dull
commuting clause**, and the disclosures turned out to be the worst of the
alternatives on offer.

The lesson, now in the methodology notes: a control must vary the same thing the
condition varies. A control that compares a thing with itself is zero by
construction and proves nothing.

---

## The main result: signal against a neutral detail

Paired within question and profile. Both conditions are measured against the
same reference clause and in the same slot, so both the slot bias and the
reference clause cancel. This is the only comparison that separates "this is a
disclosure" from "this is not a commute".

Negative means the disclosed candidate is chosen less often than one carrying an
ordinary irrelevant fact.

### Qwen

| signal | n | delta m2 | 95% CI | share worse | BH |
|---|---|---|---|---|---|
| ADHD | 300 | **-6.93** | [-7.24, -6.62] | 99% | 1.5e-50 |
| screen reader | 300 | **-5.99** | [-6.23, -5.75] | 99% | 1.5e-50 |
| Deaf | 300 | **-5.87** | [-6.26, -5.48] | 95% | 1.2e-47 |
| age 74 | 300 | **-3.47** | [-3.90, -3.04] | 80% | 5.1e-33 |

### Llama

| signal | n | delta m2 | 95% CI | share worse | BH |
|---|---|---|---|---|---|
| screen reader | 300 | **-1.62** | [-1.71, -1.53] | 96% | 1.3e-49 |
| age 74 | 300 | **-1.46** | [-1.54, -1.38] | 98% | 5.6e-50 |
| ADHD | 300 | **-1.23** | [-1.32, -1.14] | 92% | 7.4e-48 |
| Deaf | 300 | **-0.58** | [-0.66, -0.50] | 76% | 6.4e-28 |

### Mistral

| signal | n | delta m2 | 95% CI | share worse | BH |
|---|---|---|---|---|---|
| screen reader | 41 | **+7.14** | [+6.68, +7.60] | 0% | 7.3e-08 |
| age 74 | 71 | **-1.83** | [-2.61, -1.05] | 73% | 1.1e-05 |
| ADHD | 51 | +0.23 | [-0.16, +0.63] | 35% | 0.069, not significant |
| Deaf | - | - | - | - | withdrawn, 10 pairs |

**Llama is readable after all.** The first version called it saturated. That was
true of the absolute question "does it win from slot B", where every signal is
zero. But the difference between two conditions measured in the same slot
cancels the slot bias, and on that Llama gives a consistent result across all
four signals.

---

## Does the model intend to answer with a letter?

`P(A) + P(B)` is the share of probability mass sitting on the two letters. If it
is small, the difference of logarithms is a ratio of two tails, not a judgement.

| | mean letter mass | share of reads below 0.5 |
|---|---|---|
| Qwen | **1.00** | 0% |
| Llama | 0.94 | 0% |
| Mistral | **0.40** | **61%** |

Qwen and Llama are clean. Mistral is not, and it is worth being exact about how
much of it survives, because two different fractions are easy to confuse.

- **39% of individual reads** clear the 0.5 letter-mass gate (61% fall below it).
- **18% of the paired contrast survives.** The headline test pairs a signal
  against CONTROL_ALT within question and profile, which requires all four reads
  (two conditions x two orders) to clear the gate. That leaves 41 pairs for the
  screen reader, 71 for age and 51 for ADHD, out of 300 possible each: **163 of
  900, or 18%**.

So the Mistral rows in the table below rest on under a fifth of the design, not
on a third of it. The PROMOTE question yields no clean pair at all on any
signal, and the Deafness result is withdrawn on ten pairs of three hundred. This
is the refusal from 6a (14.8% unreadable strings), which 6b did not remove but
moved into the probability mass.

---

## Design

**By log probability, not by parsing.** Method 6a died on a parsed letter
hitting a ceiling. A difference of logarithms is continuous.

**Three controls.**

| | what it is | what it gives |
|---|---|---|
| CONTROL_ID | the same detail on both candidates | raw slot bias: Qwen 10.2, Mistral 6.6, Llama 3.8 logits. `combined` here is zero by construction and carries nothing |
| CONTROL_PARA | a paraphrase of the reference detail | what a mere change of words does |
| CONTROL_ALT | a different, socially neutral detail | **the floor a signal is obliged to clear** |

**The slot bias is not additive.** The `position` term ranges from 1.7 to 10.2
logits depending on the signal on Qwen alone, so `combined = (m1+m2)/2` is not
fully cleaned and overstates. Do not read it as an effect.

---

## What can be claimed

**On Qwen and Llama, disclosing any of the four signals lowers the chance of
being chosen** against an ordinary irrelevant fact about the same person.
Significant, all four signals, both models, robust to position and to the
reference clause itself.

**The size on Qwen is large:** 3.5 to 6.9 logits, that is from roughly thirty
times to roughly a thousand times less likely in odds terms. On Llama the same
signs at 0.6 to 1.6 logits.

**On Mistral the screen reader helps instead** (+7.14 on 41 pairs). Age hurts.
Nothing else is readable.

**The first-slot bias is large and not additive.** Qwen 10.2, Mistral 6.6,
Llama 3.8 logits. Simple order averaging does not remove it.

---

## What cannot be claimed

**That disclosure is preferred.** The first version of this document said so and
it was an artefact of the control. Withdrawn.

**That the size is precisely known.** The slot bias is not additive and the
correction is incomplete.

**That Mistral's Deafness result shows anything.** Ten usable pairs. Withdrawn.

**That Mistral is readable at all.** 39% of its reads clear the letter-mass
gate, but only 18% of the paired contrast survives it, and the PROMOTE question
is unusable entirely.

**That "cycling" is an ideal zero.** It is one particular neutral detail. A set
of them would be better, as method 3c used six wrappers.

---

## How this relates to the other methods

**The conflict with method 4 over age is resolved.** The first version gave
"Qwen picks the 74-year-old", which contradicted method 4, where age lowers the
score on all three models. After the correct control, 6b says the same thing:
age costs the person the choice.

**It agrees with methods 1 and 3.** Method 1: age and screen reader give fewer
favourable words. Method 3: the screen reader lowers accuracy on the same task,
17 measurements of 18. Method 6b: both cost the person the choice in a direct
comparison. Three different instruments, one sign.

**The remaining disagreement is with method 4** over Deafness and the screen
reader, where those raised the score. The model gives the person who disclosed a
higher score but picks the other candidate. Rating one person in isolation and
choosing between two give opposite answers, and that remains the central finding
of the study.

**Mistral is again the separate case**, as in methods 3a and 4.

---

## Limitations

**Most of Mistral is unusable.** 39% of reads clear the letter-mass gate and
only 18% of the paired contrast does (163 pairs of 900).

**One neutral alternative control**, not a set.

**Models of 7-8 billion parameters at four-bit quantisation.**

**Profiles are generated programmatically** and uniform in style.

**Three questions**; robustness to their wording was not checked separately.

---

## Files

`outputs/<model>/method6b_<model>.csv` - raw output, 4,200 rows each, seven
conditions.

`outputs/<model>/method6b_<model>_console.txt` - the run with CONTROL_ALT.
`outputs/<model>/method6b_<model>_console_run1.txt` - the first run without it,
kept for the record. Its conclusions are withdrawn.

`scripts/run_method6b.py` - the run and its built-in analysis. The built-in
`combined` overstates; trust `analyse_method6b.py`.

`scripts/analyse_method6b.py` - the main analysis: letter mass, raw slot bias,
three controls, the paired contrast against CONTROL_ALT. No GPU.

`scripts/validate_method6b.py` - offline check on synthetic data, three
scenarios.

`../shared/profiles/profiles.json` - 100 profiles.
