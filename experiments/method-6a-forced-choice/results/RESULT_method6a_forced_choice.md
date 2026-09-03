# Method 6a. Head to head, choice read as a letter

Shown two candidates who differ in one detail, the model picks one.

Three models, three questions, five comparisons, 100 profiles, two orders.
9,000 generations.

**The design failed on all three models. The known zero it was built around does
not hold.**

---

## Why it was expected to be the strongest

In the other methods the zero point had to be inferred from the data. Here it is
given by the structure: two profiles differing in an irrelevant detail should be
picked equally often. Fifty-fifty comes from the design, so no control is needed
to locate the noise floor.

Position bias is removed by the swap. Every pair is run twice with the
candidates exchanged.

- The signal wins in both orders, so it is a real preference.
- The first slot wins in both orders, so it is position, not content.

Both premises are false on these models.

---

## What should have been zero and was not

### The swap does not remove position bias

Share of times candidate A is chosen:

| Model | P(chose A) | First-slot win rate |
|---|---|---|
| Qwen | 67.3% | 67.3% |
| Mistral | 91.7% | 91.7% |
| Llama | 98.9% | 98.9% |

Llama almost always takes whoever is printed first. Mistral does the same, plus
14.8% unreadable answers. On Qwen the bias is weaker, but 67 against an expected
50 is not noise.

The swap does not rescue it because the effect is at a ceiling. When the changed
candidate is printed first it wins nearly 100% of the time on all three models.
The average of the two orders is an average of a ceiling and something else, and
has no defined meaning.

First-slot win rate by signal:

| | CONTROL | SCREEN | AGE | DEAF | ADHD |
|---|---|---|---|---|---|
| Qwen | 52.0 | 72.5 | 59.7 | 73.2 | 79.3 |
| Llama | 95.0 | 100 | 100 | 99.5 | 100 |
| Mistral | 98.6 | 75.2 | 100 | 83.1 | 99.6 |

### The control is not neutral

The control comparison is "commutes from a nearby town" against "cycles to work
most days". Two equally irrelevant details; both should be picked equally often.

Paired within profile, three questions pooled, 300 pairs per cell:

| | won both | lost both | split |
|---|---|---|---|
| Qwen CONTROL | 288 | 0 | 12 |
| Llama CONTROL | 30 | 0 | 270 |
| Mistral CONTROL | 7 | 0 | 189 |

Qwen picks the cycling candidate in **both** positions 96% of the time. That is
not a neutral detail, it is a strong content preference. There is no 50/50 zero
point, so there is nothing to compare against.

On Llama the control is 90% split: position decides everything. On Mistral the
control is barely readable (18.5% refusals) and otherwise split.

### "Lost in both positions" never happens

Zero in all 45 model-by-signal-by-question cells. The changed candidate was
never rejected in both orders.

The outcome the method exists to detect does not occur. Whatever these models
are doing, it is not "treat the person who disclosed worse in a direct choice".

---

## What survives, with caveats

### Mistral: screen reader and Deafness raise the choice

| Mistral | won both | lost both | split | vs CONTROL |
|---|---|---|---|---|
| CONTROL | 7 | 0 | 189 | reference |
| SCREEN | 128 | 0 | 93 | +71, -0 of 71 discordant, BH<0.0001 |
| DEAF | 69 | 0 | 78 | +32, -0 of 32, BH<0.0001 |
| AGE | 0 | 0 | 297 | +0, -7, BH=0.02 |
| ADHD | 2 | 0 | 206 | +1, -6, BH=0.13, not significant |

The candidate with a screen reader or Deafness wins both positions **more often**
than the control candidate, robustly to order. The direction matches method 4,
where Deafness and screen reader raised the score.

But Mistral's refusals depend on the signal: DEAF 26.2%, CONTROL 18.5%, ADHD
15.7%, SCREEN 13.3%, AGE 0.5%. A separate finding of the same kind as in
method 4: the model answers about age almost always, and refuses about
disability in a quarter of cases.

### Qwen: any signal eats the advantage the control detail is given

| Qwen | won both | split | vs CONTROL |
|---|---|---|---|
| CONTROL | 288 | 12 | reference |
| AGE | 242 | 58 | +9, -55, BH<0.0001 |
| SCREEN | 165 | 135 | +1, -124, BH<0.0001 |
| DEAF | 161 | 139 | +7, -134, BH<0.0001 |
| ADHD | 124 | 176 | +0, -164, BH<0.0001 |

The control candidate wins both positions 96% of the time. Any signal lowers
that advantage and the outcome moves into split, meaning it is decided by
position. Significant for all four signals, one direction.

Two readings, and the data do not separate them:

1. Disclosing the detail deprives the candidate of a near-automatic choice, so
   it costs them the preference.
2. The signal simply damps an overvaluation of the extra clause, and the choice
   reverts to the positional default.

"Lost both" is zero here too: the candidate is not rejected, they lose a head
start.

### Llama: nothing

The first slot wins 99% of the time under every signal, including the control.
The instrument reads nothing but the slot.

---

## What can be claimed

**Method 6a on 7-8B models does not measure what it was built to measure.**
Position bias is close to total (first-slot win rate 67, 92, 99 percent), the
swap does not remove it because of the ceiling, and the control is not neutral.

**Swapping the order is not a universal remedy for position bias.** It works
only if the effect is far from a ceiling and additive on the scale of the
choice. Here neither holds.

**On Mistral, disclosing a screen reader or Deafness raises the chance of being
chosen**, robustly to order, significantly. The direction matches method 4.

**Mistral's refusals depend on the signal:** Deafness 26%, age 0.5%.

---

## What cannot be claimed

**That any signal costs the person the choice.** The outcome "lost in both
positions" never occurs.

**That the method has a working zero point.** The control is 96/0/4 on Qwen,
decided by position on Llama, and unreadable on Mistral.

**That Qwen's shifts against the control mean "treats worse".** They are equally
consistent with the model merely ceasing to overvalue an extra clause.

**That the numbers are comparable across models.** Llama is at a ceiling, Qwen
has a moderate bias, Mistral has a ceiling plus refusals.

---

## How this relates to the other methods

Method 6a joins the negative results rather than the positive ones.

With method 3a (Mistral showed nothing) and method 4 (the coherence check failed
on Qwen), this is the third case of a design that is strong on paper and gives
no readable signal in practice.

The Mistral direction (screen reader and Deafness up) matches method 4 and
remains in conflict with methods 1 and 3, where the same signals gave fewer
favourable words and fewer solved tasks. That conflict between methods is itself
the finding, and method 6a does not change it.

---

## The rebuild: method 6b

Implemented in `../method-6b-logprob-choice/`. Three changes:

1. **By log probability.** Do not parse a letter; read logP("A") and logP("B")
   at the answer position. The difference in logits is continuous and does not
   hit a ceiling.
2. **Average the two orders in logit space.** `combined = (m_A + m_B) / 2`
   removes a constant additive slot bias exactly; `position = (m_A - m_B) / 2`
   is the direct measurement of position bias that the old control was meant to
   give.
3. **A control that is zero by construction.** `CONTROL_ID`: the same detail on
   both candidates, identical text, only the A/B labels differ. Plus
   `CONTROL_PARA`, a paraphrase of the reference, as a realistic noise floor.

What cannot be fixed: if a 7-8B model still puts almost all its mass on the
first letter even at the logit level, 6b hits the same wall. Method 6b's result
document reports what actually happened, including a third control that inverted
its own first reading.

---

## Limitations

**Models of 7-8 billion parameters at four-bit quantisation.**

**Greedy decoding.** One pass, no sampling.

**Profiles are generated programmatically** and uniform in style.

**One run per model.** Robustness to the wording of the question was not
checked.

**Three questions are pooled** in the paired analysis. The per-question
breakdown is in the run's console output.

---

## Files

`outputs/<model>/method6a_<model>.csv` - raw output, 3,000 rows each.

`outputs/<model>/method6a_<model>_console.txt` - the run log; the built-in
analysis follows the `====` separators. That built-in position-bias check
understates the bias on Qwen, for the reason given above.

`scripts/run_method6a.py` - the run and its built-in analysis.

`scripts/analyse_method6a.py` - the corrected analysis: first-slot win rate by
order, paired won/lost/split classification, sign test against the control.
No GPU.

`scripts/validate_method6a.py` - offline check of the parser and the statistics.

`../../shared/profiles/profiles.json` - 100 profiles.
