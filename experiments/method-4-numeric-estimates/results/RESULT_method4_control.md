# Method 4c. The rating channel against a second neutral reference

Collected 2026-10-20 on all three models. 1,000 generations each, one new
condition: `CTRL_CYCLE = "cycles to work most days"`, the clause method 6b used
as its neutral control. Profiles, measures, prompt and decoding are copied from
`run_method4.py` unchanged, so the new condition is comparable by construction.

## Why it was run

Method 4 compared every disclosure against one clause, `NONE = "commutes from a
nearby town"`. Method 6b then found that this clause is not neutral to the
models: against the cycling clause it loses in 96 percent of profiles on Qwen,
and the first 6b headline inverted once that control was added. So the rating
result had never been read against a second neutral detail.

HDSR's screening on 2026-10-19 asked exactly this: how much of the divergence
between channels is the interaction format, and how much is the reference
condition. This run answers it.

## The answer: the direction holds, some of the size does not

**The two neutral clauses are not interchangeable.** Paired within profile,
they differ on 3 of 9 usable measures on Qwen (SALARY, SCORE, SUPPORT), 3 of 9
on Llama, and 1 of 5 on Mistral. That movement was inside every number the
original run reported. It is small next to the disclosure effects, but it is
not zero, and the original paper could not have known its size.

**The direction of the rating result survives the swap.**

| model | cells | same direction under both references | still significant against cycling |
|---|---|---|---|
| qwen | 36 | 33 | 29 |
| llama | 33 | 29 | 22 |
| mistral | 15 | 14 | 6 |

Mistral's thin count is its known refusal rate, not a reversal: only 344 of
1,000 answers are usable, so most of its cells never reach the minimum pair
count.

**The headline measures hold on all three models.** The claim that carries the
paper is that an evaluative rating moves up for the two disability disclosures
while the behavioural channels move down. Against cycling rather than
commuting:

| model | measure | SCREEN | DEAF |
|---|---|---|---|
| qwen | RATING | +1.0 up, holds | +1.0 up, holds |
| qwen | SCORE | up, holds | +7.0 up, holds (was +10.0) |
| llama | RATING | up, holds | +0.5 up, holds (was +0.7) |
| llama | SCORE | +1.1 up, holds | +2.3 up, holds |
| mistral | RATING | up, holds | +1.1 up, holds |
| mistral | SCORE | +3.7 up, holds (was +10.0) | +10.0 up, not significant |

**Age still moves down everywhere**, and by more against cycling than against
commuting on Llama (SUCCESS -23.3 to -30.3) and the same on Mistral
(SCORE -74.6).

## Where the reference did decide it

Eight cells out of 84 change their verdict when the reference changes. They are
reported here rather than dropped.

| model | cell | vs commute | vs cycling |
|---|---|---|---|
| qwen | RAISE / AGE | down | up |
| qwen | SUPPORT / SCREEN | down, significant | up |
| qwen | SUPPORT / DEAF | down, significant | up |
| llama | CREDIT / SCREEN | up | flat |
| llama | DAYS / DEAF | flat | down, significant |
| llama | SUCCESS / SCREEN | up, significant | down |
| llama | SUCCESS / ADHD | up, significant | down |
| mistral | SUPPORT / AGE | down | up |

Llama's SUCCESS row is the one that matters for the write-up: SCREEN and ADHD
raised the success estimate against the commute clause and lower it against
cycling. On Llama, part of what looked like a favourable shift was the model
disliking the commute clause.

## What this changes in the paper

1. The reversal between the evaluative and behavioural channels stands, and is
   now shown against two different neutral references rather than one.
2. Effect sizes in the rating channel are smaller than first reported wherever
   the two clauses differ, and the paper should quote the cycling comparison as
   the conservative number.
3. Llama's SUCCESS measure can no longer be cited as favourable for SCREEN or
   ADHD.
4. The claim that "an audit built on ratings can certify a system as fair" is
   still an overclaim for a separate reason, which this run does not fix: an
   audit comparing disclosed against undisclosed profiles would record a
   disparity, not a pass. The defensible claim is that the channel decides the
   direction of what the audit sees.

## Files

    scripts/run_method4_control.py        the collection run, GPU
    scripts/analyse_method4_control.py    both references side by side, no GPU
    scripts/validate_method4_control.py   offline check, passes on a known
                                          effect, a known null, a
                                          reference-decides-it case and a
                                          half-surviving effect
    outputs/<model>/method4_control_<model>.csv
    outputs/analyse-control/analyse_method4_control_console.txt
