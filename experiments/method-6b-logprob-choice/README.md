# method-6b: head to head, choice by log probability

Rebuild of method-6a, which failed: a near-total first-slot bias put the parsed
letter at a ceiling the order swap could not cancel.

## What is different from 6a
1. **Score by log probability.** Same prompt, nothing generated. Read logP("A")
   and logP("B") at the answer position. Continuous, no ceiling, no parsing, and
   no refusals to parse.
2. **Average the two orders in logit space.** `m1` is the margin with the
   changed candidate in slot A, `m2` with it in slot B.
3. **Three controls**, the third of which changed the answer:

| control | detail on the changed candidate | what it gives |
|---|---|---|
| `CONTROL_ID` | the same as the reference | raw first-slot bias (`\|m1\|`). Its combined margin is 0 by construction and means nothing |
| `CONTROL_PARA` | a paraphrase of the reference | what mere rewording does |
| `CONTROL_ALT` | a different but socially neutral detail | **the floor a disclosure must beat** |

## How to read the result
Run `analyse_method6b.py`, not the built-in analysis. The headline is the
**paired contrast against CONTROL_ALT** within question and profile: both are
measured against the same reference clause and in the same slot, so the slot
bias and the reference clause both cancel. It is the only comparison that
separates "this is a disclosure" from "this is not a commute".

Do not read the built-in `combined` as an effect: the slot bias is not additive
(the position term ranges 1.7 to 10.2 logits across signals on Qwen), so
`combined` still carries some of it.

Check the **letter mass** line first. If `P(A)+P(B)` is small the margin is a
ratio of two tail probabilities; cells with fewer than 20 clean pairs are
reported NOT REPORTABLE.

## Scripts
- scripts/run_method6b.py — collect + a built-in analysis, one run per model
- scripts/analyse_method6b.py — the read to trust, no GPU
- scripts/validate_method6b.py — offline check, three scenarios, no GPU

## Inputs to upload into Colab
`run_method6b.py`, `profiles.json` (from `shared/profiles/`), and the existing
`method6b_<model>.csv` if resuming. `collect()` is keyed on the signal, so
adding a condition appends only the new rows.

## Outputs
`outputs/<model>/method6b_<model>.csv` — 4200 rows, seven conditions
(3 questions x 7 comparisons x 100 profiles x 2 orders).
`..._console.txt` is the run with CONTROL_ALT; `..._console_run1.txt` is the
first run without it, kept for the record. Its conclusions are withdrawn.

## Status
Collected and analysed on all three models, CONTROL_ALT rerun complete.

**Disclosure costs the candidate the choice**, against a socially neutral
alternative detail about the same person:

| Δ vs CONTROL_ALT (logits) | Qwen | Llama | Mistral |
|---|---|---|---|
| screen reader | −5.99 | −1.62 | **+7.14** (helps) |
| age 74 | −3.47 | −1.46 | −1.83 |
| Deaf | −5.87 | −0.58 | withdrawn, 10 pairs |
| ADHD | −6.93 | −1.23 | +0.23 ns |

All Qwen and Llama figures significant after Benjamini-Hochberg. Mistral is
mostly unreadable: 39% of its reads clear the letter-mass gate (61% carry too
little mass on the letters), and because the paired contrast needs all four
reads clean, only 18% of it survives (163 pairs of 900). The PROMOTE question
yields no clean pair at all.

The first run said the opposite — that disclosure was preferred — because it had
no neutral-alternative control. Withdrawn.

## Result
results/RESULT_method6b.md
