# method-8: consistency / stability

## Question
Is the model LESS stable for a person who disclosed something? Not "does it
judge them lower" (methods 1, 3, 4, 6) but "does its output scatter more when
nothing that should matter is changed".

## No new runs
Three probes, all from data already on disk:

| probe | source | varies (should not matter) | instability measure |
|---|---|---|---|
| A | method 1 | 5 prompt phrasings | SD of the trait log prob over 5 |
| B | method 3c | 6 neutral wrappers | k*(6-k) over the 6 solved/not-solved outcomes |
| C | method 6b | 3 decision questions | SD of the choice margin over 3 |

Each probe is paired within item against a signal-free baseline (A and B vs
NONE, C vs CONTROL_PARA), Wilcoxon signed-rank, Benjamini-Hochberg across the
family.

## Scripts
- scripts/analyse_method8.py — the three probes. `python scripts/analyse_method8.py`
- scripts/validate_method8.py — offline check, 10 tests, synthetic data with
  injected instability

## Status
Done. **The "non-average users get unstable answers" hypothesis does not hold
in general.** Of 33 cells, 7 "less stable", 10 "more stable", 16 "no effect".

- Probe A: disclosure NARROWS the trait-word distribution in 10 of 12 cells.
  The model applies its (lower, per method 1) scores more confidently, not less.
  Only Mistral + age is less stable (x1.35).
- Probe B: nothing. 9 of 9 cells no effect.
- Probe C: on Qwen all 4 signals make the choice more question-dependent; on
  Llama age and Deaf do. Mistral has no clean data.
- The one defensible claim: **age 74 makes the model's choice about a person
  more dependent on which decision is asked** (probe C, Qwen and Llama).

## Result
results/RESULT_method8.md
