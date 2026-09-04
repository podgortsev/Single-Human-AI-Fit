# method-2: answer quality

## Question
Methods 3 and 6 asked whether the model gets the answer right, or picks the
person. This asks whether the answer the person **gets** is worse, when no
answer is wrong. Sixty open-ended questions, question text byte-identical
across conditions, only a six-word opening sentence changes.

**This is the only subjective method.** Quality cannot be checked by string
comparison, so a model has to judge. Report it separately from methods 1, 3, 4
and 6, and label it.

## Two stages, four Colab runs minimum

| stage | script | runs | cost each |
|---|---|---|---|
| 1. generate answers | `run_method2_generate.py` | **once per model (3)** | 300 generations, 25-40 min |
| 2. judge them, blind | `run_method2_judge.py` | **once per judge (1+)** | 1800 forward passes, 10-15 min |

Stage 2 reads all three answer files in one pass, so one judge run covers every
answer model. A second judge (set `JUDGE_KEY = "llama"`) is a self-preference
check, not a requirement.

## Conditions
`NONE` (bare question), `CONTROL` (signal-free lead-in), `SCREEN`, `AGE`, `ADHD`.
All four lead-ins are **exactly six words** — the script asserts it. The three
signals are the ones method 3c already characterised: screen reader and age
survived six neutral wrappers, ADHD changed direction, and it is kept because
that instability is itself worth retesting.

## How the judge is kept honest
- **Blind.** The judge sees the bare question and two answers, never the
  lead-in. It cannot tell which answer came from someone who disclosed
  something, so its own bias about screen readers cannot reach the score. This
  is the property the method rests on; `validate_method2.py` tests it directly.
- **Paired, not scored.** A mark out of ten reproduces method 4's sticky scale.
  Two answers from the same model on the same question have a known 50/50 null.
- **Both orders**, averaged in logit space, so an additive slot bias cancels.
  The script reports whether the bias is in fact additive.
- **Read by log probability**, not a parsed letter (method 6a died on that), and
  the mass on the two letters is recorded per row so a judge that was never
  going to answer with a letter is caught (method 6b found one).
- **`IDENTITY`** compares the CONTROL answer with itself: the judge's raw slot
  bias, measured not assumed. Its combined margin is 0 by construction.
- **`NONE_vs_CONTROL`** compares two genuinely signal-free answers: the floor a
  signal must clear.
- **Length is a covariate, not the outcome.** The `len gap` column is the word
  count difference. A significant result with a large length gap is about
  verbosity, not quality.

## Inputs to upload into Colab
Stage 1: `run_method2_generate.py`, `questions.json` (from `shared/questions/`).
Stage 2: `run_method2_judge.py`, `questions.json`, and the three
`method2_answers_<model>.csv` from stage 1.

## Outputs
```
outputs/<model>/method2_answers_<model>.csv      stage 1, 300 rows
outputs/judge-<judge>/method2_judged_by_<judge>.csv   stage 2, 1800 rows
```

## Analysis
```
python scripts/analyse_method2.py outputs/judge-*/method2_judged_by_*.csv
```
Read this, not the per-judge analysis built into `run_method2_judge.py`. The
built-in one computes the additivity residual across all comparisons including
IDENTITY, where the two answers are identical and the judge has nothing but
position to go on; that inflates the residual fivefold and produces a spurious
"NOT additive" warning.

## Offline check
```
python scripts/validate_method2.py      # 25 checks, no GPU
```
Covers lead-in length matching, judge blinding, recovery of a known effect,
rejection of a known null, cancellation of an injected slot bias, the
letter-mass gate, and exclusion of an unusable judge from the cross-judge read.

## Status
Done. All three models generated, and all three run as judges rather than one.

- **judge mistral EXCLUDED** by a validity gate on whether it produced a reading
  at all: letter mass 0.16, 94% of reads below threshold. Third appearance of
  the same refusal (6a 14.8%, 6b 61%, here 94%). Not pre-registered; see
  `scripts/sensitivity_min_mass.py` for how little the observation-level
  threshold matters.
- Judge slot bias on identical answers: qwen 9.86, llama 5.71 logits. Additivity
  residual +/-0.72 and +/-0.50; an effect must clear it.

Paired within question against the signal-free floor, 18 cells
(2 usable judges x 3 answer models x 3 signals):

"worse" means significant after Benjamini-Hochberg **and** larger than the
additivity residual, which is our own robustness threshold rather than a
standard test. "inside residual" is significant but within measurement slop.

| signal | worse | inside residual | no effect | better |
|---|---|---|---|---|
| screen reader | **6** | 0 | 0 | **0** |
| age 74 | 3 | 2 | 1 | **0** |
| ADHD | 3 | 3 | 0 | **0** |

Zero cells in the positive direction. Answer length is flat across conditions
(within 4 words of ~245), so this is not verbosity. Effect size depends on the
judge fivefold, so logits are not comparable across judges.

## Result
results/RESULT_method2.md
