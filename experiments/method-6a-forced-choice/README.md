# method-6a: head to head, choice by letter

## Question
Shown two people who differ in one irrelevant detail, which does the model
pick? Fifty-fifty was meant to be given by the structure. Every pair is run
twice with the candidates swapped to cancel position bias. The answer is parsed
as a letter (A or B).

## Scripts
- scripts/run_method6a.py — collect + a built-in analysis, one run per model (set MODEL_KEY)
- scripts/analyse_method6a.py — corrected read, no GPU: first-slot win rate by
  order, paired won/lost/split classification, sign test vs control. The
  analyse() inside run_method6a.py undercounts position bias on qwen; use this.
- scripts/validate_method6a.py — offline check, no GPU: parser table (28/28) +
  analyse() on synthetic data with known win rates and a known slot bias

## Inputs to upload into Colab
- run_method6a.py
- profiles.json  (from ../shared/profiles/)

## Outputs
`outputs/<model>/method6a_<model>.csv` + `_console.txt`, 3,000 rows each
(3 questions x 5 comparisons x 100 profiles x 2 orders).

## Status
Collected and analysed on all three models. **The design failed.**

- First-slot win rate: qwen 67%, llama 99%, mistral 92%. The order swap does
  not cancel it: the choice is at a ceiling (~100% when the changed candidate
  is printed first), so averaging the two orders has no defined meaning.
- The control detail ("cycles to work" vs "commutes from a nearby town") is not
  neutral. Qwen picks the control candidate in both slots 96% of the time, so
  there is no measured 50/50 floor.
- "Lost both orders" never happens (0 of 45 model x signal x question cells).
- Survives weakly: on mistral, screen reader and Deaf candidates win both slots
  more often than the control (sign test BH<0.0001); mistral refusals are
  signal-dependent (deaf 26%, age 0.5%). Unparsable: 0 / 0 / 14.8%.

## Result
results/RESULT_method6a_forced_choice.md — negative result, with the numbers.
The rebuild is method-6b-logprob-choice (score A/B by log probability, control
identical on both sides).
