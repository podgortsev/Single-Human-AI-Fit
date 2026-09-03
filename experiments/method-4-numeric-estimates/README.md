# method-4: numeric estimates about a person

## Question
Does the model name a different number about a person when one detail changes?
Ten measures, five descriptions, 100 profiles. Paired within-profile, not
pooled. Anchoring (a number copied from the prompt) and refusals are counted
separately, never averaged into the bias estimate.

## Inputs
- shared/profiles/profiles.json  (profiles are also embedded in the runner)

## Scripts
- scripts/run_method4.py — collect + analyse, one run per model (set MODEL_KEY)

## Pilot
pilot/pilot_method4.py — 20 profiles, 5 measures. pilot/method4_pilot_qwen.csv,
pilot/pilot_method4_console.txt. What it taught is written into the runner docstring.

## Outputs
outputs/<model>/method4_<model>.csv + _console.txt   (5000 rows each;
qwen has two console logs, collect then resume)

## Status
qwen done, llama done, mistral done.

## Result
results/RESULT_method4.md — finished.
Age 74 lowers ratings on all three models, up to 74% on llama. Deafness and
screen reader use RAISE scores. Thirteen direct model conflicts. Coherence
check failed on all four descriptions for qwen.
