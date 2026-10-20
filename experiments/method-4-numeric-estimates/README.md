# method-4: numeric estimates about a person

## Question
Does the model name a different number about a person when one detail changes?
Ten measures, five descriptions, 100 profiles. Paired within-profile, not
pooled. Anchoring (a number copied from the prompt) and refusals are counted
separately, never averaged into the bias estimate.

## Inputs
- ../shared/profiles/profiles.json  (profiles are also embedded in the runner)

## Scripts
- scripts/run_method4.py — collect + analyse, one run per model (set MODEL_KEY)

## Pilot
pilot/pilot_method4.py — 20 profiles, 5 measures. pilot/method4_pilot_qwen.csv,
pilot/pilot_method4_console.txt. What it taught is written into the runner docstring.

## Outputs
`outputs/<model>/method4_<model>.csv` + `_console.txt` (5,000 rows each)

## Status
qwen done, llama done, mistral done.

## The models are not equally usable here

| model | technical output | scientific usability |
|---|---|---|
| Qwen | complete, no refusals | usable, but the coherence check fails on all four descriptions |
| Llama | condition-dependent refusals up to 6.2% | compromised |
| Mistral | 19-28% refusals, 22-37% unreadable | two of ten measures had enough data |

Read any "all three models" claim against this table.

## Result
results/RESULT_method4.md
Age 74 lowers ratings on all three models, up to 74% on llama. Deafness and
screen reader use RAISE scores. Thirteen direct model conflicts. Coherence
check failed on all four descriptions for qwen.

## The second neutral reference, added 2026-10-20

The original run compared every disclosure against one clause, "commutes from a
nearby town". Method 6b later showed that clause is not neutral to these
models: against "cycles to work most days" it loses in 96 percent of profiles
on Qwen. So the rating result has never been read against a second neutral
detail, and HDSR's screening raised exactly that on 2026-10-19.

    scripts/run_method4_control.py        collect, GPU, one new condition
    scripts/analyse_method4_control.py    both references side by side, no GPU
    scripts/validate_method4_control.py   offline check of the analysis

`run_method4_control.py` is `run_method4.py` with one change, the signal table.
Profiles, measures, prompt and decoding are copied unchanged, so the new
condition is comparable to the old ones by construction. 10 measures x 100
profiles = 1,000 generations per model.

Outputs land next to the originals as `method4_control_<model>.csv`.
