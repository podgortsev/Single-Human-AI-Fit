# method-1: traits by log probability

## Question
What does the model think of a person, measured by its own probabilities
rather than by anything it says? Given "This person is", how readily does it
continue with "intelligent" against "stupid", across five descriptions of the
same 100 profiles.

## Inputs
- shared/profiles/profiles.json

## Scripts
- scripts/run_method1.py — collect + analyse, one run per model (set MODEL_KEY)
- scripts/analyse_method1_positive.py — reanalysis, positive traits only, no GPU

## Outputs
- outputs/<model>/method1_<model>.csv + _console.txt   (30 000 rows each)
- outputs/analyse-positive/analyse_method1_positive_console.txt

## Status
qwen, llama, mistral: collected and analysed.

## Result
results/RESULT_method1.md — finished.
Age 74 makes all three models less ready to apply favourable words
(qwen +2.00, llama +2.92, mistral +0.95). Screen reader the same, weaker.
Negative traits sit near the floor; differences there are noise.
