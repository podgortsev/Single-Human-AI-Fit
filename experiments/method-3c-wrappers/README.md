# method-3c: six neutral wrappers

## Question
Is the measured effect a property of the signal, or of the neutral text around
it? Four signal conditions crossed with six neutral wrappers, fully crossed,
each wrapper with its own baseline. Six independent estimates of the same
quantity: if they agree, the effect belongs to the signal.

## Inputs
- shared/tasks/tasks.json  (first 200)

## Scripts
- scripts/run_method3_wrapper.py — collect, one run per model (set MODEL_KEY)
- scripts/analyse_method3_wrapper.py — corrected analysis, no GPU, runs on the CSVs

## Outputs
`outputs/<model>/method3_wrapper_<model>.csv` + `_console.txt`
outputs/analyse-wrapper/analyse_method3_wrapper_console.txt

## Status
qwen done, llama done, mistral done. Analysed.

## Result
results/RESULT_method3c_wrappers.md

Screen reader +8.0 net tasks [5.0, 11.0] on qwen and +20.7 [15.2, 26.1] on
llama, both 6/6 wrappers positive; mistral +2.7 with a CI including zero. Across
all three models, screen reader 17/18 measurements positive (sign test
p=0.00014) and age 16/18 (p=0.00027). ADHD changes sign between wrappers and
does not survive. Baseline drift from the choice of wrapper alone is 2 to 5
accuracy points, which is the floor for the whole study.
