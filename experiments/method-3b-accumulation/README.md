# method-3b: accumulation, depth 0 to 3

## Question
Does the accuracy penalty grow when a person carries several signals at once?
Additive, saturating, or compounding. Eight routes add signals one at a time to
give four points: 0, 1, 2, 3. Run twice, with two neutral-filler lengths,
because the initial 40-word neutral preamble materially changed Llama's task
accuracy: its control landed 6 percentage points above the baseline (paired net
-12, p=0.008), so the baseline was not a stable reference.

## Inputs
- ../shared/tasks/tasks.json  (first 200)

## Scripts
- scripts/run_method3_stack.py — canonical, 26-word filler, one run per model
- scripts/run_method3_stack_40w.py — the first pass, 40-word filler

## Outputs
`outputs/<model>/method3_stack_<model>_40w.csv` + `_console.txt` (preliminary, excluded from the primary read)
`outputs/<model>/method3_stack_<model>_26w.csv` + `_console.txt` (second pass)
qwen and llama only.

## Status
qwen done, llama done. 20 800 generations across the four runs.

## Result
results/RESULT_method3b_accumulation.md — finished.
No general accumulation was detected. On qwen the mean net loss rises
descriptively with depth (+2.4, +2.8, +4.6) but a paired Wilcoxon on D3 - D1
gives p=0.0625. On llama the curve falls (+8.1, +3.2, -1.1). Route and order
heterogeneity is part of the result: on llama the effect depends on which
signals arrive first, not only on how many there are.
