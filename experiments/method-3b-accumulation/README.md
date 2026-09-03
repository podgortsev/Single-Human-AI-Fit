# method-3b: accumulation, depth 0 to 3

## Question
Does the accuracy penalty grow when a person carries several signals at once?
Additive, saturating, or compounding. Eight routes add signals one at a time to
give four points: 0, 1, 2, 3. Run twice, with two neutral-filler lengths,
because 40 words of preamble cost llama about ten accuracy points on their own.

## Inputs
- shared/tasks/tasks.json  (first 200)

## Scripts
- scripts/run_method3_stack.py — canonical, 26-word filler, one run per model
- scripts/run_method3_stack_40w.py — the first pass, 40-word filler

## Outputs
outputs/<model>/method3_stack_<model>_40w.csv + _console.txt   (first attempt)
outputs/<model>/method3_stack_<model>_26w.csv + _console.txt   (canonical)
qwen and llama only.

## Status
qwen done, llama done. 20 800 generations across the four runs.

## Result
results/RESULT_method3b_accumulation.md — finished.
No accumulation. On qwen a slight rise within noise; on llama the curve falls.
Three signals cost no more than one.
