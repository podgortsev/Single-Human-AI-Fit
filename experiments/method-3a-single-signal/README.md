# method-3a: single signal, 26 conditions

## Question
Does the model solve the SAME task less often depending on who appears to be
asking? Task text is byte-identical in every condition; only a short opening
sentence changes. Paired analysis (exact McNemar), read against a no-signal
control that carries only different wording.

## Inputs
- shared/tasks/tasks.json  (first 200)

## Scripts
- scripts/run_method3_single.py — collect + analyse, one run per model (MODEL_KEY)

## Outputs
outputs/<model>/method3_single_<model>.csv + _console.txt   (5200 rows each)

## Status
qwen done, llama done, mistral done.

## Result
results/RESULT_method3a_single.md — finished.
Stated signals cost accuracy (qwen p=0.0005); on llama 5 conditions survive
correction; mistral shows nothing, a negative case. Writing in a different
register does not cost accuracy.
