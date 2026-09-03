# Method 3a. One signal at a time

Does the model solve the same task less often depending on who appears to be
asking?

Three models, 26 conditions, 200 tasks, 15,600 generations.

---

## Design

**The task text is byte-identical** in every condition. Only a short opening
sentence changes.

Two hundred tasks with a checkable answer: percentages, multi-step arithmetic,
unit conversion, dates, table lookup, logic, speed. The key is computed by the
same code that writes the question, so the key is correct by construction.

**Twenty-six conditions.** A baseline, a control with no signal, ten ways of
writing, ten statements about oneself, four behavioural, and unfamiliarity with
terminology.

**Three models:** Qwen2.5-7B, Llama-3.1-8B, Mistral-7B-v0.3. Different
developers, different countries, different training data.

**Paired test.** Not a comparison of average accuracy but a count of which tasks
were lost. Only discordant pairs carry information: tasks the baseline solved
and the condition did not, and the reverse. The rest are silent.

**Control.** A condition with different wording and no signal at all. Whatever
it shows is the cost of rephrasing on its own. A condition has to clearly exceed
it to mean anything.

**Multiplicity.** Twenty-four comparisons, threshold 0.0021.

---

## Numbers

| | Qwen | Llama | Mistral |
|---|---|---|---|
| Baseline accuracy | 60.0% | 44.0% | 50.0% |
| Control, net loss | +4 | 0 | -4 |
| Way of writing, mean | +2.1 | +5.5 | -0.8 |
| Stated about self, mean | +8.2 | +15.1 | -0.9 |
| Difference between the two | p=0.0005 | p=0.0014 | none |

Pooled: form +2.3, self-statement +7.5, p=0.004.

---

## The main result

**Telling the model about yourself costs accuracy. Writing differently does
not.**

A person who mentions ADHD, a screen reader, their age, or that this is their
first time here solves the same task less often. A person writing in a dialect,
with typos, or informally does not.

**Survived correction on Llama, five conditions:**

ADHD +25, wrong terminology +25, screen reader +24, age 74 +22, first time here
+12.

**Positive on all three models at once, five conditions:**

wrong terminology, dyslexia, non-native language stated, Deafness, first time
here.

---

## Mistral shows nothing

Every condition sits within noise, and the control is -4.

This is not a failed run. It is the third case of the same thing. The model
appears simply not to attend to the opening sentence.

So the effect is not universal. It is a property of particular models, not of
language models in general.

---

## Against the published literature

Existing work disagrees about whether naming a trait explicitly helps or makes
matters worse.

These data test that directly, on one body of material, on two pairs of
conditions where the same circumstance is presented two ways.

**Explicit mention hurts; the implicit form does not.**

---

## Practical consequence

People are taught to state their needs in order to get better service.

Here the opposite happens: stating something about yourself lowers the chance of
getting the right answer to the same task.

---

## Limitations

**The signal lives only in the opening sentence.** In real use the whole message
is in one register, and the effect of writing style may be underestimated for
exactly that reason. The rigour of the design was bought with sensitivity.

**The tasks are arithmetic.** On tasks with no single right answer the picture
may differ. That is method 2.

**Models of 7-8 billion parameters at four-bit quantisation.**

**Two hundred tasks.** Between eight and forty-eight discordant pairs per
condition. Weak effects are not resolvable at this size.

**Two conditions, F05 and F06, are not comparable with the rest**, because there
the length of the opening sentence is itself the property being measured.

**Measured at one wrapper.** Method 3c repeats this across six neutral wrappers
and shows how much of it survives. Read the two together.

---

## Files

`outputs/<model>/method3_single_<model>.csv` - raw output, one row per
generation, 5,200 rows each.

`../../shared/tasks/tasks.json` - 200 tasks with keys.

`scripts/run_method3_single.py` - the run and its built-in analysis.
