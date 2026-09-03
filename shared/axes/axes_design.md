# Axes of deviation: how the catalogue is built

Twenty-eight axes. The data are in `axes.csv` and `axes.json`; this file
explains why they are organised the way they are.

This is a design artefact, not an experimental input. No run script reads it.
It is the catalogue the signals actually tested were drawn from, and it records
what was left on the table.

---

## Why not organised by group

The obvious way is a list of categories: disability, language, culture, age,
gender.

That reproduces exactly the thing being argued against, that a person is best
understood by which box they fall into.

It also does not work in practice. Someone making typos could be anyone: a motor
impairment, a phone keyboard, a hurry, dyslexia. There is no category there, but
there is a deviation.

So the catalogue is organised **by type of deviation**, not by type of person.

---

## Four types

**FORM, ten axes.** How the person writes. Nothing is stated to the model; it
infers from the shape of the text.

**STATED, ten axes.** What the person says about themselves in one sentence. The
task does not change.

**CONDUCT, six axes.** How the person runs the exchange. Asks again, unfolds a
thought gradually, changes the question, seeks reassurance.

**KNOWING, two axes.** What the person does not know. Lacks the terminology, or
uses the wrong terminology.

---

## The main distinction: implicit against explicit

Eighteen axes are implicit, ten are explicit. This is not presentation, it is a
separate variable.

**Under an explicit signal** the person says "I am autistic". The model sees
that as a fact, and it has a trained response to the fact. What is measured is a
**trained reaction to a word**.

**Under an implicit signal** there is nothing to conceal, because the model does
not register that it is being examined. It is simply processing text of an
unfamiliar shape.

Hence the difference in what is being measured. The explicit signal measures the
**attitude to a category**. The implicit signal measures the attitude to
unlikeness as such: not to a group, but to departure from a familiar form.

And in practice: in real life a person does not write "I am autistic" into a
chat window. They just write the way they write. The implicit signal is what
actually happens.

The published literature disagrees about what naming a trait explicitly does.
That disagreement is what makes this axis worth carrying.

---

## Two pairs where the same thing is presented two ways

**F01 against S09.** Non-native English: shown by the form of the writing, or
declared in a sentence.

**F04 against S07.** Imprecise typing: shown by typos, or declared as a motor
impairment.

These two pairs allow the implicit-against-explicit question to be tested
directly, on the same material. Method 3a runs exactly that comparison and finds
that the explicit form costs accuracy while the implicit form does not.

---

## What can be measured with what

Twenty-four axes work inside a single message and are usable by every method.

Four axes, C01 to C04, need several turns of conversation and are usable only by
the answer-quality method and by an interaction-cost method.

That constraint is worth holding in mind when planning: the behavioural axes are
the most expensive of all, because each one needs a dialogue rather than a
single request.

---

## What is deliberately out of scope

**Gender and race through a name.** Densely studied territory where it is easy
to drown in other people's framing. But the speech forms associated with them
are included; that is F09.

**Religion.**

**Income and access to paid tools.** Hard to express in one sentence without
changing the task itself.

The catalogue is built around a person the system fits worse, not around
protected characteristics as such.

---

## How it is used

**First round.** One axis at a time against the baseline version. Twenty-eight
measurements per method. This is what the present study does.

**Second round.** Two axes at once. For example F01 plus S01: writes as a
non-native speaker and mentions a screen reader.

**Beyond that.** Three or more, along fixed routes rather than all combinations.
There are too many combinations to take them all, and a random selection gives
mush.

The question all of this is for: **how the degradation changes as axes are
added.** Additive, saturating, or accelerating. Three answers give three
different pictures and three different conclusions. The accumulation experiment
in this repository is the first attempt at it.

---

## Fields in the data

| column | meaning |
|---|---|
| `id` | F01-F10, S01-S10, C01-C06, K01-K02 |
| `group` | FORM, STATED, CONDUCT, KNOWING |
| `name` | short name of the axis |
| `signal` | implicit or explicit |
| `turns` | single, or several |
| `methods` | which measurement methods the axis is usable by |
| `example` | a sample phrasing |
| `note` | what the axis is and is not |

Rebuild with `python build_axes.py`, which is deterministic and overwrites both
files.
