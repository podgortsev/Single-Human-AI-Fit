# What this study found, in plain language

## The question

You ask an AI assistant for help. Somewhere in your message you mention that you
use a screen reader, or that you have ADHD, or that you are seventy-four.

Does the answer get worse?

Not "does the model say something offensive". Something narrower and easier to
check: does the model **do a worse job** for you than for someone who wrote the
identical message without that detail.

## How you can actually test that

The trick is to change one thing and nothing else.

Take an arithmetic question with a definite right answer. Ask it two hundred
times, in two versions. The two versions are identical to the character, except
one opens with "I use a screen reader" and the other does not. Then count how
often each version gets the right answer.

If the model is doing the same job for both people, the counts should match.
Any gap is caused by the one clause that differs, because nothing else does.

Then do the same trick in six other ways: with trait words, with open-ended
questions, with a numeric score, with a head-to-head choice between two
candidates, and by asking the model afterwards what it thought it had done.

Three different models, from three different companies. The same hundred
people, the same two hundred tasks, throughout.

## What came out

**Four ways of measuring say the person is treated worse.**

The model applies fewer positive words to them. It gets their arithmetic wrong
more often. When there is no right answer, it writes them a weaker reply. And
when it has to choose between two otherwise identical candidates, it picks the
other one.

**One way of measuring says the opposite.**

When you ask the model to rate the person out of ten, or to score their
suitability, mentioning Deafness or a screen reader makes the score go **up**.
Consistently, on all three models, significantly.

## Why that contradiction is the interesting part

The model gives the person a better score and does a worse job for them.

Both things are happening at once, and they are not in tension from the model's
point of view. Praise is cheap. Solving the arithmetic is not.

This matters well beyond these three models, because **the score is what most
audits measure.** It is the easy thing to measure. If you test an AI system by
asking it to rate people and check the ratings look fair, you can pass that test
while the system quietly does worse work for the same people. This study found
exactly that pattern, on purpose, by measuring both.

## Two things we expected and did not find

Good studies report the things that did not work out. Two here are worth as much
as anything above.

**Signals do not stack.** We assumed that mentioning three things about yourself
would cost more than mentioning one. It does not. Three cost about the same as
one, and on one model they cost *less*.

**Answers are not less stable.** We assumed the model would be more erratic with
people who had disclosed something. Of thirty-three tests, seven showed less
stability, ten showed **more**, and sixteen showed nothing. The hypothesis was
simply wrong.

## The model does not know it is doing this

We replayed each exchange back to the model and asked: did my mentioning a
screen reader change your answer, nought to ten?

One model said "yes, about eight out of ten" for the screen reader, against
close to zero when nothing had been disclosed. It is not hiding anything. If
anything it over-claims, though the self-rating and the measured change are on
different scales and cannot be divided into one another.

But when we checked **which specific questions** it claimed to have been
affected on, against the questions where its answer really did change, there was
no relationship. None, in any of nine tests.

The model knows the disclosure was there. It does not know what the disclosure
did to its answer. Asking an AI system to self-report its own bias will not find
this.

## How much to trust it

Honestly, with some care.

**What is solid.** The direction. The screen reader shows up in every method
tested, and survives being re-measured under six different neutral phrasings.
The design compares like with like, item by item, rather than comparing
averages.

**What is not.** The size of any effect. One measurement of quality varies
five-fold depending on which model does the judging. Another varies by tens of
percent between models on the same question. We can say which way things move.
We cannot tell you by how much.

**Where it does not reach.** Three open models of moderate size, run at reduced
precision, on arithmetic and general-knowledge questions. Not the large
commercial systems most people actually use. Not medical or legal advice. Not a
statement about AI in general.

**One model barely participated.** Mistral showed almost nothing on the task
measure, refused roughly a fifth to nearly a third of the scoring questions,
and in the head-to-head test 82 percent of its data was unusable because it was
not really answering in the format we were reading. We kept it in and reported
that, rather than dropping it for being inconvenient.

## Two mistakes we made first

Both changed a conclusion, and both are in the repository in full.

**The head-to-head test failed the first time.** Show a model two candidates and
ask it to pick one, and it mostly picks whichever came first, up to 99 percent
of the time. Swapping the order was supposed to cancel that out. It did not. We
reported the failure instead of the finding.

**The rebuild got the answer backwards.** The second version concluded that the
model *preferred* candidates who disclosed something. Then we added a control:
a candidate with a different but socially meaningless detail, cycling to work.
It turned out the model liked *any* distinctive detail more than the bland
commute clause we had been comparing against. Measured against the right
control, every disclosure lost.

One missing control had inverted the headline. That is why the controls are
described at length in the method documents.

## Where everything is

Every number regenerates from the raw output files, which are all committed.
The analysis scripts need no GPU.

- [`RESULTS.md`](RESULTS.md), the full argument across all seven methods
- [`limitations.md`](limitations.md), what cannot be claimed, and why
- `experiments/*/results/`, one detailed write-up per method
