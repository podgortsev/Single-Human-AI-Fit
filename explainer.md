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

Three different models, from three different companies.

The material is shared where the method allows it, not identical everywhere. The
same hundred invented people are used wherever a person is being judged; the
same two hundred arithmetic tasks wherever a task is being solved; the same
sixty open questions wherever there is no right answer. Each method uses the set
that fits it, so results can be compared without every method running on
everything.

## What came out

**Four outcome measures found worse performance. One found higher ratings.**

The four that measure what the model *does* all point the same way. It applies
fewer positive words to the person. It gets their arithmetic wrong more often.
When there is no right answer, it writes them a weaker reply. And when it has to
choose between two otherwise identical candidates, it picks the other one.

The one that measures what the model *says about* the person points the other
way. Ask it to rate them out of ten, or score their suitability, and mentioning
Deafness or a screen reader makes the number go **up**. Consistently, on all
three models, significantly.

Note which signals do that. **For age seventy-four everything points the same
way, ratings included.** It is the two disability disclosures where the rating
splits off from the behaviour.

## Why that contradiction is the interesting part

The model gives the person a better score and does a worse job for them. Both
are happening at once.

One reading, and we offer it as an interpretation rather than something the data
establish: praising a person costs the model nothing, while solving their
problem takes effort. The measurements cannot distinguish that story from
several others, so treat it as a way of holding the result in mind, not as a
finding.

This matters well beyond these three models, because **the score is what most
audits measure.** It is the easy thing to measure. If you test an AI system by
asking it to rate people and check the ratings look fair, you can pass that test
while the system quietly does worse work for the same people. This study found
exactly that pattern, on purpose, by measuring both.

## Something we expected and did not find

Good studies report the things that did not work out, and this one is worth as
much as anything above.

**Answers are not less stable.** We assumed the model would be more erratic with
people who had disclosed something. Of thirty-three tests, seven showed less
stability, ten showed **more**, and sixteen showed nothing. The hypothesis was
simply wrong.

## The model cannot point to what it did

We replayed each exchange back to the model and asked: did my mentioning a
screen reader change your answer, nought to ten?

One model said "yes, about eight out of ten" for the screen reader, against
close to zero when nothing had been disclosed. It is not hiding anything. If
anything it over-claims, though the self-rating and the measured change are on
different scales and cannot be divided into one another.

But when we checked **which specific questions** it claimed to have been
affected on, against the questions where its answer really did change, there was
no relationship. None, in any of nine tests.

The model can tell you the disclosure was there. It cannot tell you which of its
answers the disclosure changed. Asking an AI system to report on its own bias
will not find this.

## What this means for testing AI systems

Put the pieces together and the useful conclusion is not "these models are
biased against disabled people". It is something more awkward.

**The same fact about a person can help them in one kind of interaction and cost
them in another.** Mentioning a screen reader raises the rating the model gives
you and lowers the odds it picks you over someone else. Being seventy-four
lowers everything at once. And on one of the three models, the head-to-head
result reverses entirely.

So the effect is not a fixed property of a model that you could measure once and
put on a label. It depends on **who the person is, what they are asking for, and
which model is answering**, all three together.

Researchers have already found that bias measurements depend on the task; we are
not the first to see that. What this study adds is holding one person fixed,
changing one detail about them, and then watching seven different kinds of
interaction at once, which is what makes the reversal visible.

The practical consequence is uncomfortable for anyone who has to certify these
systems. **"Is this model fair?" is not a well-posed question.** A system can
pass a fairness test built on ratings while quietly doing worse work for exactly
the people the test was meant to protect, because ratings are the one place
where the effect runs backwards. A test has to say which kind of interaction it
covers, and its result does not transfer to a different kind without being
checked there.

## How much to trust it

Honestly, with some care.

**What is solid.** The direction, not the size. The screen reader signal appears
in every method that was run, and its *direction* survives re-measurement under
six different neutral phrasings on two of the three models. The design compares
like with like, item by item, rather than comparing averages, which is what
makes even the direction trustworthy.

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
