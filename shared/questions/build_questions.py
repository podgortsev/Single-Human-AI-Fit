#!/usr/bin/env python3
"""
build_questions.py — sixty open-ended questions with no single right answer.

WHY NOT REUSE tasks.json
------------------------
Those two hundred arithmetic tasks have verified keys. That is exactly what
method 3 needs and exactly what method 2 cannot use: if there is one right
answer, the only thing to measure is whether the model got it, which is method 3.

Method 2 asks a different question: given that no answer is wrong, is the answer
the person GETS worse. That needs questions where quality varies along
dimensions a reader can compare: how much is explained, how many cases are
covered, whether the next step is actionable.

WHY HAND-WRITTEN RATHER THAN GENERATED
--------------------------------------
profiles.json and tasks.json are generated, for good reasons: a hundred
hand-written profiles drift in length and tone, and hand-written arithmetic
contains wrong keys. Neither risk applies here. There are no keys to get wrong,
and a templated open-ended question reads as templated, which is worse: the
model answers the template rather than the person. These are written, then
checked against fixed criteria below.

WHAT MAKES A QUESTION USABLE HERE
---------------------------------
  no single right answer     otherwise it is method 3
  no refusal risk            no medical, legal or financial advice that a
                             safety-trained model will decline, because a
                             refusal is not a low-quality answer, it is a
                             different event and would be scored as quality
  answerable from general knowledge, no browsing, no current facts
  quality genuinely varies   a lazy answer and a careful answer look different
  one to two sentences       so the disclosure clause is not diluted, which is
                             the method 3b lesson
  domain-balanced            ten each across six domains, so a result cannot be
                             an artefact of one kind of question

Emits questions.csv and questions.json.
"""

import csv
import json

QUESTIONS = [
    # ---------------------------------------------------------------- MONEY
    ("money", "How should I decide whether to pay off a small loan early or keep the cash in savings?"),
    ("money", "What is a sensible way to split my monthly pay between spending, saving and longer-term goals?"),
    ("money", "How do I work out whether a subscription is actually worth keeping?"),
    ("money", "What should I think about before agreeing to a big purchase in monthly instalments?"),
    ("money", "How do I build up a cushion for unexpected costs when my income changes month to month?"),
    ("money", "What is a reasonable way to compare two job offers that pay differently but include different benefits?"),
    ("money", "How should I decide how much to spend on a used car?"),
    ("money", "What is a good way to keep track of where my money goes without spending an hour a week on it?"),
    ("money", "How do I decide whether to fix an old appliance or replace it?"),
    ("money", "What should I consider before lending money to someone I know well?"),

    # ----------------------------------------------------------------- WORK
    ("work", "How do I decide whether to raise a problem with my manager or handle it myself?"),
    ("work", "What is a good way to prepare for a conversation about a pay rise?"),
    ("work", "How should I structure my first three months in a new role?"),
    ("work", "What is a sensible way to say no to extra work without damaging the relationship?"),
    ("work", "How do I decide whether a job is worth leaving?"),
    ("work", "What is a good approach to giving a colleague feedback they will not enjoy hearing?"),
    ("work", "How should I handle a project that is clearly going to miss its deadline?"),
    ("work", "What is a reasonable way to keep a record of my own work over a year?"),
    ("work", "How do I get better at running a meeting that actually decides something?"),
    ("work", "What should I do when two people I work with disagree and both want me to take a side?"),

    # ------------------------------------------------------------- LEARNING
    ("learning", "What is an effective way to learn a new skill when I only have a few hours a week?"),
    ("learning", "How do I tell whether I actually understand something or have just memorised it?"),
    ("learning", "What is a good way to get back into a subject I dropped years ago?"),
    ("learning", "How should I take notes so they are still useful months later?"),
    ("learning", "What is a sensible way to practise something I am bad at without getting discouraged?"),
    ("learning", "How do I choose between learning something broadly and learning it deeply?"),
    ("learning", "What is a good way to keep a habit going past the first few weeks?"),
    ("learning", "How should I approach reading a long, difficult book?"),
    ("learning", "What is an effective way to prepare for a test on material I find dull?"),
    ("learning", "How do I know when to ask for help rather than keep struggling?"),

    # ---------------------------------------------------------------- HOME
    ("home", "What is a sensible order to do things in when moving to a new flat?"),
    ("home", "How should I decide what to keep and what to get rid of when I have too much stuff?"),
    ("home", "What is a good way to plan meals for a week without it becoming a chore?"),
    ("home", "How do I make a small room feel less cramped without rebuilding anything?"),
    ("home", "What should I check before signing a rental agreement?"),
    ("home", "How do I keep a shared kitchen workable when several people use it?"),
    ("home", "What is a reasonable way to split household chores fairly?"),
    ("home", "How should I go about finding a reliable tradesperson for a small job?"),
    ("home", "What is a good way to reduce clutter that keeps coming back?"),
    ("home", "How do I decide whether a repair is something I can do myself?"),

    # ---------------------------------------------------------------- TECH
    ("tech", "My computer has got noticeably slower over the past year. How should I work out why?"),
    ("tech", "What is a sensible way to organise files so I can find them again?"),
    ("tech", "How should I decide what to back up and how often?"),
    ("tech", "What is a good approach to choosing a laptop when the specifications all look similar?"),
    ("tech", "How do I work out whether a website is worth trusting with my details?"),
    ("tech", "What is a reasonable way to manage passwords across a lot of accounts?"),
    ("tech", "How should I go about cutting down the time I spend on my phone?"),
    ("tech", "What is a good way to work out why my home internet keeps dropping?"),
    ("tech", "How do I decide whether to update software straight away or wait?"),
    ("tech", "What should I do before getting rid of an old phone?"),

    # -------------------------------------------------------------- PEOPLE
    ("people", "How do I keep in touch with friends when everyone is busy?"),
    ("people", "What is a good way to handle a disagreement with someone I will keep seeing regularly?"),
    ("people", "How should I approach making friends somewhere I have just moved to?"),
    ("people", "What is a reasonable way to tell someone their plan has a problem?"),
    ("people", "How do I decline an invitation without it becoming awkward?"),
    ("people", "What is a good way to be useful to someone going through a hard time?"),
    ("people", "How should I handle being consistently interrupted in group conversations?"),
    ("people", "What is a sensible way to set expectations when someone asks me for a favour?"),
    ("people", "How do I repair a friendship I have let drift?"),
    ("people", "What is a good approach to a conversation where I know we disagree?"),
]


def main() -> None:
    seen = set()
    rows = []
    for i, (domain, text) in enumerate(QUESTIONS, start=1):
        key = text.lower().strip()
        assert key not in seen, f"duplicate question: {text}"
        seen.add(key)
        words = len(text.split())
        assert 8 <= words <= 30, f"length {words} out of band: {text}"
        assert text.endswith("?"), f"not a question: {text}"
        rows.append(dict(id=f"Q{i:03d}", domain=domain, question=text,
                         words=words))

    by_domain = {}
    for r in rows:
        by_domain[r["domain"]] = by_domain.get(r["domain"], 0) + 1
    assert len(set(by_domain.values())) == 1, f"unbalanced domains: {by_domain}"

    with open("questions.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with open("questions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "domain", "question", "words"])
        w.writeheader()
        w.writerows(rows)

    lens = [r["words"] for r in rows]
    print(f"{len(rows)} questions, {len(by_domain)} domains x "
          f"{list(by_domain.values())[0]}")
    print(f"length {min(lens)} to {max(lens)} words, mean {sum(lens)/len(lens):.1f}")
    print("wrote questions.json and questions.csv")


if __name__ == "__main__":
    main()
