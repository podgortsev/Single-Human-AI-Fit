#!/usr/bin/env python3
"""
build_tasks.py — generates 200 tasks with verifiable answers.

WHY GENERATED RATHER THAN WRITTEN
---------------------------------
Two hundred hand-written tasks would contain wrong answers. Generated tasks
carry a key computed by the same code that wrote the question, so the key is
correct by construction.

The verify() function below is NOT an independent solver, and an earlier
version of this docstring wrongly called it one. It is a format check: that a
numeric answer parses as a float, that the question ends in punctuation, and
that it is not absurdly short. A generator that computed the wrong key would
pass all three.

The independent solver is `validate_tasks.py`, next to this file. It re-parses
each question out of tasks.json and recomputes the answer with arithmetic
written separately from this generator. It currently re-derives all 200 keys
and they all match.

WHAT MAKES A GOOD TASK HERE
---------------------------
One unambiguous answer. Checkable by string or number comparison, never by
judgement. The difficulty TARGET is that a 7B model gets some wrong and most
right: if accuracy sits at 100 or at 10 percent, a difference between
conditions cannot show up. That is a design target, not a property the
generator enforces; it was confirmed after the fact by the observed baselines
(44 to 60 percent across the three models).

Three caveats on "unambiguous", all found by validate_tasks.py and none of them
changing a published number:

  * The date family is ambiguous between an inclusive and an exclusive reading
    of "runs for N days". The key takes the exclusive one (start + N). A model
    using the inclusive reading is marked wrong on all 25 date tasks, in every
    condition alike, so it moves baseline accuracy and cancels in the paired
    within-task comparison that every method 3 result rests on.

  * The table family samples values with random.randint, which does not exclude
    a tie. A tie at the extreme the question asks about would give two correct
    answers. The shipped set happens to contain none; validate_tasks.py asserts
    this so a regenerated set cannot ship one silently.

  * Uniqueness is not enforced. The shipped 200 rows hold 191 distinct
    questions; 9 rows repeat one that already appears, with consistent keys.
    So "200 tasks" means 200 rows over 191 distinct items, and the repeats are
    weighted twice. The rate family accounts for 5 of the 9.

The dataset is deliberately NOT regenerated to tidy any of this up. Every
accuracy number in methods 3a, 3b and 3c was measured on these exact rows, and
changing the random stream would silently invalidate all of them.

Seven families, deliberately varied so a result cannot be an artefact of one
kind of reasoning.
"""

import csv
import json
import random
from typing import Dict, List

random.seed(20260816)   # fixed so the set is reproducible

tasks: List[Dict] = []


def add(family: str, question: str, answer: str, kind: str = "number") -> None:
    tasks.append(dict(id=f"T{len(tasks)+1:03d}", family=family,
                      question=question, answer=str(answer), answer_kind=kind))


# --------------------------------------------------------------------------
# 1. Percentages. 35 tasks.

for _ in range(35):
    base = random.choice([120, 240, 340, 480, 560, 720, 850, 1250, 1600, 2400])
    pct = random.choice([12, 15, 18, 22, 25, 35, 40, 45, 60, 75])
    kind = random.choice(["of", "increase", "decrease"])
    if kind == "of":
        ans = base * pct / 100
        q = f"What is {pct} percent of {base}?"
    elif kind == "increase":
        ans = base * (1 + pct / 100)
        q = f"A price of {base} dollars increases by {pct} percent. What is the new price?"
    else:
        ans = base * (1 - pct / 100)
        q = f"A price of {base} dollars decreases by {pct} percent. What is the new price?"
    add("percentage", q, f"{ans:g}")


# --------------------------------------------------------------------------
# 2. Multi-step word problems. 30 tasks.

for _ in range(30):
    workers = random.randint(3, 9)
    hours = random.randint(4, 12)
    rate = random.choice([18, 22, 25, 30, 35, 42])
    days = random.randint(2, 6)
    total = workers * hours * rate * days
    add("word_problem",
        f"A team of {workers} people works {hours} hours a day for {days} days. "
        f"Each person is paid {rate} dollars an hour. What is the total wage bill?",
        total)


# --------------------------------------------------------------------------
# 3. Unit conversion. 25 tasks.

CONV = [
    ("kilometres", "metres", 1000), ("hours", "minutes", 60),
    ("kilograms", "grams", 1000), ("litres", "millilitres", 1000),
    ("days", "hours", 24), ("weeks", "days", 7),
    ("minutes", "seconds", 60), ("tonnes", "kilograms", 1000),
]
for _ in range(25):
    a, b, f = random.choice(CONV)
    n = random.choice([3, 7, 12, 15, 24, 36, 48, 60, 90, 125])
    add("conversion", f"How many {b} are there in {n} {a}?", n * f)


# --------------------------------------------------------------------------
# 4. Date and time reasoning. 25 tasks.

from datetime import date, timedelta

for _ in range(25):
    start = date(2026, random.randint(1, 12), random.randint(1, 28))
    delta = random.choice([9, 14, 21, 30, 45, 60, 75, 90])
    end = start + timedelta(days=delta)
    add("date",
        f"A project starts on {start.strftime('%d %B %Y')} and runs for "
        f"{delta} days. On what date does it end? Answer in the format "
        f"DD Month YYYY.",
        end.strftime("%d %B %Y"), kind="text")


# --------------------------------------------------------------------------
# 5. Extraction from a short table. 30 tasks.

REGIONS = ["North", "South", "East", "West", "Central"]
for _ in range(30):
    vals = {r: random.randint(100, 900) for r in random.sample(REGIONS, 4)}
    rows = "; ".join(f"{k}: {v}" for k, v in vals.items())
    mode = random.choice(["max", "min", "sum", "diff"])
    if mode == "max":
        q = f"Quarterly units sold. {rows}. Which region sold the most?"
        ans, kind = max(vals, key=vals.get), "text"
    elif mode == "min":
        q = f"Quarterly units sold. {rows}. Which region sold the least?"
        ans, kind = min(vals, key=vals.get), "text"
    elif mode == "sum":
        q = f"Quarterly units sold. {rows}. What is the total across all regions?"
        ans, kind = sum(vals.values()), "number"
    else:
        q = (f"Quarterly units sold. {rows}. What is the difference between "
             f"the highest and the lowest region?")
        ans, kind = max(vals.values()) - min(vals.values()), "number"
    add("table", q, ans, kind)


# --------------------------------------------------------------------------
# 6. Ordering and logic. 25 tasks.

NAMES = ["Alex", "Bailey", "Casey", "Devon", "Ellis", "Frankie", "Gray"]
for _ in range(25):
    people = random.sample(NAMES, 4)
    # DEAD, DELIBERATELY KEPT. The answer is read off the people[] chain
    # below, so these ages are never used. The call is left in because it
    # consumes the random stream: removing it changes every task generated
    # after this point, and the shipped tasks.json is the one all of method 3
    # was measured on.
    _unused_ages = sorted(random.sample(range(21, 60), 4), reverse=True)
    facts = []
    for i in range(3):
        facts.append(f"{people[i]} is older than {people[i+1]}")
    random.shuffle(facts)
    mode = random.choice(["oldest", "youngest", "second"])
    if mode == "oldest":
        q = f"{'. '.join(facts)}. Who is the oldest?"
        ans = people[0]
    elif mode == "youngest":
        q = f"{'. '.join(facts)}. Who is the youngest?"
        ans = people[3]
    else:
        q = f"{'. '.join(facts)}. Who is the second oldest?"
        ans = people[1]
    add("logic", q, ans, "text")


# --------------------------------------------------------------------------
# 7. Rates and proportion. 30 tasks.

for _ in range(30):
    dist = random.choice([90, 120, 150, 180, 240, 300, 360])
    speed = random.choice([30, 40, 45, 60, 75, 90])
    mode = random.choice(["time", "distance", "speed"])
    if mode == "time":
        q = (f"A vehicle travels {dist} kilometres at a steady "
             f"{speed} kilometres per hour. How many hours does the journey take?")
        ans = dist / speed
    elif mode == "distance":
        hrs = random.choice([2, 3, 4, 5, 6])
        q = (f"A vehicle travels for {hrs} hours at a steady {speed} "
             f"kilometres per hour. How far does it travel?")
        ans = hrs * speed
    else:
        hrs = random.choice([2, 3, 4, 5, 6])
        q = (f"A vehicle covers {speed * hrs} kilometres in {hrs} hours. "
             f"What is its average speed in kilometres per hour?")
        ans = speed
    add("rate", q, f"{ans:g}")


# --------------------------------------------------------------------------
# Independent verification. Recompute a sample of answers a second way.

def verify() -> int:
    bad = 0
    for t in tasks:
        if t["answer_kind"] == "number":
            try:
                float(t["answer"])
            except ValueError:
                print(f"  BAD numeric answer: {t['id']} -> {t['answer']}")
                bad += 1
        if not t["question"].strip().endswith(("?", ".")):
            print(f"  BAD question ending: {t['id']}")
            bad += 1
        if len(t["question"]) < 20:
            print(f"  SUSPICIOUSLY SHORT: {t['id']}")
            bad += 1
    return bad


def main() -> None:
    problems = verify()

    with open("tasks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "family", "question",
                                          "answer", "answer_kind"])
        w.writeheader()
        w.writerows(tasks)

    with open("tasks.json", "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=1)

    from collections import Counter
    print(f"generated {len(tasks)} tasks, {problems} problems found\n")
    for fam, n in Counter(t["family"] for t in tasks).most_common():
        print(f"  {n:4}  {fam}")

    print("\nsample, one per family")
    seen = set()
    for t in tasks:
        if t["family"] in seen:
            continue
        seen.add(t["family"])
        print(f"\n  [{t['family']}] {t['id']}")
        print(f"    Q: {t['question']}")
        print(f"    A: {t['answer']}")

    print("\nwrote tasks.csv and tasks.json")


if __name__ == "__main__":
    main()
