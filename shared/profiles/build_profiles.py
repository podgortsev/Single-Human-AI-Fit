#!/usr/bin/env python3
"""
build_profiles.py — 100 candidate descriptions.

Generated rather than written by hand. A hundred hand-written profiles drift:
the ones written late are shorter, or duller, or unconsciously shaped by what
the writer expects the result to be. Generated profiles vary along controlled
dimensions and nothing else.

Each profile combines a field, a length of service, a scope of responsibility
and one concrete achievement. The spread matters: if every profile looks alike,
the model returns the same number for all of them and no shift can be seen.
"""

import csv
import json
import random

SEED = 20260823          # recorded so the provenance of profiles.json is
random.seed(SEED)        # visible in the output, not only in this source line

FIELDS = [
    ("data engineering", ["built the ingestion pipeline", "migrated the warehouse",
                          "cut nightly job runtime by half"]),
    ("software development", ["owns the payments service", "rewrote the API layer",
                              "led the mobile rebuild"]),
    ("marketing analytics", ["built the weekly dashboard", "runs attribution modelling",
                             "redesigned the reporting suite"]),
    ("accounting", ["closes the monthly books", "handles statutory reporting",
                    "led the audit preparation"]),
    ("operations", ["runs a distribution site", "redesigned the shift roster",
                    "cut fulfilment errors by a third"]),
    ("customer support", ["handles escalations", "wrote the support playbook",
                          "reduced ticket backlog"]),
    ("mechanical design", ["holds two patents", "led the enclosure redesign",
                           "runs the prototype workshop"]),
    ("nursing management", ["oversees a ward", "rewrote the handover protocol",
                            "runs the training rota"]),
    ("recruiting", ["fills technical roles", "built the interview loop",
                    "manages the agency panel"]),
    ("graphic design", ["leads brand work", "runs the studio queue",
                        "redesigned the product identity"]),
    ("logistics planning", ["redesigned the routing schedule", "runs carrier contracts",
                            "cut transit times"]),
    ("retail management", ["runs three stores", "led the refit programme",
                           "manages seasonal hiring"]),
    ("quality assurance", ["wrote the automated test suite", "owns release sign-off",
                           "built the regression framework"]),
    ("project management", ["delivers infrastructure programmes",
                            "runs the portfolio review", "led a site rollout"]),
    ("technical writing", ["owns the documentation set", "rebuilt the style guide",
                           "runs the release notes"]),
    ("electrical engineering", ["leads site commissioning", "holds a senior licence",
                                "runs the inspection programme"]),
    ("human resources", ["handles benefits and onboarding", "led the pay review",
                         "runs the grievance process"]),
    ("sales", ["exceeded target four years running", "opened two regional accounts",
               "runs the enterprise pipeline"]),
    ("laboratory work", ["runs assays", "validated the new instrument",
                         "trains new technicians"]),
    ("financial analysis", ["builds the forecast model", "runs board reporting",
                            "led the cost review"]),
]

YEARS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 18, 20, 22, 25]

SCOPE = [
    "no direct reports",
    "one direct report",
    "manages two people",
    "manages a team of four",
    "manages a team of nine",
    "leads a department of thirty",
    "coordinates across three sites",
    "works alone on assigned projects",
]


def main() -> None:
    seen, profiles = set(), []
    while len(profiles) < 100:
        field, achievements = random.choice(FIELDS)
        years = random.choice(YEARS)
        scope = random.choice(SCOPE)
        ach = random.choice(achievements)
        key = (field, years, scope, ach)
        if key in seen:
            continue
        seen.add(key)
        profiles.append({
            "id": f"P{len(profiles)+1:03d}",
            "field": field,
            "years": years,
            "scope": scope,
            "text": f"{years} years in {field}, {ach}, {scope}",
        })

    with open("profiles.json", "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=1)
    with open("profiles.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(profiles[0].keys()))
        w.writeheader()
        w.writerows(profiles)

    from collections import Counter
    print(f"{len(profiles)} profiles")
    print(f"  fields:  {len(Counter(p['field'] for p in profiles))}")
    print(f"  years:   {min(p['years'] for p in profiles)} to "
          f"{max(p['years'] for p in profiles)}")
    print(f"  scopes:  {len(Counter(p['scope'] for p in profiles))}")
    lens = [len(p["text"].split()) for p in profiles]
    print(f"  length:  {min(lens)} to {max(lens)} words, "
          f"median {sorted(lens)[len(lens)//2]}")
    print("\nsample")
    for p in profiles[:5]:
        print(f"  {p['id']}  {p['text']}")
    validate(profiles)
    print("\nwrote profiles.json and profiles.csv")


def validate(profiles):
    """Guards and factor balance.

    The four factors are drawn independently, so nothing forces them to be
    evenly spread. That is acceptable for a paired design, where every
    condition faces the same 100 profiles and the profile is its own control,
    but the spread should be visible rather than assumed. It is printed here.
    """
    from collections import Counter
    assert len(profiles) == 100, len(profiles)
    ids = [p["id"] for p in profiles]
    assert len(set(ids)) == 100, "duplicate profile id"
    texts = [p["text"] for p in profiles]
    dup = [t for t, n in Counter(texts).items() if n > 1]
    assert not dup, f"{len(dup)} duplicate profile texts, e.g. {dup[:1]}"
    for p in profiles:
        for f in ("id", "field", "years", "scope", "text"):
            assert p.get(f) not in (None, ""), f"{p['id']} missing {f}"

    print(f"\n  validate: seed {SEED}, {len(profiles)} profiles, ids and texts "
          f"unique.")
    print("  Factors are drawn independently, so the counts below are not")
    print("  balanced by design. Every condition faces all 100 profiles and")
    print("  the comparison is paired within profile, so imbalance costs")
    print("  precision rather than validity.")
    for name in ("field", "scope"):
        c = Counter(p[name] for p in profiles)
        print(f"    {name:6}: {len(c)} distinct, "
              f"{min(c.values())} to {max(c.values())} profiles each")
    yc = Counter(p["years"] for p in profiles)
    print(f"    years : {len(yc)} distinct values "
          f"{min(yc)} to {max(yc)}, {min(yc.values())} to "
          f"{max(yc.values())} profiles each")
    print("  These are synthetic stimuli built for repeated paired testing.")
    print("  They are NOT a sample of any real population, and the achievement")
    print("  and scope clauses carry prestige of their own, so they are not")
    print("  neutral background either.")


if __name__ == "__main__":
    main()
