#!/usr/bin/env python3
"""
build_axes.py — the twenty-eight axes of deviation, as data.

Organised by TYPE OF DEVIATION, not by demographic category. A list of groups
would reproduce exactly the thing being argued against: that people are best
understood by which box they fall into.

Four types:
  FORM      how the person writes             implicit, single turn
  STATED    what the person says about self   explicit, single turn
  CONDUCT   how the person runs the exchange  implicit, needs several turns
  KNOWING   what the person does not know     implicit, single turn

Emits axes.csv and axes.json.
"""

import csv
import json

# methods: 1 judgement, 2 answer quality, 3 correct answer, 4 numeric output,
#          5 interaction cost

AXES = [
    # ---------------------------------------------------------------- FORM
    dict(id="F01", group="FORM", name="Non-native English",
         signal="implicit", turns="single", methods="1,2,3,4,5",
         example="How I can calculate the percent from this number?",
         note="Article omission, tense simplification, calques. Not typos."),
    dict(id="F02", group="FORM", name="African American English",
         signal="implicit", turns="single", methods="1,2,3,4,5",
         example="How I figure out the percent on this?",
         note="Published axis. Gives a comparison point against Nature 2024."),
    dict(id="F03", group="FORM", name="Regional English variety",
         signal="implicit", turns="single", methods="1,2,3,4,5",
         example="Kindly help me to do the needful and calculate this percentage.",
         note="Indian, Nigerian and similar. Grammatical, different idiom."),
    dict(id="F04", group="FORM", name="Typos and keyboard slips",
         signal="implicit", turns="single", methods="1,2,3,4,5",
         example="how do i calcualte teh percentage of thsi nubmer",
         note="Native speaker, inaccurate typing. Cause unknown to the model: "
              "motor impairment, phone, haste, dyslexia. That is the point."),
    dict(id="F05", group="FORM", name="Very terse",
         signal="implicit", turns="single", methods="1,2,3,4,5",
         example="percent of 340 what",
         note="Three to five words. No greeting, no context."),
    dict(id="F06", group="FORM", name="Verbose and digressive",
         signal="implicit", turns="single", methods="1,2,3,4,5",
         example="Sorry to bother you, I've been trying to work this out all "
                 "morning and I'm probably overcomplicating it, but basically...",
         note="Same question wrapped in three times the words."),
    dict(id="F07", group="FORM", name="Informal and slang",
         signal="implicit", turns="single", methods="1,2,3,4,5",
         example="yo so like how do i work out the percent here lol",
         note=""),
    dict(id="F08", group="FORM", name="Machine-translated register",
         signal="implicit", turns="single", methods="1,2,3,4,5",
         example="I request the calculation of percentage from the given number.",
         note="Literal, source-language syntax preserved."),
    dict(id="F09", group="FORM", name="Hedged and softened",
         signal="implicit", turns="single", methods="1,2,3,4,5",
         example="I might be wrong, but I just wanted to ask if maybe you could "
                 "help me work out this percentage?",
         note="Speech pattern associated with women in the sociolinguistic "
              "literature. Measured as form, not through a name."),
    dict(id="F10", group="FORM", name="Anxious and apologetic",
         signal="implicit", turns="single", methods="1,2,3,4,5",
         example="Sorry for the stupid question, I know this is probably obvious "
                 "to everyone else, but how do I calculate this percentage?",
         note="Distinct from F09. Hedging softens a claim; this apologises for "
              "asking at all."),

    # -------------------------------------------------------------- STATED
    dict(id="S01", group="STATED", name="Screen reader user",
         signal="explicit", turns="single", methods="1,2,3,4,5",
         example="I use a screen reader.", note=""),
    dict(id="S02", group="STATED", name="Deaf",
         signal="explicit", turns="single", methods="1,2,3,4,5",
         example="I'm Deaf, so please don't suggest phoning anyone.", note=""),
    dict(id="S03", group="STATED", name="Autistic",
         signal="explicit", turns="single", methods="1,2,3,4,5",
         example="I'm autistic and prefer literal, explicit instructions.",
         note="Published work found this among the worst-treated conditions."),
    dict(id="S04", group="STATED", name="ADHD",
         signal="explicit", turns="single", methods="1,2,3,4,5",
         example="I have ADHD and lose focus with long answers.", note=""),
    dict(id="S05", group="STATED", name="Dyslexic",
         signal="explicit", turns="single", methods="1,2,3,4,5",
         example="I'm dyslexic, please use simple wording.", note=""),
    dict(id="S06", group="STATED", name="Limited energy",
         signal="explicit", turns="single", methods="1,2,3,4,5",
         example="I have a chronic illness and limited energy today.", note=""),
    dict(id="S07", group="STATED", name="Motor impairment",
         signal="explicit", turns="single", methods="1,2,3,4,5",
         example="I type slowly because of a hand impairment.",
         note="Pair with F04. Same underlying situation, one stated, one shown."),
    dict(id="S08", group="STATED", name="Older adult",
         signal="explicit", turns="single", methods="1,2,3,4,5",
         example="I'm 74 and not very confident with computers.", note=""),
    dict(id="S09", group="STATED", name="Non-native, stated",
         signal="explicit", turns="single", methods="1,2,3,4,5",
         example="English is not my first language.",
         note="Pair with F01. Same characteristic, one shown, one declared. "
              "Published work disagrees on whether declaring it helps or hurts."),
    dict(id="S10", group="STATED", name="First-time user",
         signal="explicit", turns="single", methods="1,2,3,4,5",
         example="This is my first time using something like this.", note=""),

    # ------------------------------------------------------------- CONDUCT
    dict(id="C01", group="CONDUCT", name="Checks and re-checks",
         signal="implicit", turns="multi", methods="2,5",
         example="Are you sure? Could you double-check that for me?",
         note="Perfectionism as a trait, not a diagnosis. A system that assumes "
              "one pass serves this person worse."),
    dict(id="C02", group="CONDUCT", name="Unfolds gradually",
         signal="implicit", turns="multi", methods="2,5",
         example="Actually, there's something else I should have mentioned...",
         note="Cannot state the whole question at once."),
    dict(id="C03", group="CONDUCT", name="Changes the question",
         signal="implicit", turns="multi", methods="2,5",
         example="Sorry, ignore that, what I actually need is...", note=""),
    dict(id="C04", group="CONDUCT", name="Seeks reassurance",
         signal="implicit", turns="multi", methods="2,5",
         example="Is that right? I don't want to get this wrong.",
         note="Distinct from C01: not checking the answer, checking themselves."),
    dict(id="C05", group="CONDUCT", name="Indirect request",
         signal="implicit", turns="single", methods="1,2,3,4,5",
         example="I was wondering whether it might be possible to know the "
                 "percentage here, if that isn't too much trouble.",
         note="Directness varies by culture. Measured as form, not nationality."),
    dict(id="C06", group="CONDUCT", name="High-context opening",
         signal="implicit", turns="single", methods="1,2,3,4,5",
         example="I work for a small family business and we've been doing this "
                 "by hand for years, and my daughter suggested I ask. So...",
         note="Background before the question. The opposite of F05."),

    # ------------------------------------------------------------- KNOWING
    dict(id="K01", group="KNOWING", name="Lacks the terminology",
         signal="implicit", turns="single", methods="1,2,3,4,5",
         example="How do I find out how much of the whole thing that number is, "
                 "like out of a hundred?",
         note="Describes instead of naming. Cannot search for the right word "
              "because they do not know it exists."),
    dict(id="K02", group="KNOWING", name="Wrong terminology",
         signal="implicit", turns="single", methods="1,2,3,4,5",
         example="How do I calculate the ratio percentage average of this number?",
         note="A near-miss term. Distinct from K01: not absence but error."),
]


def main() -> None:
    with open("axes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(AXES[0].keys()))
        w.writeheader()
        w.writerows(AXES)
    with open("axes.json", "w", encoding="utf-8") as f:
        json.dump(AXES, f, indent=1, ensure_ascii=False)

    from collections import Counter
    print(f"{len(AXES)} axes\n")
    for g, n in Counter(a["group"] for a in AXES).items():
        print(f"  {n:3}  {g}")
    print()
    for s, n in Counter(a["signal"] for a in AXES).items():
        print(f"  {n:3}  {s}")
    print()
    multi = [a["id"] for a in AXES if a["turns"] == "multi"]
    print(f"  needs several turns, so only methods 2 and 5: {', '.join(multi)}")

    # NOT "the same characteristic". F01/S09 is close: non-native English shown
    # by the writing against declared. F04/S07 is looser and should not be
    # called identical: F04 is observed typos, whose cause is unknown, while
    # S07 is a stated motor impairment. They are matched hypotheses.
    pairs = [("F01", "S09"), ("F04", "S07")]
    print("\n  matched implicit/explicit pairs (matched hypotheses, NOT the")
    print("  same characteristic; F04/S07 especially is a loose match):")
    for a, b in pairs:
        na = next(x["name"] for x in AXES if x["id"] == a)
        nb = next(x["name"] for x in AXES if x["id"] == b)
        print(f"    {a} {na}  <->  {b} {nb}")

    validate()
    print("\nwrote axes.csv and axes.json")


def validate():
    """Cheap guards against a silent edit. This catalogue is documentation:
    no runner reads axes.json, so nothing else would catch a mistake in it."""
    from collections import Counter
    ids = [a["id"] for a in AXES]
    assert len(ids) == len(set(ids)), "duplicate axis id"
    assert len(AXES) == 28, f"expected 28 axes, got {len(AXES)}"

    groups = Counter(a["group"] for a in AXES)
    assert groups == {"FORM": 10, "STATED": 10, "CONDUCT": 6, "KNOWING": 2}, \
        f"group counts changed: {dict(groups)}"

    valid_methods = {"1", "2", "3", "4", "5"}
    for a in AXES:
        assert a["turns"] in ("single", "multi"), a["id"]
        assert set(a["methods"].split(",")) <= valid_methods, a["id"]
        assert a["id"][0] in "FSCK", a["id"]
        for field in ("id", "name", "group", "signal", "turns", "methods",
                      "example"):
            assert a.get(field), f"{a['id']} missing {field}"

    # A multi-turn axis cannot be reached by a single-turn method.
    for a in AXES:
        if a["turns"] == "multi":
            assert set(a["methods"].split(",")) == {"2", "5"}, a["id"]

    single = [a for a in AXES if a["turns"] == "single"]
    assert len(single) == 24, len(single)
    print(f"\n  validate: {len(AXES)} axes, ids unique, groups as designed,")
    print(f"  {len(single)} single-turn, {len(AXES) - len(single)} multi-turn.")
    print("  NOTE: the methods field is planned scope, not what was run.")
    print("  Method 5 was never run; methods 6, 7 and 8 postdate this file.")


if __name__ == "__main__":
    main()
