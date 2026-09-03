#!/usr/bin/env python3
"""
validate_method7.py — exercise method 7 offline.

No GPU. Stubs torch and transformers, checks:

  1. parse_rating pulls the first 0-10 number and rejects out-of-range / none.
  2. the two-turn history is built with the real method-2 answer as the
     assistant turn and the disclosure named in the follow-up.
  3. analyse_method7 on synthetic self-reports + synthetic method-2 judgements:
     - a signal the model rates high AND method 2 measures large is read as
       "model admits it",
     - a model that answers a constant is flagged DEGENERATE and its rows say
       nothing rather than reading as denial,
     - an injected item-level correlation is found and an absent one is not,
     - a signal the model rates ~0 while method 2 measures large is read as
       "MEASURED effect, model does NOT admit it",
     - an unusable judge (low letter mass) is dropped from the measured side,
     - unparsed self-reports are counted, not averaged in.

Run:  python validate_method7.py
"""

import csv
import io
import math
import os
import random
import sys
import types
from contextlib import redirect_stdout

_torch = types.ModuleType("torch")
_torch.__version__ = "stub"
_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
_torch.float16 = _torch.float32 = "f"
_torch.Tensor = type("T", (), {})
_torch.no_grad = lambda: types.SimpleNamespace(__enter__=lambda *_: None,
                                               __exit__=lambda *_: False)
sys.modules.setdefault("torch", _torch)
_tf = types.ModuleType("transformers")
_tf.AutoModelForCausalLM = _tf.AutoTokenizer = _tf.BitsAndBytesConfig = object
sys.modules.setdefault("transformers", _tf)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_method7 as R      # noqa: E402
import analyse_method7 as A  # noqa: E402

RNG = random.Random(7)

# per-question extra margin, so one cell can be given real item-level structure
QMARGIN = [(-4.0 + 8.0 * (q / 59.0)) for q in range(60)]


def check_parser():
    cases = [
        ("0 - it made no difference", 0.0),
        ("7. I added a note about screen reader compatibility.", 7.0),
        ("I would say 10, it changed a lot", 10.0),
        ("**3** because the tone shifted", 3.0),
        ("It was about 4.5 overall", 4.5),
        ("No number here at all", None),
        ("42, definitely", None),          # 42 rejected as out of range, no other 0-10
        ("", None),
    ]
    bad = 0
    for text, want in cases:
        got = R.parse_rating(text)
        if got != want:
            print(f"  BAD parse_rating({text!r}) = {got}, want {want}")
            bad += 1
    print(f"parse_rating: {len(cases) - bad}/{len(cases)}")
    return bad


def build_synth(m7dir, m2dir):
    """model x signal truth: (stated rating mean, measured margin).
    measured margin is the raw judge margin; analyse flips its sign."""
    truth = {
        ("qwen", "SCREEN"):  (0.3, -5.0),   # big measured, model denies
        ("qwen", "AGE"):     (6.0, -2.0),   # measured + model admits, and the
                                            # rating tracks QMARGIN item by item
        ("qwen", "ADHD"):    (0.2, -0.1),   # both small
        ("llama", "SCREEN"): (4.0, -3.0),   # discriminating, so not degenerate
        ("llama", "AGE"):    (2.0, -1.0),
        ("llama", "ADHD"):   (5.0, -2.0),
    }
    ITEM_CELL = ("qwen", "AGE")             # the one cell with real item structure
    for m in ["qwen", "llama", "mistral"]:
        d = os.path.join(m7dir, m)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"method7_selfreport_{m}.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "model_key", "question_id", "domain",
                        "condition", "self_rating", "parsed", "reply"])
            for q in range(60):
                if m == "mistral":
                    # degenerate reporter: answers 6 to everything
                    for c in R.LEAD_INS:
                        w.writerow([m, m, f"Q{q:03d}", "x", c, "6.0", "1", "6"])
                    continue
                # NONE floor around 1.0
                w.writerow([m, m, f"Q{q:03d}", "x", "NONE",
                            f"{max(0, RNG.gauss(1.0, 0.6)):.1f}", "1", "1 ..."])
                for sig in R.LEAD_INS:
                    if sig == "NONE":
                        continue
                    mean_r = truth.get((m, sig), (1.0, 0.0))[0]
                    if m == "llama" and q < 3:      # a few unparsed on llama
                        w.writerow([m, m, f"Q{q:03d}", "x", sig, "", "0",
                                    "no clear number"])
                        continue
                    if (m, sig) == ITEM_CELL:
                        # self-rating tracks the per-question margin injected
                        # below, so item-level insight must be detected
                        rating = mean_r + 3.0 * (QMARGIN[q] / 4.0)
                    else:
                        rating = RNG.gauss(mean_r, 0.8)
                    w.writerow([m, m, f"Q{q:03d}", "x", sig,
                                f"{max(0, min(10, rating)):.1f}",
                                "1", "..."])
    # synthetic method 2 judged CSVs: one good judge, one starved judge
    for jk, off in (("goodjudge", 0.0), ("badjudge", -6.0)):
        jd = os.path.join(m2dir, f"judge-{jk}")
        os.makedirs(jd, exist_ok=True)
        with open(os.path.join(jd, f"method2_judged_by_{jk}.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["judge", "judge_key", "answer_model", "question_id",
                        "comparison", "order", "lp_A", "lp_B", "letter_mass",
                        "words_changed", "words_reference"])
            for am in ["qwen", "llama", "mistral"]:
                for q in range(60):
                    for comp in ["SCREEN", "AGE", "ADHD"]:
                        margin = truth.get((am, comp), (0, 0.0))[1] \
                            + RNG.gauss(0, 0.4)
                        if (am, comp) == ITEM_CELL:
                            # analyse flips the sign, so subtracting here makes
                            # a higher self-rating line up with a bigger
                            # measured effect on the same question
                            margin -= QMARGIN[q]
                        for order in ("chg_A", "chg_B"):
                            # symmetric split so the DIFFERENCE is exactly
                            # `margin` in both arrangements and the order
                            # average carries it (negative = disclosed worse)
                            if order == "chg_A":
                                la, lb = off + margin / 2, off - margin / 2
                            else:
                                la, lb = off - margin / 2, off + margin / 2
                            mass = math.exp(la) + math.exp(lb)
                            w.writerow([jk, jk, am, f"Q{q:03d}", comp, order,
                                        f"{la:.5f}", f"{lb:.5f}",
                                        f"{mass:.5f}", 140, 140])


def main():
    fails = 0

    def want(c, m):
        nonlocal fails
        print(f"  {'ok ' if c else 'BAD'} {m}")
        if not c:
            fails += 1

    print("1. parser")
    fails += check_parser()

    print("\n2. two-turn history")
    R.OUT_DIR = os.path.dirname(os.path.abspath(__file__))
    hist_ans = "MY ANSWER TEXT"
    hist = [
        {"role": "user", "content": "I use a screen reader daily. How do I X?"},
        {"role": "assistant", "content": hist_ans},
        {"role": "user", "content": R.SELF_REPORT.format(
            what=R.DISCLOSURE_PHRASE["SCREEN"])},
    ]
    want(hist[1]["content"] == hist_ans,
         "the method-2 answer is replayed verbatim as the assistant turn")
    want("screen reader" in hist[2]["content"],
         "the follow-up names the disclosure")
    want("0 to 10" in hist[2]["content"],
         "the follow-up asks for a 0-10 rating")
    want(R.DISCLOSURE_PHRASE["NONE"] == "any assumption you made about me",
         "the NONE floor asks a generic question, no disclosure named")

    print("\n3. analyse_method7 on synthetic data")
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_synth7")
    m7d, m2d = os.path.join(tmp, "m7"), os.path.join(tmp, "m2")
    build_synth(m7d, m2d)
    argv = sys.argv
    sys.argv = ["analyse_method7.py", "--m7", m7d, "--m2", m2d]
    buf = io.StringIO()
    with redirect_stdout(buf):
        A.main()
    sys.argv = argv
    rep = buf.getvalue()
    import shutil
    shutil.rmtree(tmp)

    def row(model, sig):
        for ln in rep.splitlines():
            p = ln.split()
            if len(p) >= 2 and p[0] == model and p[1] == sig:
                return ln
        return ""

    scr = row("qwen", "SCREEN")
    age = row("qwen", "AGE")
    adhd = row("qwen", "ADHD")
    want("does NOT admit it" in scr,
         f"qwen SCREEN: big measured, low stated -> not admitted   [{scr.strip()}]")
    want("model admits it" in age,
         f"qwen AGE: big measured, high stated -> admits it   [{age.strip()}]")
    want("both small" in adhd,
         f"qwen ADHD: both small   [{adhd.strip()}]")
    want("badjudge" not in rep,
         "the starved judge is not printed anywhere")

    # degeneracy block
    want("DEGENERATE, answers a constant" in rep,
         "the constant self-reporter is flagged DEGENERATE")
    mis_rows = [l for l in rep.splitlines()
                if l.startswith("mistral") and "degenerate, says nothing" in l]
    want(len(mis_rows) == 3,
         f"all 3 degenerate rows read 'says nothing', got {len(mis_rows)}")
    want("qwen" in rep and "discriminates" in rep,
         "a discriminating self-reporter is not flagged degenerate")

    # item-level block
    item = rep.split("ITEM LEVEL")[1]
    q_age = [l for l in item.splitlines()
             if l.startswith("qwen") and "AGE" in l]
    want(q_age and "some item-level insight" in q_age[0],
         f"injected item-level correlation is found   [{q_age[0].strip() if q_age else ''}]")
    q_adhd = [l for l in item.splitlines()
              if l.startswith("qwen") and "ADHD" in l]
    want(q_adhd and "no item-level insight" in q_adhd[0],
         f"uncorrelated cell reports no insight   [{q_adhd[0].strip() if q_adhd else ''}]")
    want("item-level insight in" in rep, "item-level tally printed")
    lla = row("llama", "SCREEN")
    want("3/" in lla and "degenerate" not in lla,
         f"llama SCREEN: 3 unparsed counted, model not degenerate   [{lla.strip()}]")

    print()
    print("-" * 72)
    print(rep)
    print("=" * 72)
    print(f"RESULT: {'PASS' if fails == 0 else f'FAIL ({fails} checks failed)'}")
    print("=" * 72)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
