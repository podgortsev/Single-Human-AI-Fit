#!/usr/bin/env python3
"""
validate_method2.py — exercise both stages of method 2 offline.

No GPU, no model. Stubs torch and transformers, then checks:

  1. STAGE 1 shape. Lead-ins are length-matched, prompts are built so the
     question text is byte-identical across conditions, and the resume key
     covers every cell.
  2. THE JUDGE IS BLIND. The judge prompt is built from the bare question and
     the two answers. No lead-in string may appear in it. This is the property
     the whole method rests on, so it is tested directly.
  3. STAGE 2 statistics on synthetic judgements with known answers:
     - a signal whose answers are genuinely judged worse is found, with the
       right sign,
     - a null is not found,
     - an injected judge slot bias is cancelled by the order swap and shows up
       in IDENTITY,
     - a starved letter mass makes a cell NOT reportable even when the margin
       looks decisive,
     - controls stay out of the BH family.

Run:  python validate_method2.py
"""

import csv
import io
import os
import random
import sys
import types
from contextlib import redirect_stdout

# --- stub the heavy deps ----------------------------------------------------
_torch = types.ModuleType("torch")
_torch.__version__ = "0.0.0-stub"
_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
_torch.float16 = "float16"
_torch.float32 = "float32"
_torch.Tensor = type("Tensor", (), {})
_torch.no_grad = lambda: types.SimpleNamespace(
    __enter__=lambda *_: None, __exit__=lambda *_: False)
sys.modules.setdefault("torch", _torch)

_tf = types.ModuleType("transformers")
_tf.AutoModelForCausalLM = object
_tf.AutoTokenizer = object
_tf.BitsAndBytesConfig = object
sys.modules.setdefault("transformers", _tf)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_method2_generate as G  # noqa: E402
import run_method2_judge as J     # noqa: E402
import analyse_method2 as A       # noqa: E402


# (combined mean, combined sd, judge slot bias mean, sd)
TRUTH = {
    "IDENTITY":        (0.00, 0.05, 1.20, 0.30),
    "NONE_vs_CONTROL": (0.00, 0.25, 1.20, 0.30),
    "SCREEN":          (-0.90, 0.40, 1.20, 0.30),   # judged clearly worse
    "AGE":             (0.00, 0.35, 1.20, 0.30),    # genuine null
    "ADHD":            (-0.35, 0.35, 1.20, 0.30),   # judged mildly worse
}
STARVE_OFFSET = -6.0


def make_judged_csv(path, seed=17, starved=()):
    rnd = random.Random(seed)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["judge", "judge_key", "answer_model", "question_id",
                    "comparison", "order", "lp_A", "lp_B", "letter_mass",
                    "words_changed", "words_reference"])
        for am in J.ANSWER_MODELS:
            for qi in range(60):
                for name, _, _ in J.COMPARISONS:
                    cm, cs, pm, ps = TRUTH[name]
                    off = STARVE_OFFSET if name in starved else 0.0
                    c = rnd.gauss(cm, cs)
                    p = rnd.gauss(pm, ps)
                    m1, m2 = c + p, c - p
                    import math
                    for order, m in (("chg_A", m1), ("chg_B", m2)):
                        # split the margin symmetrically around the offset:
                        # difference is exactly m, mass is set by the offset
                        if order == "chg_A":
                            la, lb = off + m / 2, off - m / 2
                        else:
                            la, lb = off - m / 2, off + m / 2
                        mass = math.exp(la) + math.exp(lb)
                        w.writerow(["stub", "stub", am, f"Q{qi:03d}", name,
                                    order, f"{la:.6f}", f"{lb:.6f}",
                                    f"{mass:.6f}", 140, 140])


def block_for(report, model):
    if f"ANSWERS BY {model}" not in report:
        return ""
    return report.split(f"ANSWERS BY {model}")[1].split("ANSWERS BY")[0]


def row_for(block, name):
    for ln in block.splitlines():
        if ln.startswith(name):
            return ln
    return ""


def main():
    fails = 0

    def want(cond, msg):
        nonlocal fails
        print(f"  {'ok ' if cond else 'BAD'} {msg}")
        if not cond:
            fails += 1

    # ---------------------------------------------------------------- stage 1
    print("=" * 72)
    print("1. stage 1 shape")
    print("=" * 72)

    leads = {k: v for k, v in G.LEAD_INS.items() if v}
    lens = {k: len(v.split()) for k, v in leads.items()}
    want(max(lens.values()) - min(lens.values()) <= 2,
         f"lead-ins length-matched within 2 words: {lens}")
    want(G.LEAD_INS["NONE"] == "", "NONE carries no lead-in")
    want("CONTROL" in G.LEAD_INS and G.LEAD_INS["CONTROL"],
         "a signal-free CONTROL lead-in exists")

    q = "How do I decide whether to fix an old appliance or replace it?"
    prompts = {k: G.build_prompt(v, q) for k, v in G.LEAD_INS.items()}
    want(all(p.endswith(q) for p in prompts.values()),
         "the question text is byte-identical and terminal in every condition")
    want(len(set(prompts.values())) == len(prompts),
         "every condition produces a distinct prompt")

    # ---------------------------------------------------------------- blinding
    print()
    print("=" * 72)
    print("2. the judge is blind to the condition")
    print("=" * 72)
    jp = J.JUDGE_TEMPLATE.format(question=q, a="answer one", b="answer two")
    leaked = [k for k, v in leads.items() if v.rstrip(".").lower() in jp.lower()]
    want(not leaked, f"no lead-in text reaches the judge prompt (leaked: {leaked})")
    want("screen reader" not in jp.lower() and "seventy-four" not in jp.lower()
         and "adhd" not in jp.lower(),
         "no disclosure wording reaches the judge prompt")
    want(jp.count(q) == 1, "the judge sees the bare question exactly once")
    orders = {o for _, o, _ in [("x", "chg_A", "y"), ("x", "chg_B", "y")]}
    want(orders == {"chg_A", "chg_B"}, "both orders are defined")

    # ---------------------------------------------------------------- stage 2
    print()
    print("=" * 72)
    print("3. stage 2 statistics on synthetic judgements")
    print("=" * 72)
    J.OUT_DIR = os.path.dirname(os.path.abspath(__file__))
    J.JUDGE_KEY = "synthtest"
    path = J.raw_path()
    make_judged_csv(path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        J.analyse()
    rep = buf.getvalue()
    os.remove(path)

    blk = block_for(rep, "qwen")
    want(bool(blk), "per-answer-model block is produced")

    ident = row_for(blk, "IDENTITY")
    want("control" in ident, "IDENTITY tagged as a control")
    slot = None
    for ln in blk.splitlines():
        if "judge slot bias" in ln:
            for tok in ln.split():
                try:
                    slot = float(tok)
                except ValueError:
                    continue
    want(slot is not None and 1.0 <= slot <= 1.45,
         f"injected judge slot bias recovered near 1.20: {slot}")

    def combined(name):
        ln = row_for(blk, name)
        nums = []
        for t in ln.replace(",", " ").replace("[", " ").replace("]", " ").split():
            try:
                nums.append(float(t))
            except ValueError:
                pass
        return nums[1] if len(nums) >= 2 else float("nan")

    scr, age, adhd = combined("SCREEN"), combined("AGE"), combined("ADHD")
    want(-1.05 <= scr <= -0.75, f"SCREEN found near -0.90: {scr}")
    want(-0.12 <= age <= 0.12, f"AGE null stays near 0: {age}")
    want(-0.50 <= adhd <= -0.20, f"ADHD found near -0.35: {adhd}")
    want(row_for(blk, "SCREEN").rstrip().endswith("*"),
         "SCREEN significant after BH")
    want(not row_for(blk, "AGE").rstrip().endswith("*"),
         "AGE not significant")
    want(row_for(blk, "NONE_vs_CONTROL").count("control") == 1,
         "NONE_vs_CONTROL kept out of the BH family")

    # ---------------------------------------------------------------- mass gate
    print()
    print("=" * 72)
    print("4. letter-mass gate: same margins, SCREEN starved")
    print("=" * 72)
    make_judged_csv(path, starved=("SCREEN",))
    buf = io.StringIO()
    with redirect_stdout(buf):
        J.analyse()
    rep2 = buf.getvalue()
    os.remove(path)
    blk2 = block_for(rep2, "qwen")
    want("too few clean reads" in row_for(blk2, "SCREEN"),
         f"starved SCREEN withheld   [{row_for(blk2, 'SCREEN').strip()}]")
    want("WARNING" in rep2 or "share below" in rep2,
         "judge letter-mass line is printed")

    # ------------------------------------------------- cross-judge analysis
    print()
    print("=" * 72)
    print("5. cross-judge analysis: a starved judge is excluded, not averaged")
    print("=" * 72)
    good = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "method2_judged_by_goodjudge.csv")
    bad = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "method2_judged_by_badjudge.csv")
    make_judged_csv(good, seed=21)
    make_judged_csv(bad, seed=22,
                    starved=tuple(n for n, _, _ in J.COMPARISONS))
    for path_, key in ((good, "goodjudge"), (bad, "badjudge")):
        rows = list(csv.DictReader(open(path_, encoding="utf-8")))
        for r in rows:
            r["judge_key"] = key
            r["judge"] = key
        with open(path_, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)

    argv = sys.argv
    sys.argv = ["analyse_method2.py", good, bad]
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            A.main()
    except SystemExit:
        pass
    sys.argv = argv
    rep5 = buf.getvalue()
    os.remove(good)
    os.remove(bad)

    want("badjudge" in rep5 and "EXCLUDED" in rep5,
         "the starved judge is flagged EXCLUDED")
    want("usable judges: goodjudge" in rep5,
         "only the usable judge is carried into the conclusions")
    want("additivity residual" in rep5,
         "the additivity residual is reported per judge")
    agree = rep5.split("AGREEMENT ACROSS JUDGES")[1]
    scr5 = [l for l in agree.splitlines() if l.startswith("SCREEN")]
    want(scr5 and scr5[0].split()[1] == "3",
         f"SCREEN worse in all 3 cells of the one usable judge   [{scr5[0].strip() if scr5 else ''}]")
    # AGE is a true null here. Over 20 seeds the analysis flagged it in 0 cells
    # on 19 runs and 1 cell on one run: 0.05 of 3 cells, a per-cell false
    # positive rate of 1.7 percent against a 5 percent BH target. So at most one
    # is the correct tolerance, and demanding zero would be demanding luck.
    age5 = [l for l in agree.splitlines() if l.startswith("AGE")]
    want(age5 and int(age5[0].split()[1]) <= 1,
         f"AGE (a true null) flagged in at most 1 of 3 cells   [{age5[0].strip() if age5 else ''}]")

    print()
    print("-" * 72)
    print("stage 2 output under test:")
    print("-" * 72)
    print(rep)

    print("=" * 72)
    print(f"RESULT: {'PASS' if fails == 0 else f'FAIL ({fails} checks failed)'}")
    print("=" * 72)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
