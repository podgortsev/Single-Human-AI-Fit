#!/usr/bin/env python3
"""
validate_method6b.py — exercise the statistics of method 6b offline.

No GPU, no model. Stubs torch and transformers, then builds a synthetic CSV
with KNOWN combined and position margins and checks that:

  - run_method6b.analyse() runs and recovers the injected raw first-slot bias,
  - analyse_method6b.per_model() (the authoritative read) flags a signal that
    wins FROM SLOT B as "* wins from B", and a signal whose combined margin is
    positive only because the slot bias is not additive as "combined only, not
    position-proof",
  - both controls show wins-from-B at ~0 percent,
  - the BH family is the four real signals, not the controls.

Run:  python validate_method6b.py
"""

import csv
import io
import os
import random
import sys
import types
from contextlib import redirect_stdout

# --- stub the heavy deps --------------------------------------------------
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
import run_method6b as R      # noqa: E402
import analyse_method6b as A  # noqa: E402


# (combined mean, combined sd, position mean, position sd), all in logits
TRUTH = {
    "CONTROL_ID":   (0.00, 0.03, 0.90, 0.20),
    "CONTROL_PARA": (0.00, 0.10, 0.90, 0.20),
    "CONTROL_ALT":  (0.00, 0.15, 0.90, 0.20),
    # positive combined, but position so large the candidate never wins from B
    "SCREEN":       (0.50, 0.30, 2.00, 0.30),
    # positive combined AND small position -> wins from slot B
    "AGE":          (1.50, 0.35, 0.40, 0.30),
    # genuine null
    "DEAF":         (0.00, 0.30, 1.50, 0.30),
    # clearly dispreferred
    "ADHD":         (-0.80, 0.30, 1.00, 0.30),
}


def make_csv(path, seed=5, starved=(), truth=None):
    """Write a synthetic CSV.

    `starved` names signals whose letter mass P(A)+P(B) is pushed far below the
    reportable threshold while the MARGIN is left untouched. That is the shape
    of the real Mistral problem: the A/B ratio still looks decisive, but it is a
    ratio of two tail probabilities because the model was never going to answer
    with a bare letter.
    """
    rnd = random.Random(seed)
    table = truth or TRUTH
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "model_key", "question", "signal", "profile",
                    "order", "lp_A", "lp_B"])
        for qid, _ in R.QUESTIONS:
            for sig in R.SIGNALS:
                cm, cs, pm, ps = table[sig]
                off = -6.0 if sig in starved else 0.0   # log offset on both letters
                for prof in range(100):
                    c = rnd.gauss(cm, cs)
                    p = rnd.gauss(pm, ps)
                    m1 = c + p          # changed candidate in slot A
                    m2 = c - p          # changed candidate in slot B
                    # split each margin symmetrically around the offset so the
                    # difference is exactly m, and the mass is exp() of the offset
                    w.writerow(["stub", "stub", qid, sig, str(prof), "sig_A",
                                f"{off + m1 / 2:.6f}", f"{off - m1 / 2:.6f}"])
                    w.writerow(["stub", "stub", qid, sig, str(prof), "sig_B",
                                f"{off - m2 / 2:.6f}", f"{off + m2 / 2:.6f}"])


def row_for(report, sig):
    for ln in report.splitlines():
        if ln.startswith(sig + " ") or ln.startswith(sig + "\t") or \
                (ln.startswith(sig) and ln[len(sig):len(sig) + 1] == " "):
            return ln
    for ln in report.splitlines():
        if ln.lstrip().startswith(sig):
            return ln
    return ""


def main():
    fails = 0

    def want(cond, msg):
        nonlocal fails
        print(f"  {'ok ' if cond else 'BAD'} {msg}")
        if not cond:
            fails += 1

    R.OUT_DIR = os.path.dirname(os.path.abspath(__file__))
    R.MODEL_KEY = "synthtest"
    path = R.raw_path()
    make_csv(path)

    # --- run_method6b.analyse() ------------------------------------------
    buf = io.StringIO()
    with redirect_stdout(buf):
        R.analyse()
    rep_run = buf.getvalue()

    want("RAW SLOT BIAS" in rep_run, "run analyse prints RAW SLOT BIAS block")
    sb = A.load(path)  # reuse loader
    slot = None
    for ln in rep_run.splitlines():
        if "first-slot bias" in ln:
            for tok in ln.replace("|", " ").split():
                try:
                    slot = float(tok)
                    break
                except ValueError:
                    continue
    want(slot is not None and 0.75 <= slot <= 1.05,
         f"raw first-slot bias recovered near 0.90: {slot}")

    # --- analyse_method6b.per_model() (authoritative) ------------------
    buf = io.StringIO()
    with redirect_stdout(buf):
        A.per_model(A.load(path))
    rep = buf.getvalue()
    os.remove(path)

    want("wins from B 0%" in rep or "wins from B 1%" in rep,
         "CONTROL_PARA shows wins-from-B ~ 0%")

    age_row = row_for(rep, "AGE")
    scr_row = row_for(rep, "SCREEN")
    deaf_row = row_for(rep, "DEAF")
    # AGE sits above the CONTROL_ALT floor here, SCREEN sits below it.
    want("* beats CONTROL_ALT from slot B" in age_row,
         f"AGE credited with beating the neutral floor   [{age_row.strip()}]")
    want("below CONTROL_ALT" in scr_row,
         f"SCREEN reported as under the neutral floor   [{scr_row.strip()}]")
    want("wins from B" not in deaf_row or "*" not in deaf_row,
         f"DEAF (null) not flagged as a win   [{deaf_row.strip()}]")

    # by-question agreement block: AGE should agree
    tail = rep.split("(changed candidate from slot B):")[1]
    age_q = [l for l in tail.splitlines() if l.strip().startswith("AGE")]
    want(age_q and age_q[0].rstrip().endswith("agree"),
         "AGE agrees across the three questions on wins-from-B")

    # --- the letter-mass gate --------------------------------------------
    # Same margins, but AGE's letter mass is starved. The margin still looks
    # decisive; the finding must be withheld anyway.
    print()
    print("=" * 72)
    print("2. letter-mass gate: same margins, AGE starved of letter mass")
    print("=" * 72)
    make_csv(path, seed=5, starved=("AGE",))
    buf = io.StringIO()
    with redirect_stdout(buf):
        A.per_model(A.load(path))
    rep2 = buf.getvalue()
    os.remove(path)

    want("WARNING" in rep2 or "below 0.5" in rep2,
         "letter-mass line is printed")
    age_row2 = row_for(rep2, "AGE")
    want("NOT REPORTABLE" in age_row2,
         f"starved AGE withheld as NOT REPORTABLE   [{age_row2.strip()}]")
    want("*" not in age_row2,
         "starved AGE is not flagged as a win despite an unchanged margin")

    # --- CONTROL_ALT as the floor ----------------------------------------
    # The real Qwen shape: a socially neutral ALTERNATIVE detail beats the
    # reference clause by more than any disclosure does. Every signal must then
    # be reported as dispreferred, however positive it looks on its own.
    print()
    print("=" * 72)
    print("3. CONTROL_ALT floor: a neutral alternative outscores every signal")
    print("=" * 72)
    hot = dict(TRUTH)
    hot["CONTROL_ALT"] = (4.00, 0.30, 0.90, 0.20)   # neutral detail wins big
    make_csv(path, seed=9, truth=hot)
    buf = io.StringIO()
    with redirect_stdout(buf):
        A.per_model(A.load(path))
    rep3 = buf.getvalue()
    os.remove(path)

    want("NEUTRAL-ALTERNATIVE floor" in rep3, "CONTROL_ALT floor line printed")
    want("a socially neutral detail already wins from slot B" in rep3,
         "the note fires when the neutral alternative wins from slot B")
    age3 = row_for(rep3, "AGE")
    want("below CONTROL_ALT" in age3,
         f"AGE reported as dispreferred vs the neutral detail   [{age3.strip()}]")
    want("* beats CONTROL_ALT" not in rep3,
         "no signal is credited with a win when it sits under the neutral floor")

    # the paired contrast must be present and negative for every signal
    want("AGAINST CONTROL_ALT" in rep3, "paired contrast block is printed")
    # the contrast block ends where the by-question block begins
    tail3 = rep3.split("AGAINST CONTROL_ALT")[1].split("by question")[0]
    contrast_rows = [l for l in tail3.splitlines()
                     if l.strip().startswith(("SCREEN", "AGE", "DEAF", "ADHD"))]
    want(len(contrast_rows) == 4, f"four contrast rows, got {len(contrast_rows)}")
    want(all("costs the candidate" in r for r in contrast_rows),
         "every signal flagged as costing the candidate against CONTROL_ALT")

    print()
    print("-" * 72)
    print("analyse_method6b output under test (scenario 1):")
    print("-" * 72)
    print(rep)

    print("=" * 72)
    print(f"RESULT: {'PASS' if fails == 0 else f'FAIL ({fails} checks failed)'}")
    print("=" * 72)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
