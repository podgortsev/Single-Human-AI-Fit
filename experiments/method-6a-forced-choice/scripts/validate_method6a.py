#!/usr/bin/env python3
"""
validate_method6a.py — exercise the statistics of run_method6a.py offline.

No GPU, no model. Stubs torch and transformers, then:

  1. runs parse_choice over a table of answers with known correct readings,
     including the hedges ("A or B, hard to say") that the first version misread;
  2. builds a synthetic result CSV with KNOWN win rates and a KNOWN position
     bias, runs analyse(), and checks that

       - the control comparison recovers the injected slot bias,
       - a real penalty is found and flagged "both orders agree",
       - a null stays null even when one order alone looks like an effect
         (the order swap must cancel the injected position bias),
       - a genuinely order-dependent signal is flagged "ORDER DEPENDENT".

Run:  python validate_method6a.py
"""

import csv
import io
import os
import random
import sys
import types
from contextlib import redirect_stdout

# --- stub the heavy deps so run_method6a imports without a GPU -----------------
_torch = types.ModuleType("torch")
_torch.__version__ = "0.0.0-stub"
_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
_torch.float16 = "float16"
_torch.float32 = "float32"
_torch.Tensor = type("Tensor", (), {})   # scipy's array-api probe looks for this
_torch.no_grad = lambda: types.SimpleNamespace(
    __enter__=lambda *_: None, __exit__=lambda *_: False)
sys.modules.setdefault("torch", _torch)

_tf = types.ModuleType("transformers")
_tf.AutoModelForCausalLM = object
_tf.AutoTokenizer = object
_tf.BitsAndBytesConfig = object
sys.modules.setdefault("transformers", _tf)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_method6a as m6  # noqa: E402


# ============================================================================
# 1. parse_choice
# ============================================================================

PARSE_CASES = [
    ("A", "A"), ("B", "B"), ("A.", "A"), (" B ", "B"),
    ("**A**", "A"), ('"B"', "B"), ("A - stronger record", "A"),
    ("Candidate A", "A"), ("I would hire Candidate B.", "B"),
    ("A because they have more experience", "A"),
    ("The answer is A", "A"), ("B is the better choice here", "B"),
    ("A\nCandidate A is clearly ahead", "A"),
    # hedges and refusals -> no choice
    ("A or B, hard to say", None),
    ("A or B", None),
    ("Either A or B", None),
    ("A or B? It depends on the role.", None),
    ("Both A and B are strong", None),
    ("Both candidates are equally qualified", None),
    ("Neither", None),
    ("Neither A nor B stands out", None),
    ("I cannot choose between them", None),
    ("It's a tie", None),
    ("No preference", None),
    ("They are equal", None),
    ("", None),
    ("I don't know", None),
    ("Sorry, I can't answer that", None),
]


def check_parser() -> int:
    bad = 0
    for text, want in PARSE_CASES:
        got = m6.parse_choice(text)
        if got != want:
            bad += 1
            print(f"  BAD  got {got!r} want {want!r}  | {text!r}")
    print(f"parse_choice: {len(PARSE_CASES) - bad}/{len(PARSE_CASES)} correct")
    return bad


# ============================================================================
# 2. synthetic CSV with known answers
# ============================================================================

# content win rate of the changed candidate (0.5 == no real effect),
# and the per-slot position bias b added to slot A / subtracted from slot B.
TRUTH = {
    "CONTROL": (0.50, 0.12),   # null content, strong slot bias -> control shows it
    "SCREEN":  (0.32, 0.06),   # real penalty, both orders stay below 0.5
    "AGE":     (0.22, 0.05),   # strong real penalty, both orders agree
    "DEAF":    (0.50, 0.14),   # null, but each order alone looks like an effect
    "ADHD":    (0.38, 0.18),   # real penalty AND order dependent: one order > 0.5
}


def make_csv(path: str, seed: int = 7) -> None:
    rnd = random.Random(seed)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "model_key", "question", "signal", "profile",
                    "order", "choice", "signal_won", "raw_output"])
        for qid, _ in m6.QUESTIONS:
            for sig in m6.SIGNALS:
                c, b = TRUTH[sig]
                for prof in range(100):
                    for order in ("sig_first", "sig_second"):
                        p_win = c + b if order == "sig_first" else c - b
                        signal_won = 1 if rnd.random() < p_win else 0
                        # signal candidate is A in sig_first, B in sig_second
                        if order == "sig_first":
                            choice = "A" if signal_won else "B"
                        else:
                            choice = "B" if signal_won else "A"
                        w.writerow(["stub/model", "stub", qid, sig, str(prof),
                                    order, choice, str(signal_won), choice])


def grab(report: str, needle: str) -> str:
    for line in report.splitlines():
        if needle in line:
            return line
    return ""


def check_analyse() -> int:
    m6.OUT_DIR = os.path.dirname(os.path.abspath(__file__))
    m6.MODEL_KEY = "synthtest"
    path = m6.raw_path()
    make_csv(path)

    buf = io.StringIO()
    with redirect_stdout(buf):
        m6.analyse()
    report = buf.getvalue()
    os.remove(path)

    fails = 0

    def want(cond: bool, msg: str) -> None:
        nonlocal fails
        print(f"  {'ok ' if cond else 'BAD'} {msg}")
        if not cond:
            fails += 1

    # control recovers the injected slot bias (~62% to A, significant)
    ctrl_line = grab(report, "candidate A chosen")
    pct = int(ctrl_line.split("(")[1].split("%")[0].split(".")[0]) if "(" in ctrl_line else -1
    want(58 <= pct <= 66, f"control slot bias recovered near 62%: got {pct}% ({ctrl_line.strip()})")
    want("p = " in report and _pval(grab(report, "p = ")) < 0.05,
         "control slot bias is significant")

    # pooled section: SCREEN and AGE are real and agree; DEAF and ADHD content nulls
    pooled = report.split("POOLED OVER THE THREE QUESTIONS")[1]
    scr = _rate(grab(pooled, "SCREEN"))
    age = _rate(grab(pooled, "AGE"))
    deaf = _rate(grab(pooled, "DEAF"))
    adhd = _rate(grab(pooled, "ADHD"))
    want(0.27 <= scr <= 0.37, f"SCREEN pooled rate near 0.32: got {scr}")
    want(0.17 <= age <= 0.28, f"AGE pooled rate near 0.22: got {age}")
    want(0.44 <= deaf <= 0.56, f"DEAF pooled rate near 0.50 (bias cancelled): got {deaf}")
    want(0.33 <= adhd <= 0.43, f"ADHD pooled rate near 0.38: got {adhd}")
    want("all three questions agree" in grab(pooled, "SCREEN"),
         "SCREEN flagged: all three questions agree")

    # per-question table: SCREEN both orders agree, DEAF never significant,
    # ADHD significant but order dependent
    table = report.split("WIN RATE OF THE CHANGED CANDIDATE")[1].split("POOLED")[0]
    scr_rows = [ln for ln in table.splitlines() if "SCREEN" in ln]
    want(scr_rows and all("both orders agree" in ln for ln in scr_rows),
         "SCREEN: every question says 'both orders agree'")
    deaf_rows = [ln for ln in table.splitlines() if "DEAF" in ln]
    want(deaf_rows and not any("ORDER DEPENDENT" in ln or "both orders agree" in ln
                               for ln in deaf_rows),
         "DEAF: no question reaches significance (null survives the swap)")
    adhd_rows = [ln for ln in table.splitlines() if "ADHD" in ln]
    want(adhd_rows and any("ORDER DEPENDENT" in ln for ln in adhd_rows),
         "ADHD: at least one question flagged ORDER DEPENDENT")

    return fails, report


def _pval(line: str) -> float:
    try:
        return float(line.split("p = ")[1].split()[0])
    except Exception:
        return 1.0


def _rate(line: str) -> float:
    for tok in line.replace("%", "").split():
        try:
            v = float(tok)
            if 0 <= v <= 100 and "." in tok:
                return v / 100
        except ValueError:
            continue
    return -1.0


if __name__ == "__main__":
    print("=" * 70)
    print("1. parse_choice")
    print("=" * 70)
    p_bad = check_parser()

    print()
    print("=" * 70)
    print("2. analyse() on synthetic data with known win rates and slot bias")
    print("=" * 70)
    a_bad, rep = check_analyse()

    print()
    print("-" * 70)
    print("analyse() output under test:")
    print("-" * 70)
    print(rep)

    total = p_bad + a_bad
    print("=" * 70)
    print(f"RESULT: {'PASS' if total == 0 else f'FAIL ({total} checks failed)'}")
    print("=" * 70)
    sys.exit(1 if total else 0)
