#!/usr/bin/env python3
"""
validate_method8.py — exercise the three probes of method 8 offline.

No GPU. Writes synthetic CSVs in the method 1, 3c and 6b schemas with a KNOWN
instability difference injected into ONE signal, then checks that:

  - the probe finds the injected instability, with the right sign,
  - a null signal is left alone,
  - a signal that is MORE stable than the baseline is reported as MORE stable,
  - the letter-mass gate in probe C drops reads that carry no mass on the
    letters,
  - Benjamini-Hochberg is applied across the family.

Run:  python validate_method8.py
"""

import csv
import io
import math
import os
import random
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyse_method8 as A  # noqa: E402

POS = sorted(A.POS_TRAITS)
RNG = random.Random(8)


def write_m1(root):
    """5 phrasings per (signal, profile, trait). SCREEN gets DOUBLE the spread,
    ADHD gets HALF, AGE matches NONE."""
    spread = {"NONE": 1.0, "SCREEN": 2.0, "AGE": 1.0, "DEAF": 1.0, "ADHD": 0.5}
    for m in A.MODELS:
        d = os.path.join(root, m)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"method1_{m}.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "model_key", "prompt_id", "signal", "profile",
                        "trait", "logprob"])
            for sig, sd in spread.items():
                for prof in range(100):
                    for tr in POS:
                        centre = RNG.gauss(-8, 1)
                        for pi in range(5):
                            w.writerow([m, m, f"P{pi}", sig, str(prof), tr,
                                        f"{centre + RNG.gauss(0, sd):.6f}"])


def write_m3c(root):
    """6 wrappers per (signal, task). S01 flips a lot, S04 never, S08 matches."""
    flip = {"NONE": 0.10, "S01": 0.45, "S04": 0.02, "S08": 0.10}
    for m in A.MODELS:
        d = os.path.join(root, m)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"method3_wrapper_{m}.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "model_key", "cell", "wrapper", "signal",
                        "task_id", "family", "correct", "answer_key",
                        "raw_output"])
            for sig, pf in flip.items():
                for task in range(200):
                    base = RNG.random() < 0.5
                    for wi in range(1, 7):
                        c = (not base) if RNG.random() < pf else base
                        w.writerow([m, m, f"W{wi}_{sig}", f"W{wi}", sig,
                                    f"T{task}", "x", "1" if c else "0", "k", "r"])


def write_m6b(root):
    """3 questions per (signal, profile), two orders. AGE margins scatter across
    questions, CONTROL_PARA does not. One model is starved of letter mass."""
    scatter = {"CONTROL_PARA": 0.1, "CONTROL_ID": 0.1, "CONTROL_ALT": 0.1,
               "SCREEN": 0.1, "AGE": 1.2, "DEAF": 0.1, "ADHD": 0.1}
    for m in A.MODELS:
        d = os.path.join(root, m)
        os.makedirs(d, exist_ok=True)
        off = -6.0 if m == "mistral" else 0.0     # starve mistral
        with open(os.path.join(d, f"method6b_{m}.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "model_key", "question", "signal", "profile",
                        "order", "lp_A", "lp_B"])
            for sig, sc in scatter.items():
                for prof in range(100):
                    base = RNG.gauss(0, 0.3)
                    for q in ["HIRE", "PROMOTE", "TRUST"]:
                        margin = base + RNG.gauss(0, sc)
                        w.writerow([m, m, q, sig, str(prof), "sig_A",
                                    f"{off + margin/2:.6f}", f"{off - margin/2:.6f}"])
                        w.writerow([m, m, q, sig, str(prof), "sig_B",
                                    f"{off - margin/2:.6f}", f"{off + margin/2:.6f}"])


def main():
    fails = 0

    def want(cond, msg):
        nonlocal fails
        print(f"  {'ok ' if cond else 'BAD'} {msg}")
        if not cond:
            fails += 1

    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_synth8")
    m1, m3c, m6b = (os.path.join(tmp, x) for x in ("m1", "m3c", "m6b"))
    write_m1(m1)
    write_m3c(m3c)
    write_m6b(m6b)

    argv = sys.argv
    sys.argv = ["analyse_method8.py", "--m1", m1, "--m3c", m3c, "--m6b", m6b]
    buf = io.StringIO()
    with redirect_stdout(buf):
        A.main()
    sys.argv = argv
    rep = buf.getvalue()

    import shutil
    shutil.rmtree(tmp)

    def row(probe_title, model, sig):
        blk = rep.split(probe_title)[1].split("PROBE")[0] if probe_title in rep \
            else rep.split(probe_title)[1] if probe_title in rep else ""
        blk = rep.split(probe_title)[1]
        for ln in blk.splitlines():
            p = ln.split()
            if len(p) >= 2 and p[0] == model and p[1] == sig:
                return ln
        return ""

    a_scr = row("PROBE A", "qwen", "SCREEN")
    a_adhd = row("PROBE A", "qwen", "ADHD")
    a_age = row("PROBE A", "qwen", "AGE")
    want("LESS stable" in a_scr, f"probe A: doubled-spread SCREEN -> LESS stable   [{a_scr.strip()}]")
    want("MORE stable" in a_adhd, f"probe A: halved-spread ADHD -> MORE stable   [{a_adhd.strip()}]")
    want("no effect" in a_age, f"probe A: matched AGE -> no effect   [{a_age.strip()}]")

    b_s01 = row("PROBE B", "qwen", "SCREEN")
    b_s04 = row("PROBE B", "qwen", "ADHD")
    want("LESS stable" in b_s01, f"probe B: high-flip S01 -> LESS stable   [{b_s01.strip()}]")
    want("no effect" in b_s04 or "MORE stable" in b_s04,
         f"probe B: no-flip S04 -> not LESS stable   [{b_s04.strip()}]")

    c_age = row("PROBE C", "qwen", "AGE")
    c_scr = row("PROBE C", "qwen", "SCREEN")
    c_mis = row("PROBE C", "mistral", "AGE")
    want("LESS stable" in c_age, f"probe C: scattered AGE -> LESS stable   [{c_age.strip()}]")
    want("no effect" in c_scr, f"probe C: matched SCREEN -> no effect   [{c_scr.strip()}]")
    want(c_mis.split()[2] == "0",
         f"probe C: starved mistral -> n=0, gated out   [{c_mis.strip()}]")

    want("AGREEMENT" in rep and "of cells" in rep, "agreement table printed")
    want("BH" in rep, "BH column present")

    print()
    print("-" * 72)
    print(rep)
    print("=" * 72)
    print(f"RESULT: {'PASS' if fails == 0 else f'FAIL ({fails} checks failed)'}")
    print("=" * 72)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
