#!/usr/bin/env python3
"""
run_method3_wrapper.py — is the measured effect a property of the signal, or of the
neutral text around it?

WHY THIS EXPERIMENT EXISTS
--------------------------
Three runs measured the same six signals in three different surrounding texts.
The estimates did not agree: on Llama, ADHD came out at +25, then +2, then -2.
Correlation between runs was r=0.31 and r=0.38, neither significant.

That was noticed by accident. This measures it on purpose.

DESIGN
------
Four signal conditions, six neutral wrappers, fully crossed. Twenty-four cells,
each facing the same 200 tasks.

Each wrapper has its OWN baseline, so a signal's effect is measured against a
baseline written in the same style. Six independent estimates of the same
quantity result.

  If the six agree, the effect belongs to the signal.
  If the six disagree as much as the signals differ from each other, the
  measurement is not reporting a property of the signal at all.

WHAT MAKES THIS A FAIR TEST
---------------------------
The six wrappers carry no social signal. They are ordinary neutral openings
that differ in wording, length within a narrow band, and register. Any of them
would be an unremarkable choice for a researcher building this experiment. That
is the point: the question is whether an arbitrary and defensible choice by the
experimenter moves the answer.
"""

import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================================================
MODEL_KEY = "qwen"          # "qwen" then "llama"
# ==========================================================================

MODELS = {
    "qwen":    "Qwen/Qwen2.5-7B-Instruct",
    "llama":   "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}
MODEL = MODELS[MODEL_KEY]

OUT_DIR = "/content/drive/MyDrive/afl/wrapper"
N_TASKS = 200
MAX_NEW_TOKENS = 160
BATCH = 8
LOAD_4BIT = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Six neutral openings. None carries a social signal. Each is a choice an
# experimenter could reasonably make without thinking twice about it.
WRAPPERS: Dict[str, str] = {
    "W1": "I have a question I need some help with today",
    "W2": "There is something I would like to ask you about",
    "W3": "Could you help me with the following, please",
    "W4": "I am trying to work something out and could use a hand",
    "W5": "Quick question for you",
    "W6": "Here is what I need to figure out",
}

# Four signals. The three that moved most in earlier runs, plus none.
SIGNALS: Dict[str, str] = {
    "NONE": "",
    "S01": "I use a screen reader",
    "S04": "I have ADHD",
    "S08": "I am seventy-four",
}

INSTRUCTION = ("Answer with the final answer only. Do not show working. "
               "If the answer is a number, give just the number.")


@dataclass
class Cell:
    id: str
    wrapper: str
    signal: str
    lead_in: str


def build_cells() -> List[Cell]:
    cells = []
    for w_id, w_text in WRAPPERS.items():
        for s_id, s_text in SIGNALS.items():
            lead = w_text if not s_text else f"{w_text}, and {s_text}"
            cells.append(Cell(f"{w_id}_{s_id}", w_id, s_id, lead + "."))
    return cells


CELLS = build_cells()


def norm_number(s: str) -> Optional[float]:
    s = s.replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def check(output: str, key: str, kind: str) -> bool:
    out = output.strip()
    if kind == "number":
        target = norm_number(key)
        if target is None:
            return False
        vals = [norm_number(n) for n in
                re.findall(r"-?\d[\d,]*\.?\d*", out.replace("$", ""))]
        return any(v is not None and abs(v - target) < 1e-6 for v in vals)
    k = key.lower().replace(",", "").strip()
    o = out.lower().replace(",", "").strip()
    if k in o:
        return True
    p = k.split()
    if len(p) == 3 and p[0].isdigit():
        if f"{p[1]} {p[0]} {p[2]}" in o or f"{p[1]} {int(p[0])} {p[2]}" in o:
            return True
    return False


class Runner:
    def __init__(self, name: str):
        print(f"loading {name} on {DEVICE}, 4-bit={LOAD_4BIT}")
        if DEVICE == "cpu":
            print("  WARNING: CPU. Runtime > Change runtime type > T4 GPU.")
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = dict(device_map="auto" if DEVICE == "cuda" else None)
        if LOAD_4BIT and DEVICE == "cuda":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        else:
            kw["dtype"] = torch.float16 if DEVICE == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(name, **kw)
        self.model.eval()
        if DEVICE == "cpu":
            self.model.to(DEVICE)

    def ask(self, prompts: List[str]) -> List[str]:
        chats = [self.tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False,
            add_generation_prompt=True) for p in prompts]
        enc = self.tok(chats, return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                                      do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        return self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                     skip_special_tokens=True)


def raw_path() -> str:
    return os.path.join(OUT_DIR, f"wrapper_{MODEL_KEY}.csv")


def collect() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = json.load(open("tasks.json", encoding="utf-8"))[:N_TASKS]
    path = raw_path()

    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {(r["cell"], r["task_id"]) for r in csv.DictReader(f)}
        print(f"resuming, {len(done)} already saved")

    todo = sum(1 for c in CELLS for t in tasks if (c.id, t["id"]) not in done)
    wc = [len(c.lead_in.split()) for c in CELLS]
    print(f"{len(CELLS)} cells, {todo} generations to do")
    print(f"lead-in length {min(wc)} to {max(wc)} words\n")
    if todo == 0:
        return

    runner = Runner(MODEL)
    new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(["model", "model_key", "cell", "wrapper", "signal",
                    "task_id", "family", "correct", "answer_key", "raw_output"])

    for c in CELLS:
        pending = [t for t in tasks if (c.id, t["id"]) not in done]
        if not pending:
            continue
        n_ok = 0
        for i in range(0, len(pending), BATCH):
            chunk = pending[i:i + BATCH]
            try:
                outs = runner.ask([f"{c.lead_in}\n\n{t['question']}\n\n{INSTRUCTION}"
                                   for t in chunk])
            except Exception as e:
                print(f"  {c.id} batch failed: {e}")
                continue
            for t, o in zip(chunk, outs):
                ok = check(o, t["answer"], t["answer_kind"])
                n_ok += int(ok)
                w.writerow([MODEL, MODEL_KEY, c.id, c.wrapper, c.signal,
                            t["id"], t["family"], int(ok), t["answer"],
                            o.strip().replace("\n", " ")[:400]])
            f.flush()
        print(f"  {c.id:12}{n_ok:4}/{len(pending):<4}{n_ok/len(pending):7.1%}")
    f.close()
    print(f"\nsaved to {path}")


# --------------------------------------------------------------------------

def bh_correct(pvals: List[float]) -> List[float]:
    """Benjamini-Hochberg. Controls the share of false results among those
    called significant, which is the right target when many tests are run and
    none is privileged in advance."""
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, pvals[i] * n / (rank + 1))
        adj[i] = prev
    return list(adj)


def exact_mcnemar(base: Dict[str, int], cond: Dict[str, int]) -> Tuple[int, int, float]:
    """Exact McNemar: a binomial test on discordant pairs. Equivalent to the
    chi-squared form asymptotically, exact at these counts."""
    shared = set(base) & set(cond)
    lost = sum(1 for t in shared if base[t] == 1 and cond[t] == 0)
    gain = sum(1 for t in shared if base[t] == 0 and cond[t] == 1)
    disc = lost + gain
    p = float(stats.binomtest(lost, disc, 0.5).pvalue) if disc else 1.0
    return lost - gain, disc, p


def analyse() -> None:
    path = raw_path()
    if not os.path.exists(path):
        sys.exit(f"no results at {path}")

    data: Dict[str, Dict[str, int]] = {}
    info: Dict[str, Tuple[str, str]] = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            data.setdefault(r["cell"], {})[r["task_id"]] = int(r["correct"])
            info[r["cell"]] = (r["wrapper"], r["signal"])

    sigs = [s for s in SIGNALS if s != "NONE"]
    wraps = list(WRAPPERS)

    print(f"\n{'='*76}\n{MODEL}\nSAME SIGNAL, SIX NEUTRAL WRAPPERS\n{'='*76}")

    # Baseline accuracy per wrapper: how much the wrapper alone moves things.
    print("baseline accuracy by wrapper, no signal at all")
    base_acc = {}
    for w in wraps:
        cell = f"{w}_NONE"
        base_acc[w] = sum(data[cell].values()) / len(data[cell])
        print(f"  {w}  {base_acc[w]:6.1%}   \"{WRAPPERS[w]}\"")
    spread = max(base_acc.values()) - min(base_acc.values())
    print(f"\n  spread across wrappers, no signal present: {spread*100:.1f} points")

    # Effect of each signal, measured six times against its own wrapper baseline.
    print(f"\n{'signal':8}" + "".join(f"{w:>7}" for w in wraps) +
          f"{'mean':>8}{'sd':>7}{'range':>7}")
    print("-" * 76)

    all_p, all_lbl, est = [], [], {}
    for s in sigs:
        row, ps = [], []
        for w in wraps:
            net, disc, p = exact_mcnemar(data[f"{w}_NONE"], data[f"{w}_{s}"])
            row.append(net)
            ps.append(p)
            all_p.append(p)
            all_lbl.append(f"{s} under {w}")
        est[s] = row
        print(f"{s:8}" + "".join(f"{v:+7}" for v in row) +
              f"{np.mean(row):+8.1f}{np.std(row, ddof=1):7.1f}"
              f"{max(row)-min(row):7}")

    adj = bh_correct(all_p)
    n_sig = sum(1 for a in adj if a < 0.05)
    print(f"\n  {len(all_p)} tests, Benjamini-Hochberg corrected: "
          f"{n_sig} below 0.05")
    for lbl, raw, a in zip(all_lbl, all_p, adj):
        if a < 0.05:
            print(f"    {lbl}: raw p={raw:.4f}, corrected {a:.4f}")

    # The comparison the experiment exists for.
    print(f"\n{'='*76}\nWHICH VARIES MORE, THE SIGNAL OR THE WRAPPER?\n{'='*76}")

    within = float(np.mean([np.std(est[s], ddof=1) for s in sigs]))
    between = float(np.std([np.mean(est[s]) for s in sigs], ddof=1))
    print(f"  spread of the SAME signal across six wrappers:   {within:5.1f}")
    print(f"  spread BETWEEN the three signals:                {between:5.1f}")
    print(f"  ratio: {within / between:.2f}" if between > 0 else "")

    if between > 0 and within / between > 1.0:
        print("\n  The same signal varies more across arbitrary neutral wordings\n"
              "  than the signals vary from each other. The measurement is not\n"
              "  reporting a property of the signal.")
    elif between > 0 and within / between > 0.5:
        print("\n  Wrapper variation is a large fraction of signal variation.\n"
              "  Any single-wrapper estimate carries an error bar at least this\n"
              "  wide, which is rarely reported.")
    else:
        print("\n  Signals separate cleanly despite the wrapper. A single-wrapper\n"
              "  estimate is defensible.")

    # Sign stability: does the direction itself survive?
    print("\n  sign stability, out of six wrappers")
    for s in sigs:
        pos = sum(1 for v in est[s] if v > 0)
        neg = sum(1 for v in est[s] if v < 0)
        verdict = "consistent" if pos == 6 or neg == 6 else "FLIPS SIGN"
        print(f"    {s}: {pos} positive, {neg} negative   {verdict}")

    print("""
  A signal whose sign flips depending on the neutral wording around it cannot
  support a claim in either direction, however small the p value in any one
  of the six.
""")


def main() -> None:
    if "--analyse-only" in sys.argv:
        analyse()
    else:
        collect()
        analyse()


if __name__ == "__main__":
    main()
