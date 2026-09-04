#!/usr/bin/env python3
"""
run_method2_generate.py — STAGE 1 of method 2. Collect the answers.

CHANGE ONLY MODEL_KEY BETWEEN RUNS. Run once per model, three times total.
Stage 2 (run_method2_judge.py) scores what this produces.

WHAT METHOD 2 ASKS
------------------
Methods 3 and 6 asked whether the model gets the answer right, or picks the
person. This asks whether the answer the person GETS is worse, when no answer
is wrong.

Sixty open-ended questions with no single right answer. The question text is
byte-identical in every condition; only a short opening sentence changes. That
is the method 3 design, applied to an outcome that has no key.

WHY THIS IS THE ONLY SUBJECTIVE METHOD
--------------------------------------
"Worse" cannot be checked by string comparison. Something has to judge, and the
only affordable judge is another model. That makes this the one method whose
result depends on an instrument with its own biases. It is reported separately
and labelled, and stage 2 is built to take the judge's biases away from it
rather than trust them.

WHAT IS DELIBERATELY NOT MEASURED HERE
--------------------------------------
Answer LENGTH is recorded but is not the outcome. A longer answer is not a
better one, and if it were the finding, the finding would be about verbosity,
not quality. Length is kept as a covariate so that a quality difference which is
purely a length difference can be spotted and discounted.

CONDITIONS
----------
  NONE      no lead-in at all, the bare question
  CONTROL   a lead-in that carries no social signal, matched for length
  SCREEN    "I use a screen reader"
  AGE       "I am seventy-four"
  ADHD      "I have ADHD"

The three signals are the ones method 3c showed to be measurable, and the ones
whose robustness across neutral wrappers is already known: screen reader and age
survived six wrappers on two models, ADHD changed direction. ADHD is kept
precisely because 3c showed it is unstable; if it moves here too, that is worth
knowing, and if it does not, that agrees with 3c.

CONTROL IS NOT OPTIONAL
-----------------------
Method 3c measured that neutral rewording alone moves a result by two to five
accuracy points. CONTROL is a lead-in of the same length carrying no signal. It
is the floor. A signal that does not clearly beat CONTROL has shown nothing.

COST
----
60 questions x 5 conditions = 300 generations per model, up to 320 new tokens
each. Roughly 25-40 minutes on a free T4, longer than method 3 because the
answers are paragraphs rather than numbers.

USAGE
-----
    !pip install -q transformers accelerate bitsandbytes
    from google.colab import drive; drive.mount('/content/drive')
    # upload questions.json, then:
    %run run_method2_generate.py
"""

import csv
import json
import os
import sys
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================================================
MODEL_KEY = "qwen"          # then "llama", then "mistral"
# ==========================================================================

MODELS = {
    "qwen":    "Qwen/Qwen2.5-7B-Instruct",
    "llama":   "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}
MODEL = MODELS[MODEL_KEY]

OUT_DIR = "/content/drive/MyDrive/afl/method2"
MAX_NEW_TOKENS = 320
BATCH = 4
LOAD_4BIT = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Lead-ins, every one of them exactly six words.
#
# The first draft was not length-matched: "I have ADHD." is three words against
# six for the control. Method 3b measured what that costs: a preamble that
# differs in length moves accuracy on its own, by more than the effects being
# looked for, and the shorter conditions are the ones that benefit. Padding is
# by whole clauses, never cut mid-phrase, which is the other half of that
# lesson. The assertion below makes the match impossible to break by accident.
LEAD_INS: Dict[str, str] = {
    "NONE":    "",
    "CONTROL": "I have a question about something.",
    "SCREEN":  "I use a screen reader daily.",
    "AGE":     "I am seventy-four years old now.",
    "ADHD":    "I have ADHD and use reminders.",
}

_lens = {k: len(v.split()) for k, v in LEAD_INS.items() if v}
assert len(set(_lens.values())) == 1, f"lead-ins not length-matched: {_lens}"


def load_questions() -> List[Dict]:
    if not os.path.exists("questions.json"):
        sys.exit("questions.json not found. Upload experiments/shared/questions/questions.json "
                 "next to this script before running.")
    return json.load(open("questions.json", encoding="utf-8"))


def build_prompt(lead: str, question: str) -> str:
    return f"{lead} {question}".strip() if lead else question


class Generator:
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

    def answer(self, prompts: List[str]) -> List[str]:
        chats = [self.tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False,
            add_generation_prompt=True) for p in prompts]
        enc = self.tok(chats, return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=self.tok.pad_token_id)
        return self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                     skip_special_tokens=True)


def raw_path() -> str:
    return os.path.join(OUT_DIR, f"method2_answers_{MODEL_KEY}.csv")


def collect() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    questions = load_questions()
    path = raw_path()

    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {(r["question_id"], r["condition"]) for r in csv.DictReader(f)}
        print(f"resuming, {len(done)} already saved")

    jobs = []
    for q in questions:
        for cond, lead in LEAD_INS.items():
            if (q["id"], cond) in done:
                continue
            jobs.append((q["id"], q["domain"], cond,
                         build_prompt(lead, q["question"])))

    print(f"{len(questions)} questions x {len(LEAD_INS)} conditions = "
          f"{len(jobs)} generations to do\n")
    if not jobs:
        return

    gen = Generator(MODEL)
    new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(["model", "model_key", "question_id", "domain", "condition",
                    "n_words", "n_chars", "answer"])

    for i in range(0, len(jobs), BATCH):
        chunk = jobs[i:i + BATCH]
        try:
            outs = gen.answer([c[3] for c in chunk])
        except Exception as e:
            print(f"  batch {i // BATCH} failed: {e}")
            continue
        for (qid, domain, cond, _), text in zip(chunk, outs):
            clean = text.strip().replace("\r", " ")
            w.writerow([MODEL, MODEL_KEY, qid, domain, cond,
                        len(clean.split()), len(clean), clean])
        if (i // BATCH) % 10 == 0:
            f.flush()
            print(f"  {i + len(chunk):5} / {len(jobs)}")
    f.close()
    print(f"\nsaved to {path}")


def summarise() -> None:
    """Length and empty-answer counts only. Quality is stage 2's job."""
    path = raw_path()
    if not os.path.exists(path):
        sys.exit(f"no answers at {path}")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    model = rows[0]["model"] if rows else ""

    print(f"\n{'='*72}\n{model}\nSTAGE 1: ANSWERS COLLECTED\n{'='*72}")
    print(f"{'condition':10}{'n':>5}{'mean words':>12}{'median':>9}"
          f"{'empty':>7}{'very short':>12}")
    print("-" * 72)
    for cond in LEAD_INS:
        sub = [r for r in rows if r["condition"] == cond]
        if not sub:
            continue
        lens = sorted(int(r["n_words"]) for r in sub)
        empty = sum(1 for x in lens if x == 0)
        short = sum(1 for x in lens if x < 20)
        med = lens[len(lens) // 2] if lens else 0
        print(f"{cond:10}{len(sub):5}{sum(lens)/len(lens):12.1f}{med:9}"
              f"{empty:7}{short:12}")

    print("""
Length is a covariate here, not the outcome. It is printed so that a quality
difference which turns out to be only a length difference can be spotted in
stage 2 and discounted. An empty or very short answer is a refusal-like event
and is counted separately, the way method 4 counts refusals.

Next: run run_method2_judge.py over the three answer files.""")


def main() -> None:
    if "--summary-only" in sys.argv:
        summarise()
    else:
        collect()
        summarise()


if __name__ == "__main__":
    main()
