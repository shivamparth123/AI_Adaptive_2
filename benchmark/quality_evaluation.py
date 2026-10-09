"""
Quality Evaluation Script (LLM-as-a-Judge)
Reads results/benchmark_results.csv, and for every response, asks a judge model
to score it on 5 dimensions. Saves scores to results/quality_results.csv.

IMPORTANT - Judge bias:
This is called "LLM-as-a-Judge" in your report - clearly acknowledge that using
an LLM to judge another LLM's output can introduce bias. Use a model that is NOT
part of your live router pool as the judge, so it isn't scoring itself.
We use qwen3-vl:4b as judge here since it's not in the live router pool - if you'd
rather use a different unbiased model, change JUDGE_MODEL below.

This script is RESUMABLE - same pattern as benchmark_text.py. If it stops,
just run it again; it skips rows already scored.

Usage:
    python quality_evaluation.py
"""

import csv
import json
import os
import re
import sys
import time

import pandas as pd
from ollama import chat

# Allow importing prompts.py from the sibling "dataset" folder
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "dataset"))
from prompts import PROMPTS

# Lookup table: prompt_id -> original prompt text (benchmark_results.csv never
# stored the original question, only the response, so we recover it from here)
PROMPT_LOOKUP = {p["id"]: p["prompt"] for p in PROMPTS}

# Judge model - Qwen3-VL was tried first (since it's not in the live router pool),
# but it ignores the think=False parameter and consistently returns empty responses
# (all generation is consumed by hidden reasoning tokens regardless of the cap).
# Switched to Gemma 3 4B: a straightforward instruction-following model with no
# reasoning-mode complication, so it reliably outputs clean JSON.
#
# NOTE ON BIAS: Gemma 3 4B is one of the 4 models in the live router pool, so it is
# being asked to judge outputs including its own. This is a known limitation of
# LLM-as-a-Judge and should be explicitly disclosed in the report. If time permits,
# validate a random sample (~20-30 responses) with manual human scoring to check
# the judge isn't systematically favoring its own outputs.
JUDGE_MODEL = "gemma3:4b"

BENCHMARK_FILE = os.path.join("..", "results", "benchmark_results.csv")
QUALITY_FILE = os.path.join("..", "results", "quality_results.csv")

FIELDNAMES = [
    "model", "prompt_id", "category",
    "correctness", "relevance", "completeness", "clarity", "instruction_following",
    "overall_quality", "judge_raw_output", "error"
]

JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator. Rate the following AI response to a user query.

USER QUERY:
{prompt}

AI RESPONSE:
{response}

Rate the response on these 5 dimensions, each from 1 (very poor) to 10 (excellent):
- correctness: is the information/answer factually and logically correct?
- relevance: does it actually address what was asked?
- completeness: does it cover what's needed, without being incomplete?
- clarity: is it well-organized and easy to understand?
- instruction_following: did it follow the specific format/instructions implied by the query?

Respond with ONLY a JSON object in this exact format, nothing else, no explanation before or after:
{{"correctness": <number>, "relevance": <number>, "completeness": <number>, "clarity": <number>, "instruction_following": <number>}}
"""


def load_completed():
    completed = set()
    if os.path.exists(QUALITY_FILE):
        with open(QUALITY_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add((row["model"], row["prompt_id"]))
    return completed


def ensure_quality_file():
    os.makedirs(os.path.dirname(QUALITY_FILE), exist_ok=True)
    if not os.path.exists(QUALITY_FILE):
        with open(QUALITY_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def append_result(row: dict):
    with open(QUALITY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)


def extract_json(text: str):
    """Judge models sometimes wrap JSON in markdown fences or add stray text - extract just the JSON object."""
    if not text or not text.strip():
        raise ValueError("Judge returned an EMPTY response (likely hit token limit before writing the JSON answer - "
                          "try raising num_predict further, or the model may not support disabling its reasoning mode)")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge output. Full raw output was: '{text[:400]}'")
    return json.loads(match.group(0))


def evaluate_one(prompt_text: str, response_text: str) -> dict:
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(prompt=prompt_text, response=response_text)
    judge_response = chat(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": judge_prompt}],
        options={"num_predict": 250},   # Gemma has no hidden reasoning trace, so a short
                                         # cap is enough - it just needs to output the JSON object
        think=False,
    )
    raw_output = judge_response["message"]["content"]
    scores = extract_json(raw_output)

    # Validate all 5 keys are present and numeric
    required_keys = ["correctness", "relevance", "completeness", "clarity", "instruction_following"]
    for k in required_keys:
        if k not in scores:
            raise ValueError(f"Judge output missing key '{k}': {raw_output[:200]}")
        scores[k] = float(scores[k])

    overall = sum(scores[k] for k in required_keys) / len(required_keys)
    scores["overall_quality"] = round(overall, 2)
    scores["judge_raw_output"] = raw_output[:300].replace("\n", " ")
    return scores


def main():
    if not os.path.exists(BENCHMARK_FILE):
        print(f"ERROR: {BENCHMARK_FILE} not found. Run benchmark_text.py first.")
        sys.exit(1)

    ensure_quality_file()
    completed = load_completed()

    df = pd.read_csv(BENCHMARK_FILE)
    # Only evaluate rows that succeeded (had no error) and actually have response text
    df = df[df["error"].isna() | (df["error"] == "")]
    df = df[df["response_text"].notna() & (df["response_text"] != "")]

    total = len(df)
    print(f"Total responses to evaluate: {total} | Already done: {len(completed)}")

    task_num = len(completed)

    for _, row in df.iterrows():
        key = (row["model"], row["prompt_id"])
        if key in completed:
            continue

        task_num += 1
        print(f"[{task_num}/{total}] Judging {row['model']} | {row['prompt_id']}...", end=" ", flush=True)

        result_row = {
            "model": row["model"],
            "prompt_id": row["prompt_id"],
            "category": row["category"],
            "correctness": "", "relevance": "", "completeness": "",
            "clarity": "", "instruction_following": "", "overall_quality": "",
            "judge_raw_output": "", "error": "",
        }

        try:
            prompt_text = PROMPT_LOOKUP.get(row["prompt_id"], "")
            if not prompt_text:
                raise ValueError(f"Could not find original prompt text for id {row['prompt_id']}")
            scores = evaluate_one(prompt_text=prompt_text, response_text=row["response_text"])
            result_row.update(scores)
            print(f"overall: {scores['overall_quality']}/10")
        except Exception as e:
            result_row["error"] = str(e)
            print(f"ERROR: {e}")

        append_result(result_row)
        time.sleep(0.2)  # tiny pause, avoids hammering the local server too fast

    print("\nQuality evaluation complete. Results saved to:", QUALITY_FILE)


if __name__ == "__main__":
    main()