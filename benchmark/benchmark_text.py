"""
Phase 2 - Text Benchmark Script
Runs the full prompt dataset (prompts.py) through all specified models,
logging response time, tokens/sec, RAM usage, and CPU usage into a CSV.

IMPORTANT: This is RESUMABLE. If it crashes or you stop it (Ctrl+C),
just run it again - it will skip combinations already in the results CSV
and continue from where it left off.

Expected runtime: This will take HOURS on a CPU-only laptop, especially
for phi4-mini, qwen3:4b, gemma3:4b, and particularly deepseek-r1:8b.
Run it overnight or while you're not using the laptop for anything heavy.

Usage:
    python benchmark_text.py
"""

import csv
import os
import sys
import time
import psutil

# Allow importing prompts.py from the sibling "dataset" folder
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "dataset"))

from ollama import chat
from prompts import PROMPTS

# ---------------- CONFIG ----------------

# Exact model tags as shown by `ollama list` on your machine
MODELS = [
    "llama3.2:3b",
    "phi4-mini:latest",
    "qwen3:4b",
    "gemma3:4b",
    "deepseek-r1:8b",   # benchmark-only, excluded from live router later
    # "qwen3-vl:4b",    # handled separately with image prompts, see benchmark_vision.py
]

RESULTS_DIR = "../results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "benchmark_results.csv")

FIELDNAMES = [
    "model", "prompt_id", "category", "subcategory",
    "response_time_sec", "ram_before_mb", "ram_after_mb", "ram_delta_mb",
    "cpu_percent", "eval_count", "eval_duration_sec", "tokens_per_sec",
    "response_text", "error"
]

# ---------------- HELPERS ----------------

def load_completed():
    """Return a set of (model, prompt_id) pairs already benchmarked, so we can resume."""
    completed = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add((row["model"], row["prompt_id"]))
    return completed


def ensure_results_file():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def append_result(row: dict):
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)


def run_single_prompt(model: str, prompt_entry: dict) -> dict:
    """Run one prompt through one model, measuring time/RAM/CPU/tokens."""
    prompt_id = prompt_entry["id"]
    category = prompt_entry["category"]
    subcategory = prompt_entry["subcategory"]
    prompt_text = prompt_entry["prompt"]

    process = psutil.Process(os.getpid())
    # NOTE: this measures RAM of THIS python process, not the ollama server process.
    # Ollama runs as a separate process, so for a more accurate picture of the model's
    # own memory footprint, cross-check with Task Manager / `ollama ps` while this runs.
    ram_before = psutil.virtual_memory().used / (1024 * 1024)  # system-wide used RAM in MB
    cpu_before = psutil.cpu_percent(interval=None)

    result_row = {
        "model": model,
        "prompt_id": prompt_id,
        "category": category,
        "subcategory": subcategory,
        "response_time_sec": "",
        "ram_before_mb": round(ram_before, 1),
        "ram_after_mb": "",
        "ram_delta_mb": "",
        "cpu_percent": "",
        "eval_count": "",
        "eval_duration_sec": "",
        "tokens_per_sec": "",
        "response_text": "",
        "error": "",
    }

    try:
        start = time.time()
        response = chat(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            options={"num_predict": 900},   # raised from 400 - enough for full code/reasoning answers
                                             # while still preventing true runaway generation (600s+ outliers)
            think=False,                    # disables hidden reasoning trace for models that support it
                                             # (qwen3, deepseek-r1) - ignored harmlessly by models that don't
        )
        end = time.time()

        elapsed = end - start
        ram_after = psutil.virtual_memory().used / (1024 * 1024)
        cpu_after = psutil.cpu_percent(interval=None)

        # Ollama returns eval_count (tokens generated) and eval_duration (nanoseconds)
        eval_count = response.get("eval_count", None)
        eval_duration_ns = response.get("eval_duration", None)
        eval_duration_sec = (eval_duration_ns / 1e9) if eval_duration_ns else None
        tokens_per_sec = (eval_count / eval_duration_sec) if (eval_count and eval_duration_sec) else None

        response_text = response["message"]["content"].replace("\n", " ").replace("\r", " ")
        # Trim stored response text so the CSV doesn't become unwieldy; full text isn't
        # needed for metrics, just enough to spot-check quality later.
        response_text_trimmed = response_text[:2500]

        result_row.update({
            "response_time_sec": round(elapsed, 2),
            "ram_after_mb": round(ram_after, 1),
            "ram_delta_mb": round(ram_after - ram_before, 1),
            "cpu_percent": cpu_after,
            "eval_count": eval_count if eval_count is not None else "",
            "eval_duration_sec": round(eval_duration_sec, 2) if eval_duration_sec else "",
            "tokens_per_sec": round(tokens_per_sec, 2) if tokens_per_sec else "",
            "response_text": response_text_trimmed,
        })

    except Exception as e:
        result_row["error"] = str(e)

    return result_row


def main():
    ensure_results_file()
    completed = load_completed()

    total_tasks = len(MODELS) * len(PROMPTS)
    done_count = len(completed)
    print(f"Total tasks: {total_tasks} | Already completed: {done_count}")

    task_num = done_count

    for model in MODELS:
        print(f"\n=== Benchmarking model: {model} ===")
        for prompt_entry in PROMPTS:
            key = (model, prompt_entry["id"])
            if key in completed:
                continue  # already done, skip (resumability)

            task_num += 1
            print(f"[{task_num}/{total_tasks}] {model} | {prompt_entry['id']} ({prompt_entry['category']})...", end=" ", flush=True)

            row = run_single_prompt(model, prompt_entry)
            append_result(row)

            if row["error"]:
                print(f"ERROR: {row['error']}")
            else:
                print(f"{row['response_time_sec']}s, {row['tokens_per_sec']} tok/s")

    print("\nBenchmark complete. Results saved to:", RESULTS_FILE)


if __name__ == "__main__":
    main()